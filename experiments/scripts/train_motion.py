import os
from os import path as osp
import csv
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.cuda
import torch.backends.cudnn
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import cv2
import yaml
from tqdm import tqdm
import sacred
from sacred import Experiment
from torch_trainer import TorchTrainer, TrainingConfig, context

from tracktor.motion import Seq2Seq
from tracktor.frcnn_fpn import FRCNN_FPN
from tracktor.reid.resnet import resnet50
from tracktor.tracker import Tracker
from tracktor.oracle_tracker import OracleTracker
from tracktor.datasets.factory import Datasets
from tracktor.utils import plot_sequence, get_mot_accum, evaluate_mot_accums, seed_everything


class EpisodeDataset(Dataset):

    def __init__(
            self,
            data_path,
            cam_path,
            sequences,
            episode_length,
            min_length,
            offset,
            target_length,
            data_mean,
            data_std,
            max_noise=0.,
            flip=0.,
            pad=True,
            augment_target=False,
            include_cam_features=False,
            cam_noise=0.
    ):
        assert isinstance(sequences, list)
        self.episode_length = episode_length
        self.min_length = min_length
        self.offset = offset
        self.target_length = target_length
        self.data_mean = data_mean
        self.data_std = data_std
        self.max_noise = max_noise
        self.flip = flip
        self.pad = pad
        self.augment_target = augment_target
        self.include_cam_features = include_cam_features
        self.cam_noise = cam_noise

        data = []
        for filename in Path(data_path).glob('MOT17-*.txt'):
            with filename.open() as data_file:
                reader = csv.reader(data_file)
                for row in reader:
                    data.append([f'{filename.stem}-{row[1]}', filename.stem] + [float(el) for el in row])

        df = pd.DataFrame(
            data, columns=['track_id', 'sequence', 'frame', 'track', 'x', 'y', 'w', 'h', 'consider', 'class', 'vis'])
        df = df.drop('track', axis=1).reset_index(drop=True)
        df = df[(df['consider'] == 1) & (df['class'] == 1)]
        df = df[df['track_id'].isin(df.groupby(['track_id']).track_id.count()[lambda x: x >= 2].index)]
        df = df.sort_values(['track_id', 'frame'])
        df['x2'] = df['x'] + df['w']
        df['y2'] = df['y'] + df['h']
        df = df.reset_index(drop=True)

        data_cam = []
        for filename in Path(cam_path).glob('MOT17-*-warps.txt'):
            with filename.open() as fh:
                reader = csv.reader(fh)
                for row in reader:
                    data_cam.append([filename.stem[:-6]] + [float(el) for el in row])

        df_cam = pd.DataFrame(data_cam, columns=['sequence', 'frame', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6'])
        df_cam = df_cam.sort_values(['sequence', 'frame'])
        df_cam.loc[df_cam[df_cam['frame'] == 1.0].index, ['c2', 'c3', 'c4', 'c6']] = 0.
        df_cam.loc[df_cam[df_cam['frame'] == 1.0].index, ['c1', 'c5']] = 1.
        df_cam = df_cam.reset_index(drop=True)

        df = df.join(df_cam.set_index(['sequence', 'frame']), on=['sequence', 'frame'])

        self.tracks = []
        self.track_ids = []
        self.track_frames = []

        for seq in sequences:
            sequence = df[df['track_id'].str.startswith(seq)]
            for track_id in sequence['track_id'].unique():
                df_tmp = sequence[sequence['track_id'] == track_id]
                df_tmp = df_tmp[['x', 'y', 'x2', 'y2', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6']]
                df_tmp['go'] = 0.
                df_tmp['no_pad'] = 1.

                self.track_frames.append(sequence[sequence['track_id'] == track_id]['frame'].tolist())
                self.track_ids.append(track_id)
                # FORMAT: [x, y, x2, y2, c1, c2, c3, c4, c5, c6, go, no_pad]
                self.tracks.append(df_tmp.to_numpy().astype('float32'))

        self.index_map = []
        for i, t in enumerate(self.tracks):
            num_data = max((len(t) - self.min_length) // self.offset + 1, 0)
            self.index_map.extend([(i, j * self.offset) for j in range(num_data)])

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        ep_index, data_index = self.index_map[index]
        data = self.tracks[ep_index][data_index:(data_index + self.episode_length)].copy()

        length = torch.randint(self.min_length, len(data) + 1, ())
        data = data[:length]

        if self.flip > np.random.random():
            data_new = np.zeros_like(data)
            data_new[:, :4] = data[:, :4][::-1]
            data_new[:, 11] = 1
            if self.include_cam_features:
                for i in range(1, len(data)):
                    inv = cv2.invertAffineTransform(data[i, 4:10].reshape((2, 3)))
                    data_new[-i, 4:10] = inv.reshape((6,))
            data = data_new

        data = torch.from_numpy(data)

        # cam augmentation
        cam_noise = torch.randn(data.shape[0], 2) * self.cam_noise
        data[:, [6, 9]] += cam_noise

        # augmentation
        widths = data[:, 2] - data[:, 0]
        heights = data[:, 3] - data[:, 1]
        # rel_noise = (torch.rand(len(data), 4) * self.max_noise * 2) - self.max_noise
        rel_noise = torch.randn(data.shape[0], 4) * self.max_noise
        noise = rel_noise * torch.stack([widths, heights, widths, heights], dim=1)
        if not self.augment_target:
            noise[-self.target_length:] = 0.
        data[:, :4] += noise

        # zero out cam features
        if not self.include_cam_features:
            data[:, 4:10] = 0.

        # split
        data_x, data_y = data[:-self.target_length], data[-self.target_length:]

        # padding
        input_length = self.episode_length - self.target_length
        if data_x.shape[0] < input_length:
            pad = torch.zeros(input_length - data_x.shape[0], 12)
            data_x = torch.cat([pad, data_x])

        return data_x, data_y


def areasum(a, b):
    return (a[:, 2] - a[:,0]) * (a[:, 3] - a[:,1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:,1])


def intersection(a, b):
    x = torch.max(a[:, 0], b[:, 0])
    y = torch.max(a[:, 1], b[:, 1])
    w = torch.min(a[:, 2], b[:, 2]) - x
    h = torch.min(a[:, 3], b[:, 3]) - y
    return torch.max(w, torch.tensor(0.).to(w.device)) * torch.max(h, torch.tensor(0.).to(w.device))


def jaccard(ex_box, gt_box):
    ex_box, gt_box = ex_box.view(-1, 4), gt_box.view(-1, 4)
    insec = intersection(ex_box, gt_box)
    uni = areasum(ex_box, gt_box) - insec
    return insec.float() / uni


def evaluate_classes(boxes_before, boxes, boxes_pred):
    assert boxes_before.shape == boxes.shape == boxes_pred.shape
    assert boxes.shape[1] == 4

    classes_iou = jaccard(boxes_before, boxes)

    # class name -> idc mapping
    classes = {
        '[1, 0.75)': classes_iou > 0.75,
        '[0.75, 0.5)': (classes_iou <= 0.75) & (classes_iou > 0.5),
        '[0.5, 0.25)': (classes_iou <= 0.5) & (classes_iou > 0.25),
        '[0.25, 0)': (classes_iou <= 0.25) & (classes_iou > 0),
        '0': classes_iou == 0,
        'all': classes_iou >= 0
    }

    distribution = {cls: (idc.sum().float() / boxes.shape[0]).item() for cls, idc in classes.items()}
    iou_true = {cls: jaccard(boxes[idc], boxes_pred[idc]).mean().item() for cls, idc in classes.items()}
    iou_before = {cls: jaccard(boxes_before[idc], boxes_pred[idc]).mean().item() for cls, idc in classes.items()}
    iou_before_true = {cls: jaccard(boxes_before[idc], boxes[idc]).mean().item() for cls, idc in classes.items()}

    df = pd.DataFrame(
        [distribution, iou_true, iou_before, iou_before_true],
        ['share', 'IoU(true, pred)', 'IoU(before, pred)', 'IoU(before, true)']
    )

    return {'distrib': distribution, 'iou_true': iou_true,
            'iou_before': iou_before, 'iou_before_true': iou_before_true, 'df': df}


class Trainer(TorchTrainer):

    def __init__(self, config, validate_tracktor):
        super().__init__(config)
        self.validate_tracktor = validate_tracktor
        self.mse = nn.MSELoss(reduction='none')
        self.optim = torch.optim.Adam(context.model.parameters(), lr=context.cfg.lr)
        self.sched = ReduceLROnPlateau(self.optim, patience=context.cfg.patience, verbose=True)

    def criterion(self, input, target):
        input = input.view(-1, 4)
        target = target.view(-1, 4)

        if context.cfg.loss == 'iou':
            mask = intersection(input, target) > 0
            if (mask == 0).sum():
                # non-overlapping boxes
                loss = self.mse(input[~mask], target[~mask]).mean() * context.cfg.lmbda
            else:
                loss = torch.tensor(0.).cuda()
            loss += (jaccard(input[mask], target[mask]) * -1).mean()

        elif context.cfg.loss == 'mse':
            loss = self.mse(input, target).mean()

        else:
            raise ValueError()

        return loss

    def epoch(self):
        loss_epoch = []
        iou_epoch = []
        miou_epoch = []

        all_input = []
        all_gt = []
        all_pred_pos = []
        all_prev_pos = []

        for x, target in context.data_loader:
            x, target = x.cuda(), target.cuda()

            _input = torch.zeros(x.shape[0], context.cfg.model_args['input_length'], 12).cuda()
            _input[:, :, :4] = x[:, 1:, :4] - x[:, :-1, :4]  # raises error if model and dataset lengths do not match
            _input[:, :, 4:10] = x[:, 1:, 4:10]
            _input[:, :, 11] = 1.
            _input[(x[:, :, 11] == 0.)[:, :-1]] = 0.

            if not context.validate:
                self.optim.zero_grad()
                do_tf = context.cfg.teacher_forcing > 0 and np.random.uniform() < context.cfg.teacher_forcing
                out = context.model(_input, target, do_tf)
            else:
                out = context.model.predict(_input, target, target.shape[1])

            out[:, :, :4] = out[:, :, :4] * torch.tensor(context.cfg.data_std).cuda()[:4] + torch.tensor(
                context.cfg.data_mean).cuda()[:4]
            assert out.shape[1] == 1

            last_input = x[:, -1, :].unsqueeze(1)
            pred_pos = last_input[:, :, :4] + _input[:, [-1], :4] + out[:, :, :4]

            loss = self.criterion(pred_pos, target[:, :, :4])
            loss_epoch.append(loss.detach().cpu())

            if not context.validate:
                loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.optim.step()

            all_input.append(x.detach().cpu())
            all_gt.append(target[:, :, :4].detach().cpu())
            all_pred_pos.append(pred_pos.detach().cpu())
            all_prev_pos.append(torch.cat([
                last_input[:, :, :4].detach().cpu(),
                target[:, :-1, :4].detach().cpu()
            ], dim=1))

            # evaluate iou
            iou = jaccard(pred_pos.view(-1, 4).detach(), target[:, :, :4].view(-1, 4).detach())
            iou = iou[~torch.isnan(iou)]
            iou_epoch.append(iou)
            miou_epoch.append((iou > 0.7).sum().float() / len(iou))

        all_input = torch.cat(all_input)  # e.g [94920, 6, 12]
        all_gt = torch.cat(all_gt)
        all_pred_pos = torch.cat(all_pred_pos)
        all_prev_pos = torch.cat(all_prev_pos)

        assert all_prev_pos.shape[1] == all_gt.shape[1] == all_pred_pos.shape[1] == 1
        eval_df = evaluate_classes(all_prev_pos.squeeze(1), all_gt.squeeze(1), all_pred_pos.squeeze(1))['df']

        # calculate cva performance for current epoch
        diff = all_input[:, 1:, :4] - all_input[:, :-1, :4]
        m = (all_input[:, :, 11] == 1.)[:, :-1].unsqueeze(1).float()
        v_mean = torch.bmm(m, diff) / m.sum(dim=2).unsqueeze(2)
        # set NaNs to zero (https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/4)
        v_mean[v_mean != v_mean] = 0.
        v_mean = v_mean.squeeze(1)

        pred_cva = all_input[:, -1, :4] + v_mean
        iou_cva = jaccard(pred_cva, all_gt.squeeze(1)).mean()
        miou_cva = (jaccard(pred_cva, all_gt.squeeze(1)) > 0.7).sum().float() / len(pred_cva)
        loss_cva = self.criterion(pred_cva, all_gt.squeeze(1))

        if context.validate:
            loss_epoch = self.criterion(all_pred_pos.squeeze(1)[:, :4], all_gt.squeeze(1)).mean()
            iou = jaccard(all_pred_pos.squeeze(1)[:, :4], all_gt.squeeze(1))
            iou_epoch = iou.mean()
            miou_epoch = ((iou > 0.7).sum().float() / len(iou))
            with open(context.log_path / f'{context.epoch}_df_val.txt', 'w') as fh:
                fh.write(eval_df.to_string())

            self.sched.step(loss_epoch, context.epoch)
        else:
            loss_epoch = torch.tensor(loss_epoch).mean()
            iou_epoch = torch.cat(iou_epoch).mean()
            miou_epoch = torch.stack(miou_epoch).float().mean()
            with open(context.log_path / f'{context.epoch}_df_train.txt', 'w') as fh:
                fh.write(eval_df.to_string())

        metrics = {
            'loss': loss_epoch,
            'iou': iou_epoch,
            'miou': miou_epoch,
            'iou_cva': iou_cva,
            'miou_cva': miou_cva,
            'loss_cva': loss_cva
        }

        if context.validate and context.epoch % context.cfg.tracktor_val_every == 0:
            metrics = {**metrics, **self.validate_tracktor(context.model, context.epoch)}

        return metrics


ex = Experiment()
ex.add_config('experiments/cfgs/tracktor.yaml')
ex.add_config('experiments/cfgs/train_motion.yaml')
ex.add_config(ex.configurations[0]._conf['tracktor']['reid_config'])


@ex.automain
def main(tracktor, siamese, train, _config, _log, _run):
    # instantiate tracktor
    sacred.commands.print_config(_run)

    # set all seeds
    seed_everything(tracktor['seed'])

    output_dir = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', 'output', 'motion', train['name']))
    sacred_config = osp.join(output_dir, 'sacred_config.yaml')

    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    with open(sacred_config, 'w') as outfile:
        yaml.dump(_config, outfile, default_flow_style=False)

    # object detection
    _log.info("Initializing object detector.")

    obj_detect: nn.Module = FRCNN_FPN(num_classes=2)
    obj_detect_state_dict = torch.load(_config['tracktor']['obj_detect_model'])
    obj_detect.load_state_dict(obj_detect_state_dict)

    obj_detect.eval()
    obj_detect.cuda()

    # reid
    reid_network = resnet50(pretrained=False, **siamese['cnn'])
    reid_network.load_state_dict(torch.load(tracktor['reid_weights']))
    reid_network.eval()
    reid_network.cuda()

    # tracktor
    if 'oracle' in tracktor:
        assert False, "No motion network specified"
        # tracker = OracleTracker(obj_detect, reid_network, tracktor['tracker'], tracktor['oracle'])
    else:
        tracker = Tracker(obj_detect, reid_network, None, tracktor['tracker'],
                          tracktor['motion'], train['datasets']['train']['min_length'])

    assert train['datasets']['train']['min_length'] == train['datasets']['val']['min_length']
    assert train['datasets']['train']['episode_length'] == train['datasets']['val']['episode_length']
    assert train['datasets']['train']['offset'] == train['datasets']['val']['offset']
    assert train['loss'] in ['mse', 'iou']

    config = TrainingConfig(
        data=[EpisodeDataset, EpisodeDataset],
        data_args=[train['datasets']['train'], train['datasets']['val']],
        model=Seq2Seq,
        model_args=train['model'],
        batch_size=train['batch_size'],
        epochs=train['epochs'],
        log_path=output_dir,
        primary_metric='miou',
        smaller_is_better=False,
        save_every=train['save_every'],
        num_workers=train['num_workers'],
        lr=train['lr'],
        patience=train['patience'],
        data_mean=train['data_mean'],
        data_std=train['data_std'],
        shuffle=train['shuffle'],
        teacher_forcing=train['teacher_forcing'],
        loss=train['loss'],
        lmbda=train['lmbda'],
        tracktor_val_every=train['tracktor_val_every']
    )

    def validate_tracktor(motion_network, epoch):
        # inject current network into tracker
        tracker.motion_network = motion_network

        time_total = 0
        num_frames = 0
        mot_accums = []
        dataset = Datasets(train['tracktor_val_dataset'])
        for seq in dataset:
            tracker.reset()

            start = time.time()

            _log.info(f"Tracking: {seq}")

            data_loader = DataLoader(seq, batch_size=1, shuffle=False)
            for i, frame in enumerate(tqdm(data_loader)):
                if len(seq) * tracktor['frame_split'][0] <= i <= len(seq) * tracktor['frame_split'][1]:
                    tracker.step(frame)
                    num_frames += 1
            results = tracker.get_results()

            time_total += time.time() - start

            _log.info(f"Tracks found: {len(results)}")
            _log.info(f"Runtime for {seq}: {time.time() - start :.1f} s.")

            if seq.no_gt:
                _log.info(f"No GT data for evaluation available.")
            else:
                mot_accums.append(get_mot_accum(results, seq))

            _log.info(f"Writing predictions to: {output_dir}")
            seq.write_results(results, output_dir)

            if tracktor['write_images']:
                plot_sequence(results, seq, osp.join(output_dir, tracktor['dataset'], str(epoch), str(seq)))

        _log.info(f"Tracking runtime for all sequences (without evaluation or image writing): "
                  f"{time_total:.1f} s ({num_frames / time_total:.1f} Hz)")

        metrics = {}
        if mot_accums:
            summary = evaluate_mot_accums(
                mot_accums,
                [str(s) for s in dataset if not s.no_gt],
                generate_overall=True,
                return_summary=True,
                metrics=train['tracktor_val_metrics']
            )
            metrics = {m: summary.loc['OVERALL', m] for m in train['tracktor_val_metrics']}

        return metrics

    # re-seed
    seed_everything(tracktor['seed'])

    trainer = Trainer(config, validate_tracktor)
    trainer.train()
