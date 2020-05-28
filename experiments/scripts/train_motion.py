import os
from os import path as osp
import time

import numpy as np
import torch
import torch.cuda
import torch.backends.cudnn
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
from torchvision.models.detection._utils import BoxCoder
from tqdm import tqdm
import sacred
from sacred import Experiment

from tracktor.motion import Seq2Seq
from tracktor.frcnn_fpn import FRCNN_FPN
from tracktor.reid.resnet import resnet50
from tracktor.tracker import Tracker
from tracktor.datasets.factory import Datasets
from tracktor.datasets.episodes import EpisodeDataset, EpisodeImageDataset
from tracktor.utils import \
    plot_sequence, get_mot_accum, evaluate_mot_accums, seed_everything, jaccard, evaluate_classes
from tracktor.motion.trainer import TorchTrainer, TrainingConfig, context


class Trainer(TorchTrainer):

    def __init__(self, config, validate_tracktor):
        super().__init__(config)
        self.validate_tracktor = validate_tracktor
        self.mse = nn.MSELoss(reduction='none')
        self.optim = torch.optim.Adam(context.model.parameters(), lr=context.cfg.lr)
        self.sched = ReduceLROnPlateau(self.optim, patience=context.cfg.patience, verbose=True)

        if context.cfg.use_box_coding:
            self.use_box_coding = True
            self.box_coder = BoxCoder(context.cfg.box_coding_weights)
        else:
            self.use_box_coding = False

    def criterion(self, input, target):
        input = input.view(-1, 4)
        target = target.view(-1, 4)
        assert context.cfg.loss == 'mse'
        return self.mse(input, target).mean()

    def criterion_coded(self, prediction, x, target, last_coded):
        target_coded = self.box_coder.encode(list(target[:, :, :4]), list(x[:, [-1], :4]))
        last_coded = last_coded[:, [-1], :4]
        loss = self.criterion(prediction[:, :, :4], torch.stack(target_coded) - last_coded)
        return loss

    def epoch(self):
        loss_epoch = []
        loss_1_epoch = []
        loss_2_epoch = []
        loss_3_epoch = []
        loss_4_epoch = []
        loss_5_epoch = []
        loss_6_epoch = []
        iou_epoch = []
        miou_epoch = []

        all_x = []
        all_out = []
        all_input = []
        all_gt = []
        all_pred_pos = []
        all_prev_pos = []

        for x, target, length in context.data_loader:
            x, target = x.cuda(), target.cuda()

            _input = torch.zeros(x.shape[0], context.cfg.model_args['input_length'], 6).cuda()
            # _input[:, :, :4] = x[:, 1:, :4] - x[:, :-1, :4]  # raises error if model and dataset lengths do not match
            # _input[:, :, 5] = 1.
            # _input[(x[:, :, 5] == 0.)[:, :-1]] = 0.

            if self.use_box_coding:
                encoded = self.box_coder.encode(list(x[:, 1:, :4]), list(x[:, :-1, :4]))
                _input[:, :, :4] = torch.stack(encoded, dim=0)
            else:
                # raises error if model and dataset lengths do not match
                _input[:, :, :4] = x[:, 1:, :4] - x[:, :-1, :4]
            _input[:, :, 5] = 1.
            _input[(x[:, :, 5] == 0.)[:, :-1]] = 0.

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

            if self.use_box_coding:
                # out is the acceleration in encoding space
                last_offset = _input[:, [-1], :4]
                pred_offset = last_offset + out[:, :, :4]
                pred_pos = self.box_coder.decode(list(pred_offset), list(last_input))
            else:
                pred_pos = last_input[:, :, :4] + _input[:, [-1], :4] + out[:, :, :4]

            # calculate loss
            if self.use_box_coding:
                # target_coded = self.box_coder.encode(list(target[:, :, :4]), list(x[:, [-1], :4]))
                last_coded = _input[:, [-1], :4]
                # loss = self.criterion(out[:, :, :4], torch.stack(target_coded) - last_coded)
                loss = self.criterion_coded(out, x, target, last_coded)
            else:
                loss = self.criterion(pred_pos, target[:, :, :4])

            loss_epoch.append(loss.detach().cpu())

            # DIFFERENT LENGTH ANALYSIS
            loss_lists = [loss_1_epoch, loss_2_epoch, loss_3_epoch, loss_4_epoch, loss_5_epoch, loss_6_epoch]
            for i, loss_list in zip(range(1, 7), loss_lists):
                mask = length == i
                if mask.any():
                    if self.use_box_coding:
                        loss_part = self.criterion_coded(out[mask], x[mask], target[mask], _input[mask])
                    else:
                        loss_part = self.criterion(pred_pos[mask], target[:, :, :4][mask])
                    loss_list.append(loss_part)

            if not context.validate and context.epoch > 0:
                loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.optim.step()

            all_x.append(x.detach().cpu())
            all_out.append(out.detach().cpu())
            all_input.append(_input.detach().cpu())
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

        all_x = torch.cat(all_x)  # e.g [94920, 6, 12]
        all_out = torch.cat(all_out)
        all_input = torch.cat(all_input)
        all_gt = torch.cat(all_gt)
        all_pred_pos = torch.cat(all_pred_pos)
        all_prev_pos = torch.cat(all_prev_pos)

        assert all_prev_pos.shape[1] == all_gt.shape[1] == all_pred_pos.shape[1] == 1
        eval_df = evaluate_classes(all_prev_pos.squeeze(1), all_gt.squeeze(1), all_pred_pos.squeeze(1))['df']

        # calculate cva performance for current epoch
        diff = all_x[:, 1:, :4] - all_x[:, :-1, :4]
        m = (all_x[:, :, 5] == 1.)[:, :-1].unsqueeze(1).float()
        v_mean = torch.bmm(m, diff) / m.sum(dim=2).unsqueeze(2)
        # set NaNs to zero (https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/4)
        v_mean[v_mean != v_mean] = 0.
        v_mean = v_mean.squeeze(1)

        pred_cva = all_x[:, -1, :4] + v_mean
        val_mask = ((pred_cva[:, 2] - pred_cva[:, 0]) >= 0) & ((pred_cva[:, 3] - pred_cva[:, 1]) >= 0)

        iou_cva = jaccard(pred_cva, all_gt.squeeze(1)).mean()
        miou_cva = (jaccard(pred_cva, all_gt.squeeze(1)) > 0.7).sum().float() / len(pred_cva)
        if self.use_box_coding:
            offset_cva = self.box_coder.encode(list(pred_cva[val_mask].unsqueeze(1)[:, :, :4]), list(all_x[val_mask][:, [-1], :4]))
            coded_cva = torch.stack(offset_cva) - all_input[val_mask][:, [-1], :4]
            loss_cva = self.criterion_coded(coded_cva, all_x[val_mask], all_gt[val_mask], all_input[val_mask])
        else:
            loss_cva = self.criterion(pred_cva, all_gt.squeeze(1))

        if context.validate:
            if self.use_box_coding:
                loss_epoch = self.criterion_coded(all_out, all_x, all_gt, all_input)
            else:
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

        loss_1_epoch = torch.tensor(loss_1_epoch).mean()
        loss_2_epoch = torch.tensor(loss_2_epoch).mean()
        loss_3_epoch = torch.tensor(loss_3_epoch).mean()
        loss_4_epoch = torch.tensor(loss_4_epoch).mean()
        loss_5_epoch = torch.tensor(loss_5_epoch).mean()
        loss_6_epoch = torch.tensor(loss_6_epoch).mean()

        metrics = {
            'loss': loss_epoch,
            'iou': iou_epoch,
            'miou': miou_epoch,
            'iou_cva': iou_cva,
            'miou_cva': miou_cva,
            'loss_cva': loss_cva,
            'loss_1': loss_1_epoch,
            'loss_2': loss_2_epoch,
            'loss_3': loss_3_epoch,
            'loss_4': loss_4_epoch,
            'loss_5': loss_5_epoch,
            'loss_6': loss_6_epoch
        }

        if context.validate and context.epoch % context.cfg.tracktor_val_every == 0:
            metrics = {**metrics, **self.validate_tracktor(context.model, context.epoch)}

        return metrics


ex = Experiment()
ex.add_config('experiments/cfgs/tracktor.yaml')
ex.add_config('experiments/cfgs/train_motion.yaml')
ex.add_config(ex.configurations[0]._conf['tracktor']['reid_config'])


@ex.automain
def main(tracktor, reid, train, _config, _log, _run):
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
    reid_network = resnet50(pretrained=False, **reid['cnn'])
    reid_network.load_state_dict(torch.load(tracktor['reid_weights']))
    reid_network.eval()
    reid_network.cuda()

    # tracktor
    if 'oracle' in tracktor:
        assert False, "No motion network specified"
        # tracker = OracleTracker(obj_detect, reid_network, tracktor['tracker'], tracktor['oracle'])
    else:
        tracker = Tracker(obj_detect, reid_network, None, tracktor['tracker'],
                          tracktor['motion'], train['dataset_args']['train']['min_length'])

    assert train['dataset_args']['train']['min_length'] == train['dataset_args']['val']['min_length']
    assert train['dataset_args']['train']['episode_length'] == train['dataset_args']['val']['episode_length']
    assert train['dataset_args']['train']['offset'] == train['dataset_args']['val']['offset']
    assert train['loss'] in ['mse', 'iou']

    config = TrainingConfig(
        data=[eval(train['dataset'])] * 2,
        data_args=[train['dataset_args']['train'], train['dataset_args']['val']],
        model=eval(tracktor['motion']['model']),
        model_args=tracktor['motion']['model_args'],
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
        tracktor_val_every=train['tracktor_val_every'],
        use_box_coding=tracktor['motion']['use_box_coding'],
        box_coding_weights=tracktor['motion']['box_coding_weights'],
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
