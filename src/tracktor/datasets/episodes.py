import csv
from pathlib import Path
import math

import configparser
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision.models.detection.transform import resize_boxes
from torchvision.transforms import ToTensor

from tracktor.utils import get_center, get_width, get_height, make_pos, jaccard


def infer_scale(feature, original_size):
    # assumption: the scale is of the form 2 ** (-k), with k integer
    size = feature.shape[-2:]
    possible_scales = []
    for s1, s2 in zip(size, original_size):
        approx_scale = float(s1) / float(s2)
        scale = 2 ** torch.tensor(approx_scale).log2().round().item()
        possible_scales.append(scale)
    assert possible_scales[0] == possible_scales[1]
    return possible_scales[0]


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
            cam_features_1_only=False,
            cam_noise=[0., 0., 0., 0., 0., 0.]
    ):
        assert isinstance(sequences, list)
        assert isinstance(cam_noise, list)
        assert len(cam_noise) == 6

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
        self.cam_features_1_only = cam_features_1_only
        self.cam_noise = torch.tensor(cam_noise)

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

        # apply cmc on positions
        # for i in range(len(data)):
        #    # apply warp from position i to all prior positions
        #    warp = data[i, 4:10].view((2, 3))
        #    for j in range(i):
        #        data[j, :4] = warp_pos(data[j, :4].unsqueeze(0), warp)

        # cam augmentation
        cam_noise = torch.randn(data.shape[0], 6) * self.cam_noise
        data[:, 4:10] += cam_noise

        # augmentation
        widths = data[:, 2] - data[:, 0]
        heights = data[:, 3] - data[:, 1]
        # rel_noise = (torch.rand(len(data), 4) * self.max_noise * 2) - self.max_noise
        rel_noise = torch.randn(data.shape[0], 4) * self.max_noise
        noise = rel_noise * torch.stack([widths, heights, widths, heights], dim=1)
        # noise += cam_noise[:, [2, 5]].repeat(1, 2)
        if not self.augment_target:
            noise[-self.target_length:] = 0.
        data[:, :4] += noise

        # zero out cam features
        if not self.include_cam_features:
            data[:, 4:10] = 0.

        if self.cam_features_1_only and length - self.target_length != 1:
            data[:, 4:10] = 0.

        # split
        data_x, data_y = data[:-self.target_length], data[-self.target_length:]

        # padding
        input_length = self.episode_length - self.target_length
        if data_x.shape[0] < input_length:
            pad = torch.zeros(input_length - data_x.shape[0], 12)
            data_x = torch.cat([pad, data_x])

        return data_x, data_y, length - 1


class EpisodeImageDataset(Dataset):

    def __init__(
            self,
            data_path,
            image_features_path,
            sequences,
            episode_length,
            min_length,
            offset,
            target_length,
            vis_threshold,
            max_noise=0.,
            augment_target=False,
            skip_prob=0.,
            skip_n_min=0,
            skip_n_max=0,
            fix_skip_n_per_ep=True,
            cam_motion_prob=0.,
            cam_motion_all_seqs=False,
            cam_motion_all_frames=False,
            cam_motion_cont_prob=0.,
            flip_prob=0.,
            flipped_features_path=None,
            jitter=False,
            mmap=False
    ):
        assert isinstance(sequences, list)

        self.data_path = Path(data_path)
        self.image_features_path = Path(image_features_path)
        self.episode_length = episode_length
        self.min_length = min_length
        self.offset = offset
        self.target_length = target_length
        self.vis_threshold = vis_threshold
        self.max_noise = max_noise
        self.augment_target = augment_target
        self.skip_prob = skip_prob
        self.skip_n_min = skip_n_min
        self.skip_n_max = skip_n_max
        self.fix_skip_n_per_ep = fix_skip_n_per_ep
        self.cam_motion_prob = cam_motion_prob
        self.cam_motion_all_seqs = cam_motion_all_seqs
        self.cam_motion_all_frames = cam_motion_all_frames
        self.cam_motion_cont_prob = cam_motion_cont_prob
        self.flip_prob = flip_prob
        self.flipped_features_path = Path(flipped_features_path) if flipped_features_path else None
        self.jitter = jitter
        self.mmap_mode = 'r' if mmap else None

        self.train_seqs = [d.stem for d in (self.data_path / 'train').iterdir() if d.is_dir()]
        self.test_seqs = [d.stem for d in (self.data_path / 'test').iterdir() if d.is_dir()]
        self.transforms = ToTensor()
        self.image_features = {}
        self.image_sizes = {}
        self.original_image_sizes = {}
        self.flipped_features = {}

        positions = []
        for seq in sequences:
            # load images into memory
            if seq in self.train_seqs:
                seq_path = self.data_path / 'train' / seq
            elif seq in self.test_seqs:
                seq_path = self.data_path / 'test' / seq
            else:
                raise ValueError(f'Sequence {seq} not found.')

            config_file = seq_path / 'seqinfo.ini'
            assert config_file.exists(), f'Config file does not exist: {config_file}'

            config = configparser.ConfigParser()
            config.read(config_file)
            gt_file = seq_path / 'gt' / 'gt.txt'

            with gt_file.open() as fh:
                reader = csv.reader(fh)
                break_counter = 0
                for row in reader:
                    if float(row[8]) < self.vis_threshold:
                        break_counter += 1
                        continue
                    positions.append([f'{seq}-{row[1]}-{break_counter}', seq] + [float(el) for el in row])

            self.image_features[seq] = np.load(self.image_features_path / f'{seq}-features.npy',
                                               mmap_mode=self.mmap_mode)
            self.image_sizes[seq] = np.load(self.image_features_path / f'{seq}-sizes.npy')
            self.original_image_sizes[seq] = np.load(self.image_features_path / f'{seq}-origsizes.npy')

            if self.flipped_features_path is not None:
                self.flipped_features[seq] = np.load(self.flipped_features_path / f'{seq}-features.npy',
                                                     mmap_mode=self.mmap_mode)

        df = pd.DataFrame(
            positions,
            columns=['track_id', 'sequence', 'frame', 'track', 'x', 'y', 'w', 'h', 'consider', 'class', 'vis'])
        df = df.drop('track', axis=1).reset_index(drop=True)
        df = df[(df['consider'] == 1) & (df['class'] == 1)]
        df = df[df['track_id'].isin(df.groupby(['track_id']).track_id.count()[lambda x: x >= 2].index)]
        df = df.sort_values(['track_id', 'frame'])
        df['x2'] = df['x'] + df['w']
        df['y2'] = df['y'] + df['h']
        df = df.reset_index(drop=True)

        self.tracks = []
        self.track_ids = []
        self.track_frames = []

        for track_id in df['track_id'].unique():
            df_tmp = df[df['track_id'] == track_id]
            df_tmp = df_tmp[['x', 'y', 'x2', 'y2']]
            df_tmp['go'] = 0.
            df_tmp['no_pad'] = 1.
            self.track_frames.append(df[df['track_id'] == track_id]['frame'].astype(int).tolist())
            self.track_ids.append(track_id)
            self.tracks.append(df_tmp.to_numpy().astype('float32'))

        self.index_map = []
        for i, t in enumerate(self.tracks):
            num_data = max((len(t) - self.min_length) // self.offset + 1, 0)
            self.index_map.extend([(i, j * self.offset) for j in range(num_data)])

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        ep_index, data_index = self.index_map[index]
        seq = self.track_ids[ep_index][:8]
        do_skip = torch.rand(()) < self.skip_prob

        if do_skip:
            max_len = self.episode_length + (self.episode_length - 1) * self.skip_n_max
        else:
            max_len = self.episode_length

        data = self.tracks[ep_index][data_index:(data_index + max_len)].copy()
        length = torch.randint(self.min_length, min(self.episode_length, len(data)) + 1, ()).item()

        max_skip = 0
        if do_skip:
            max_skip = min(math.floor((len(data) - length) / (length - 1)), self.skip_n_max)

        if self.fix_skip_n_per_ep:
            gaps = torch.randint(self.skip_n_min, max_skip + 1, (1,)).repeat(length - 1)
        else:
            gaps = torch.randint(self.skip_n_min, max_skip + 1, (length - 1,))

        data_idc = torch.cumsum(torch.cat([torch.tensor([data_index]), gaps + 1]), 0)  # +1 to account for the selected elems
        data = torch.from_numpy(self.tracks[ep_index][data_idc].copy())
        assert data.shape[0] == length

        # cam motion injection
        feat_translation = torch.zeros(data.shape[0], 2)

        if self.cam_motion_all_seqs:
            allowed_seqs = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-10', 'MOT17-11']
        else:
            allowed_seqs = ['MOT17-02', 'MOT17-04']

        if torch.rand(()) < self.cam_motion_prob and seq in allowed_seqs and data[0, 3] - data[0, 1] < 125:
            if self.cam_motion_all_frames:
                w = data[0, 2] - data[0, 0]
                h = data[0, 3] - data[0, 1]
                dx_cam = torch.FloatTensor(data.shape[0]).uniform_(-w, w)
                dy_cam = torch.FloatTensor(data.shape[0]).uniform_(-(2 * h) / 3, (2 * h) / 3)
                # apply delta to all positions starting from idx
                data[:, :4] += torch.stack([dx_cam, dy_cam, dx_cam, dy_cam], dim=1)
                feat_translation[:, :] = torch.stack([dx_cam, dy_cam], dim=1)
            else:
                idx_cam = torch.randint(1, data.shape[0], ())
                w = data[idx_cam, 2] - data[idx_cam, 0]
                h = data[idx_cam, 3] - data[idx_cam, 1]
                dx_cam = torch.FloatTensor(1).uniform_(-w, w)
                dy_cam = torch.FloatTensor(1).uniform_(-(2 * h) / 3, (2 * h) / 3)
                # apply delta to all positions starting from idx
                data[idx_cam:, :4] += torch.tensor([dx_cam, dy_cam, dx_cam, dy_cam])
                feat_translation[idx_cam:, :] = torch.tensor([dx_cam, dy_cam])

        # independently from above cam motion injection, inject continuous cam motion
        if torch.rand(()) < self.cam_motion_cont_prob and seq in allowed_seqs and data[0, 3] - data[0, 1] < 125:
            # inject linear motion, starting at a random position
            idx_cam = torch.randint(1, data.shape[0], ())
            w = data[idx_cam, 2] - data[idx_cam, 0]
            h = data[idx_cam, 3] - data[idx_cam, 1]
            dx_cam = torch.FloatTensor(1).uniform_(-w, w)
            dy_cam = torch.FloatTensor(1).uniform_(-(h / 3), h / 3)
            # continously apply delta to all positions starting from idx
            data[idx_cam:, :4] += torch.tensor([dx_cam, dy_cam, dx_cam, dy_cam]).repeat(data[idx_cam:].shape[0], 1).cumsum(0)
            feat_translation[idx_cam:, :] += torch.tensor([dx_cam, dy_cam]).repeat(data[idx_cam:].shape[0], 1).cumsum(0)

        # frames indices start at 1, saved features indices start at 0, so subtract 1
        frames = np.array([self.track_frames[ep_index][i] for i in data_idc]) - 1
        original_image_sizes = torch.from_numpy(self.original_image_sizes[seq][frames])
        image_sizes = torch.from_numpy(self.image_sizes[seq][frames])

        # noise
        if self.max_noise > 0.:
            widths = data[:, 2] - data[:, 0]
            heights = data[:, 3] - data[:, 1]
            rel_noise = torch.randn(data.shape[0], 4) * self.max_noise
            noise = rel_noise * torch.stack([widths, heights, widths, heights], dim=1)
            if not self.augment_target:
                noise[-self.target_length:] = 0.
            data[:, :4] += noise

        if self.flip_prob > 0.0 and torch.rand(()) < self.flip_prob:
            # horizontally flip positions
            width = original_image_sizes[0, 1]
            data[:, [0, 2]] = width.repeat(data.shape[0], 2) - data[:, [0, 2]]
            if self.flipped_features_path is None:
                # load normal features and flip them in width dimension
                image_features = torch.from_numpy(np.flip(self.image_features[seq][frames], axis=3).copy())
            else:
                # load flipped features
                image_features = torch.from_numpy(self.flipped_features[seq][frames])
        else:
            # load normal features
            image_features = torch.from_numpy(self.image_features[seq][frames])

        # if self.flip_prob > 0.0 and torch.rand(()) < self.flip_prob:
        #     width = original_image_sizes[0, 1]
        #     data[:, [0, 2]] = width.repeat(data.shape[0], 2) - data[:, [0, 2]]
        #     if self.flipped_features_path is not None:
        #         image_features = torch.from_numpy(self.flipped_features[seq][frames]).float()
        # else:
        #     image_features = torch.from_numpy(self.image_features[seq][frames]).float()

        # split to input and target
        # resize these to 1080 x 1920
        boxes_in, boxes_target = data[:-self.target_length].clone(), data[-self.target_length:].clone()
        scales = torch.tensor([1080, 1920]) / torch.tensor(original_image_sizes[0]).float()
        scale = scales.min()
        boxes_in[:, :4] = boxes_in[:, :4] * scale
        boxes_target[:, :4] = boxes_target[:, :4] * scale

        # padding
        input_length = self.episode_length - self.target_length
        if boxes_in.shape[0] < input_length:
            pad = torch.zeros(input_length - boxes_in.shape[0], 6)
            boxes_in = torch.cat([pad, boxes_in])

        # these will be resized in the collate function to allow for later RoI pooling
        boxes_all = data[:, :4].clone()
        # we don't need target positions for RoI
        boxes_all[-self.target_length:] = 0.

        return boxes_in, boxes_target, boxes_all, image_features, original_image_sizes, image_sizes, length, feat_translation


def collate(elems):
    boxes_in, boxes_target, boxes_all, image_features, original_image_sizes, image_sizes, lengths, feat_translation = zip(
        *elems)
    boxes_in = default_collate(boxes_in)
    lengths = default_collate(lengths)
    boxes_target = default_collate(boxes_target)
    image_features = torch.cat(image_features)
    original_image_sizes = torch.cat(original_image_sizes)
    image_sizes = torch.cat(image_sizes)

    # get resized bounding boxes for later RoI pooling
    first_idc = [int(sum(lengths[:i])) for i in range(0, len(lengths))]
    boxes_resized = []
    feat_translation_resized = []
    for seq_start, boxes, feat_trans in zip(first_idc, boxes_all, feat_translation):
        boxes_resized.append(resize_boxes(boxes, original_image_sizes[seq_start], image_sizes[seq_start]))
        feat_translation_resized.append(
            resize_boxes(feat_trans.repeat(1, 2), original_image_sizes[seq_start], image_sizes[seq_start])[:, :2]
        )
    boxes_resized = torch.cat(boxes_resized)
    feat_translation_resized = torch.cat(feat_translation_resized)

    # calculate feature translation in feature scale
    scale = infer_scale(image_features, image_sizes[0])
    feat_translation = (feat_translation_resized * scale).round()

    # apply translation to feature map
    pad_w = int(feat_translation[:, 0].abs().max())
    pad_h = int(feat_translation[:, 1].abs().max())
    if pad_w == 0 and pad_h == 0:
        feat_out = image_features
    else:
        feat_padded = F.pad(image_features, [pad_w, pad_w, pad_h, pad_h])
        origin = torch.tensor([pad_w, pad_h])
        new_coords = (origin - feat_translation).long()

        h, w = image_features.shape[-2:]
        feat_out = []
        for i in range(feat_padded.shape[0]):
            x, y = new_coords[i]
            feat_out.append(feat_padded[i, :, y:(y + h), x:(x + w)])
        feat_out = torch.stack(feat_out)

    return boxes_in, boxes_target, boxes_resized, feat_out, image_sizes, lengths
