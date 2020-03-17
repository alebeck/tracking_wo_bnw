from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as f
from torchvision.ops import MultiScaleRoIAlign

from .utils import conv, correlate, flatten
from tracktor.utils import get_height, get_width
from tracktor.frcnn_fpn import FRCNN_FPN


class Seq2Seq(nn.Module):

    def __init__(
            self,
            input_size,
            hidden_size,
            output_size,
            input_length,
            n_layers=1,
            dropout=0.,
            cmc_only_len_1=False
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_length = input_length
        self.n_layers = n_layers
        self.cmc_only_len_1 = cmc_only_len_1

        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=n_layers, dropout=dropout)

        self.attn = nn.Linear(self.hidden_size + self.output_size, self.input_length)
        self.attn_combine = nn.Linear(self.hidden_size + self.output_size, self.hidden_size)

        self.decoder = nn.LSTM(input_size + hidden_size, hidden_size, batch_first=True, num_layers=n_layers,
                               dropout=dropout)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, target, teacher_forcing=False):
        B = x.shape[0]

        encoder_out = self.encoder(x)  # encoder_out[0]: 32, 60, 48
        last_h = encoder_out[1][0]
        last_c = torch.zeros(self.n_layers, B, self.hidden_size).cuda()

        decoder_in = torch.cat([torch.tensor([[[0., 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]])] * B).cuda()
        decoder_in[:, :, 4:10] = target[:, :, 4:10]
        out_seq = []

        for i in range(target.shape[1]):
            attn_weights = f.softmax(
                self.attn(torch.cat([last_h[0], decoder_in.squeeze(1)], dim=1)), dim=1)  # 32, 60
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_out[0])  # 32, 1, 48
            decoder_in = torch.cat([decoder_in, attn_applied], dim=2)  # 32, 1, 54

            _, (last_h, last_c) = self.decoder(decoder_in, (last_h, last_c))
            out = self.linear2(f.relu(self.linear1(last_h.sum(dim=0))))  # [1,12]
            out = out.unsqueeze(1)  # add sequence dimension
            out_seq.append(out)

            if teacher_forcing:
                decoder_in = target[:, i, :].unsqueeze(1)
            else:
                decoder_in = out.detach()

        return torch.cat(out_seq, dim=1)

    def predict(self, x, target, output_length):
        B = x.shape[0]

        encoder_out = self.encoder(x)  # encoder_out[0]: 32, 60, 48
        last_h = encoder_out[1][0]
        last_c = torch.zeros(self.n_layers, B, self.hidden_size).cuda()

        decoder_in = torch.cat([torch.tensor([[[0., 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]])] * B).cuda()
        decoder_in[:, :, 4:10] = target[:, :, 4:10]
        out_seq = []

        for i in range(output_length):
            attn_weights = f.softmax(
                self.attn(torch.cat([last_h[0], decoder_in.squeeze(1)], dim=1)), dim=1)  # 32, 60
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_out[0])  # 32, 1, 48
            decoder_in = torch.cat([decoder_in, attn_applied], dim=2)  # 32, 1, 54

            _, (last_h, last_c) = self.decoder(decoder_in, (last_h, last_c))
            out = self.linear2(f.relu(self.linear1(last_h.sum(dim=0))))
            out = out.unsqueeze(1)  # add sequence dimension
            out_seq.append(out)

            decoder_in = out

        return torch.cat(out_seq, dim=1)


class CorrelationSeq2Seq(nn.Module):

    def __init__(
            self,
            correlation_args,
            batch_norm,
            conv_channels,
            n_box_channels,
            roi_output_size,
            avg_box_features,
            hidden_size,
            input_length,
            n_layers,
            dropout,
            correlation_only,
            use_env_features,
            fixed_env,
            use_height_feature,
            correlation_last_only
    ):
        super().__init__()

        self.correlation_args = correlation_args
        self.batch_norm = batch_norm
        self.conv_channels = conv_channels
        self.n_box_channels = n_box_channels
        self.roi_output_size = roi_output_size
        self.avg_box_features = avg_box_features
        self.hidden_size = hidden_size
        self.input_length = input_length
        self.n_layers = n_layers
        self.output_size = 6
        self.dropout = dropout
        self.correlation_only = correlation_only
        self.use_env_features = use_env_features
        self.fixed_env = fixed_env
        self.use_height_feature = use_height_feature
        self.correlation_last_only = correlation_last_only

        locations_per_box = 1 if self.avg_box_features else roi_output_size ** 2
        multiplier = 2 if self.use_env_features else 1
        self.input_size = 6 + (n_box_channels * locations_per_box * multiplier)
        if self.use_height_feature:
            self.input_size += 1

        self.box_roi_pool = MultiScaleRoIAlign(
            featmap_names=[0, 1, 2, 3],
            output_size=self.roi_output_size,
            sampling_ratio=2
        )

        # layers similar to https://github.com/ClementPinard/FlowNetPytorch/blob/master/models/FlowNetC.py
        self.conv_redir = conv(self.batch_norm, 256, 32, kernel_size=1, stride=1)
        in_planes = (self.correlation_args['patch_size'] ** 2) + (0 if self.correlation_only else 32)
        self.conv3_1 = conv(self.batch_norm, in_planes, self.conv_channels)
        self.conv4 = conv(self.batch_norm, self.conv_channels, self.conv_channels)
        self.conv4_1 = conv(self.batch_norm, self.conv_channels, self.n_box_channels)

        # recurrent layers
        self.encoder = nn.LSTM(self.input_size, self.hidden_size, batch_first=True, num_layers=n_layers,
                               dropout=dropout)
        self.attn = nn.Linear(self.hidden_size + self.input_size, self.input_length)
        self.attn_combine = nn.Linear(self.hidden_size + self.output_size, self.hidden_size)
        self.decoder = nn.LSTM(self.input_size + self.hidden_size, self.hidden_size, batch_first=True,
                               num_layers=n_layers, dropout=dropout)
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.output_size)

    def prepare_decoder(self, diffs, boxes_resized, image_features, image_sizes, lengths):
        B, L, F = diffs.shape

        bounds = (torch.cumsum(lengths, dim=0) - 1).tolist()
        keep = list(set(range(len(image_features) - 1)).difference(set(bounds[:-1])))
        keep_first = flatten([list(range(b - l + 1, b - 1)) for l, b in zip(lengths, bounds)])

        correlation = correlate(image_features[:-1], image_features[1:], self.correlation_args)
        out_correlation = correlation[keep]

        if self.correlation_only:
            in_conv3_1 = out_correlation
        else:
            out_conv_redir = self.conv_redir(image_features[keep])
            in_conv3_1 = torch.cat([out_conv_redir, out_correlation], dim=1)

        out_conv3 = self.conv3_1(in_conv3_1)
        out_conv4 = self.conv4_1(self.conv4(out_conv3))

        # roi pool over feature maps to get box features
        out_conv4 = OrderedDict([(0, out_conv4)])
        proposals = list(boxes_resized[keep].unsqueeze(1))
        box_features = self.box_roi_pool(out_conv4, proposals, image_sizes[keep].tolist())

        if self.use_env_features:
            # semi-global features
            widths, heights = get_width(boxes_resized[keep]), get_height(boxes_resized[keep])

            if not self.fixed_env:
                additive = torch.stack([
                    -widths,
                    -heights / 2,
                    widths,
                    heights / 2
                ], dim=1)
            else:
                assert isinstance(self.fixed_env, int)
                additive = torch.tensor([-self.fixed_env, -self.fixed_env, self.fixed_env, self.fixed_env]).cuda()
            env_proposals = list((boxes_resized[keep] + additive).unsqueeze(1))
            env_features = self.box_roi_pool(out_conv4, env_proposals, image_sizes[keep].tolist())
            box_features = torch.cat([box_features, env_features], dim=1)

        if self.avg_box_features:
            box_features = box_features.view(*box_features.shape[:2], -1).mean(2).unsqueeze(2)

        # construct encoder input
        encoder_in = torch.zeros(B, L, self.input_size).cuda()
        encoder_in[:, :, :F] = diffs

        corr_lengths = lengths - 1
        mask = torch.zeros(encoder_in.shape[:2], dtype=torch.bool)
        for i, l in enumerate(corr_lengths):
            if l - 1 > 0:
                mask[i, -(l - 1):] = True

        target_idc = (torch.cumsum(corr_lengths, dim=0) - 1).tolist()
        in_idc = list(set(range(len(out_correlation))).difference(set(target_idc)))
        box_features_start = F if not self.use_height_feature else F + 1
        if len(in_idc) > 0 and not self.correlation_last_only:
            #encoder_in[mask][:, F:] = box_features[in_idc].view(len(in_idc), -1)
            t_tmp = encoder_in[mask]
            t_tmp[:, box_features_start:] = box_features[in_idc].view(len(in_idc), -1)
            if self.use_height_feature:
                t_tmp[:, F] = boxes_resized[keep_first][:, 3] - boxes_resized[keep_first][:, 1]
            encoder_in[mask] = t_tmp

        # feed features into encoder, retrieve hidden states
        encoder_out = self.encoder(encoder_in)  # encoder_out[0]: 32, 60, 48
        decoder_h = encoder_out[1][0]
        decoder_c = torch.zeros(self.n_layers, B, self.hidden_size).cuda()

        # construct decoder input
        decoder_in = torch.zeros(B, 1, self.input_size).cuda()
        decoder_in[:, 0, F - 2] = 1.  # start token
        decoder_in[:, 0, box_features_start:] = box_features[target_idc].view(len(target_idc), -1)
        if self.use_height_feature:
            decoder_in[:, 0, F] = boxes_resized[torch.tensor(bounds) - 1][:, 3] - boxes_resized[torch.tensor(bounds) - 1][:, 1]

        return encoder_out, decoder_in, decoder_h, decoder_c

    def forward(self, diffs, boxes_target, boxes_resized, image_features, image_sizes, lengths, teacher_forcing=False):
        encoder_out, decoder_in, last_h, last_c = \
            self.prepare_decoder(diffs, boxes_resized, image_features, image_sizes, lengths)
        out_seq = []

        for i in range(boxes_target.shape[1]):
            attn_weights = f.softmax(
                self.attn(torch.cat([last_h[0], decoder_in.squeeze(1)], dim=1)), dim=1)  # 32, 60
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_out[0])  # 32, 1, 48
            decoder_in = torch.cat([decoder_in, attn_applied], dim=2)  # 32, 1, 54

            _, (last_h, last_c) = self.decoder(decoder_in, (last_h, last_c))
            out = self.linear2(f.relu(self.linear1(last_h.sum(dim=0))))  # [1,12]
            out = out.unsqueeze(1)  # add sequence dimension
            out_seq.append(out)

            assert boxes_target.shape[1] == 1

        # apply linear0, detach necessary?, add image features
        # if teacher_forcing:
        # 	decoder_in = boxes_target[:, i, :].unsqueeze(1)
        # else:
        # 	decoder_in = out.detach()

        return torch.cat(out_seq, dim=1)

    def predict(self, diffs, boxes_resized, image_features, image_sizes, lengths, output_length):
        assert output_length == 1

        encoder_out, decoder_in, last_h, last_c = \
            self.prepare_decoder(diffs, boxes_resized, image_features, image_sizes, lengths)
        out_seq = []

        for i in range(output_length):
            attn_weights = f.softmax(
                self.attn(torch.cat([last_h[0], decoder_in.squeeze(1)], dim=1)), dim=1)  # 32, 60
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_out[0])  # 32, 1, 48
            decoder_in = torch.cat([decoder_in, attn_applied], dim=2)  # 32, 1, 54

            _, (last_h, last_c) = self.decoder(decoder_in, (last_h, last_c))
            out = self.linear2(f.relu(self.linear1(last_h.sum(dim=0))))
            out = out.unsqueeze(1)  # add sequence dimension
            out_seq.append(out)

            decoder_in = out

        return torch.cat(out_seq, dim=1)


class FRCNNSeq2Seq(CorrelationSeq2Seq):

    def __init__(
            self,
            obj_detect_weights,
            correlation_args,
            batch_norm,
            conv_channels,
            n_box_channels,
            roi_output_size,
            avg_box_features,
            hidden_size,
            input_length,
            n_layers,
            dropout,
            correlation_only,
            use_env_features,
            fixed_env,
            use_height_feature,
            correlation_last_only,
            feature_level=1
    ):
        super().__init__(
            correlation_args,
            batch_norm,
            conv_channels,
            n_box_channels,
            roi_output_size,
            avg_box_features,
            hidden_size,
            input_length,
            n_layers,
            dropout,
            correlation_only,
            use_env_features,
            fixed_env,
            use_height_feature,
            correlation_last_only
        )
        self.feature_level = feature_level
        self.frcnn: nn.Module = FRCNN_FPN(num_classes=2)
        self.frcnn.load_state_dict(torch.load(obj_detect_weights))

    def forward(self, diffs, boxes_target, boxes_resized, image_features, image_sizes, lengths, teacher_forcing=False):
        # feed images through object detector
        image_features = self.frcnn.backbone(image_features)[self.feature_level]
        # delegate remaining tasks to super class
        return super().forward(diffs, boxes_target, boxes_resized, image_features, image_sizes, lengths, teacher_forcing)

    def predict(self, diffs, boxes_resized, image_features, image_sizes, lengths, output_length):
        image_features = self.frcnn.backbone(image_features)[self.feature_level]
        return super().predict(diffs, boxes_resized, image_features, image_sizes, lengths, output_length)