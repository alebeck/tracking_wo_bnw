import time
from collections import deque, OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import cv2
from torchvision.models.detection._utils import BoxCoder
from torchvision.ops.boxes import clip_boxes_to_image, nms
from torchvision.models.detection.transform import resize_boxes

from .utils import bbox_overlaps, warp_pos, get_center, infer_scale


class Tracker:
    """The main tracking file, here is where magic happens."""
    # only track pedestrian
    cl = 1

    def __init__(self, obj_detect, reid_network, motion_network, tracker_cfg, motion_cfg, min_length):
        self.obj_detect = obj_detect
        self.reid_network = reid_network
        self.motion_network = motion_network
        self.detection_person_thresh = tracker_cfg['detection_person_thresh']
        self.regression_person_thresh = tracker_cfg['regression_person_thresh']
        self.detection_nms_thresh = tracker_cfg['detection_nms_thresh']
        self.regression_nms_thresh = tracker_cfg['regression_nms_thresh']
        self.public_detections = tracker_cfg['public_detections']
        self.inactive_patience = tracker_cfg['inactive_patience']
        self.do_reid = tracker_cfg['do_reid']
        self.max_features_num = tracker_cfg['max_features_num']
        self.reid_sim_threshold = tracker_cfg['reid_sim_threshold']
        self.reid_iou_threshold = tracker_cfg['reid_iou_threshold']
        self.do_align = tracker_cfg['do_align']
        self.align_actives = tracker_cfg['align_actives']
        self.align_inactives = tracker_cfg['align_inactives']
        self.motion_model_enabled = tracker_cfg['motion_model_enabled']
        self.motion_cfg = motion_cfg
        self.min_length = min_length
        self.cam_features_1_only = tracker_cfg['cam_features_1_only']
        self.write_inactives = tracker_cfg['write_inactives']

        self.warp_mode = eval(tracker_cfg['warp_mode'])
        self.number_of_iterations = tracker_cfg['number_of_iterations']
        self.termination_eps = tracker_cfg['termination_eps']

        if self.motion_cfg['use_box_coding']:
            self.box_coder = BoxCoder(self.motion_cfg['box_coding_weights'])

        self.tracks = []
        self.inactive_tracks = []
        self.track_num = 0
        self.im_index = 0
        self.results = {}
        self.last_warps = []
        self.last_image = None

        if self.motion_model_enabled and self.motion_cfg['use_correlation_model']:
            self.last_features = deque([], maxlen=self.motion_cfg['model_args']['input_length'] + 2)

        # load PCA model for dimensionality reduction of image features
        if self.motion_cfg['reduce_features'] is not None:
            pca = torch.load(self.motion_cfg['reduce_features'])
            self.pca_components = pca['components']
            self.pca_mean = pca['mean']

    def reset(self, hard=True):
        self.tracks = []
        self.inactive_tracks = []

        if hard:
            self.track_num = 0
            self.results = {}
            self.im_index = 0
            self.last_warps = []

    def tracks_to_inactive(self, tracks):
        self.tracks = [t for t in self.tracks if t not in tracks]
        for t in tracks:
            t.pos = t.last_pos[-1]
        self.inactive_tracks += tracks

    def add(self, new_det_pos, new_det_scores, new_det_features):
        """Initializes new Track objects and saves them."""
        num_new = new_det_pos.size(0)
        for i in range(num_new):
            self.tracks.append(Track(
                new_det_pos[i].view(1, -1),
                new_det_scores[i],
                self.track_num + i,
                new_det_features[i].view(1, -1),
                self.inactive_patience,
                self.max_features_num,
                self.motion_cfg['model_args']['input_length']
            ))
        self.track_num += num_new

    def regress_tracks(self, blob):
        """Regress the position of the tracks and also checks their scores."""
        pos = self.get_pos()

        # regress
        boxes, scores = self.obj_detect.predict_boxes(pos)
        pos = clip_boxes_to_image(boxes, blob['img'].shape[-2:])

        s = []
        for i in range(len(self.tracks) - 1, -1, -1):
            t = self.tracks[i]
            t.score = scores[i]
            if scores[i] <= self.regression_person_thresh:
                self.tracks_to_inactive([t])
            else:
                s.append(scores[i])
                # t.prev_pos = t.pos
                t.pos = pos[i].view(1, -1)

        return torch.Tensor(s[::-1]).cuda()

    def get_pos(self):
        """Get the positions of all active tracks."""
        if len(self.tracks) == 1:
            pos = self.tracks[0].pos
        elif len(self.tracks) > 1:
            pos = torch.cat([t.pos for t in self.tracks], 0)
        else:
            pos = torch.zeros(0).cuda()
        return pos

    def get_features(self):
        """Get the features of all active tracks."""
        if len(self.tracks) == 1:
            features = self.tracks[0].features
        elif len(self.tracks) > 1:
            features = torch.cat([t.features for t in self.tracks], 0)
        else:
            features = torch.zeros(0).cuda()
        return features

    def get_inactive_features(self):
        """Get the features of all inactive tracks."""
        if len(self.inactive_tracks) == 1:
            features = self.inactive_tracks[0].features
        elif len(self.inactive_tracks) > 1:
            features = torch.cat([t.features for t in self.inactive_tracks], 0)
        else:
            features = torch.zeros(0).cuda()
        return features

    def reid(self, blob, new_det_pos, new_det_scores):
        """Tries to ReID inactive tracks with provided detections."""
        new_det_features = [torch.zeros(0).cuda() for _ in range(len(new_det_pos))]

        if self.do_reid:
            new_det_features = self.reid_network.test_rois(blob['img'], new_det_pos).data

            if len(self.inactive_tracks) >= 1:
                # calculate appearance distances
                dist_mat, pos = [], []
                for t in self.inactive_tracks:
                    dist_mat.append(torch.cat([t.test_features(feat.view(1, -1)) for feat in new_det_features], dim=1))
                    pos.append(t.pos)
                if len(dist_mat) > 1:
                    dist_mat = torch.cat(dist_mat, 0)
                    pos = torch.cat(pos, 0)
                else:
                    dist_mat = dist_mat[0]
                    pos = pos[0]

                # calculate IoU distances
                iou = bbox_overlaps(pos, new_det_pos)
                iou_mask = torch.ge(iou, self.reid_iou_threshold)
                iou_neg_mask = ~iou_mask
                # make all impossible assignments to the same add big value
                dist_mat = dist_mat * iou_mask.float() + iou_neg_mask.float() * 1000
                dist_mat = dist_mat.cpu().numpy()

                row_ind, col_ind = linear_sum_assignment(dist_mat)

                assigned = []
                remove_inactive = []
                for r, c in zip(row_ind, col_ind):
                    if dist_mat[r, c] <= self.reid_sim_threshold:
                        t = self.inactive_tracks[r]
                        self.tracks.append(t)
                        t.count_inactive = 0
                        t.pos = new_det_pos[c].view(1, -1)
                        t.reset_last_pos()
                        t.add_features(new_det_features[c].view(1, -1))
                        assigned.append(c)
                        remove_inactive.append(t)

                for t in remove_inactive:
                    self.inactive_tracks.remove(t)

                keep = torch.Tensor([i for i in range(new_det_pos.size(0)) if i not in assigned]).long().cuda()
                if keep.nelement() > 0:
                    new_det_pos = new_det_pos[keep]
                    new_det_scores = new_det_scores[keep]
                    new_det_features = new_det_features[keep]
                else:
                    new_det_pos = torch.zeros(0).cuda()
                    new_det_scores = torch.zeros(0).cuda()
                    new_det_features = torch.zeros(0).cuda()

        return new_det_pos, new_det_scores, new_det_features

    def get_appearances(self, blob):
        """Uses the siamese CNN to get the features for all active tracks."""
        new_features = self.reid_network.test_rois(blob['img'], self.get_pos()).data
        return new_features

    def add_features(self, new_features):
        """Adds new appearance features to active tracks."""
        for t, f in zip(self.tracks, new_features):
            t.add_features(f.view(1, -1))

    def align(self, blob):
        """Aligns the positions of active and inactive tracks depending on camera motion."""
        t_warp = None
        if self.im_index > 0:
            im1 = np.transpose(self.last_image.cpu().numpy(), (1, 2, 0))
            im2 = np.transpose(blob['img'][0].cpu().numpy(), (1, 2, 0))
            im1_gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
            im2_gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.number_of_iterations, self.termination_eps)

            t_warp_start = time.time()
            cc, warp_matrix = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, self.warp_mode, criteria)
            t_warp = time.time() - t_warp_start

            warp_matrix = torch.from_numpy(warp_matrix)
            self.last_warps.append(warp_matrix.cuda())

            for t in self.tracks:
                t.before_align_pos = t.pos.clone()

            if self.align_actives:
                for t in self.tracks:
                    t.pos = warp_pos(t.pos, warp_matrix)

                if self.motion_model_enabled:
                    for t in self.tracks:
                        for i in range(len(t.last_pos)):
                            t.last_pos[i] = warp_pos(t.last_pos[i], warp_matrix)

            if self.align_inactives and self.do_reid:
                for t in self.inactive_tracks:
                    t.pos = warp_pos(t.pos, warp_matrix)

            for t in self.tracks:
                for i in range(len(t.last_pos_aligned)):
                    t.last_pos_aligned[i] = warp_pos(t.last_pos_aligned[i], warp_matrix)

        return t_warp

    def motion(self, blob):
        for t in self.tracks:
            assert len(t.last_pos) > 0
            last_pos = torch.cat(list(t.last_pos), dim=0)

            if len(t.last_pos) < self.min_length - 1:  # TODO account for target length != 1
                # we have a sequence length that the model was not trained for, resort to CVA-5
                if not len(t.last_pos) == 1:
                    diff = last_pos[1:] - last_pos[:-1]
                    t.pos = clip_boxes_to_image(t.pos + diff[-5:, :4].mean(0), blob['img'].shape[-2:])
                continue

            if len(t.last_pos) > 1:
                if self.motion_cfg['use_box_coding']:
                    diff = self.box_coder.encode([last_pos[1:, :4]], [last_pos[:-1, :4]])[0]
                    diff = torch.cat([diff, torch.zeros(diff.shape[0], 2).cuda()], dim=1)
                    diff[:, 5] = 1  # not padding
                else:
                    diff = last_pos[1:] - last_pos[:-1]
                    diff = torch.cat([diff, torch.zeros(diff.shape[0], 2).cuda()], dim=1)
                    diff[:, 5] = 1  # not padding
                    t.pos = clip_boxes_to_image(t.pos + diff[-1, :4], blob['img'].shape[-2:])
            else:
                diff = torch.zeros(0, 6).cuda()

            padding = torch.zeros(self.motion_cfg['model_args']['input_length'] - diff.shape[0], 6).cuda()
            padded = torch.cat([padding, diff], dim=0)
            prediction = self.motion_network.predict(padded.unsqueeze(0), 1)

            if self.motion_cfg['use_box_coding']:
                # prediction is the acceleration in encoding space
                last_offset = padded[[-1], :4]
                pred_offset = last_offset + prediction[0, 0, :4]
                pred_pos = self.box_coder.decode([pred_offset], [last_pos[-1].unsqueeze(0)])
                t.pos = clip_boxes_to_image(pred_pos[0], blob['img'].shape[-2:])
            else:
                t.pos = clip_boxes_to_image(t.pos + prediction[0, 0, :4], blob['img'].shape[-2:])

    def motion_cva(self, blob):
        for t in self.tracks:
            if len(t.last_pos) > 1:
                last_pos = torch.cat(list(t.last_pos), dim=0)
                diff = last_pos[1:] - last_pos[:-1]
                t.last_v = diff[-5:, :4].mean(0)
                t.pos = clip_boxes_to_image(t.pos + t.last_v, blob['img'].shape[-2:])

        if self.do_reid and self.motion_cfg['model_inactives']:
            for t in self.inactive_tracks:
                t.pos = clip_boxes_to_image(t.pos + t.last_v, blob['img'].shape[-2:])

    def motion_correlation(self, blob):
        for t in self.tracks:
            assert len(t.last_pos) > 0
            last_pos = torch.cat(list(t.last_pos), dim=0)

            if len(t.last_pos) < self.min_length - 1:
                raise ValueError('sequence length that the model was not trained for')

            if len(t.last_pos) > 1:
                if self.motion_cfg['use_box_coding']:
                    # [0] because batch size == 0
                    diff = self.box_coder.encode([last_pos[1:, :4]], [last_pos[:-1, :4]])[0]
                    diff = torch.cat([diff, torch.zeros(diff.shape[0], 2).cuda()], dim=1)
                    diff[:, 5] = 1  # not padding flag
                else:
                    scales = torch.tensor([1080, 1920]) / torch.tensor(self.obj_detect.original_image_sizes[0]).float()
                    scale = scales.min()

                    diff = last_pos[1:] - last_pos[:-1]
                    diff = torch.cat([diff, torch.zeros(diff.shape[0], 2).cuda()], dim=1)
                    diff[:, 5] = 1  # not padding flag
                    t.pos = clip_boxes_to_image(t.pos + diff[-1, :4], blob['img'].shape[-2:])

                    diff[:, :4] = diff[:, :4] * scale
            else:
                diff = torch.zeros(0, 6).cuda()

            padding = torch.zeros(self.motion_cfg['model_args']['input_length'] - diff.shape[0], 6).cuda()
            padded = torch.cat([padding, diff], dim=0)

            # boxes_resized has to have same length as box features, pad with zeros
            last_pos_padded = torch.cat([last_pos, torch.zeros(1, 4).cuda()])
            # convert position to resized image
            boxes_resized = resize_boxes(last_pos_padded, blob['img'].shape[-2:],
                                         self.obj_detect.preprocessed_images.image_sizes[0])

            features = torch.cat(list(self.last_features)[-last_pos_padded.shape[0]:])

            if self.do_align and self.align_actives:
                trans = get_center(t.pos) - get_center(t.before_align_pos)
                scale = infer_scale(features, self.obj_detect.preprocessed_images.image_sizes[0])
                # resize translation
                trans_res = resize_boxes(
                    trans.repeat(1, 2),
                    self.obj_detect.original_image_sizes[0],
                    self.obj_detect.preprocessed_images.image_sizes[0]
                )[0, :2]

                t.feature_translations.append((trans_res * scale).round())
                last_translations = t.feature_translations[-(features.shape[0] - 1):]
                accu_trans = torch.stack(last_translations).flip(dims=(0,)).cumsum(dim=0).flip(dims=(0,))

                feature_h, feature_w = features.shape[-2:]

                # assemble past feature maps
                features_translated = []
                for i in range(features.shape[0] - 1):
                    trans_feat = accu_trans[i]
                    # translate i-th map with i-th translation
                    pad_left = int(trans_feat[0]) if trans_feat[0] > 0 else 0
                    pad_right = int(trans_feat[0].abs()) if trans_feat[0] < 0 else 0
                    pad_top = int(trans_feat[1]) if trans_feat[1] > 0 else 0
                    pad_bottom = int(trans_feat[1].abs()) if trans_feat[1] < 0 else 0

                    feat_padded = F.pad(features[i], [pad_left, pad_right, pad_top, pad_bottom])
                    features_translated.append(
                        feat_padded[:, pad_bottom:(pad_bottom + feature_h), pad_right:(pad_right + feature_w)]
                    )
                    assert features_translated[-1].shape[-2:] == (feature_h, feature_w)

                # add last feature map
                features_translated.append(features[-1])
                features = torch.stack(features_translated)

            prediction = self.motion_network.predict(
                padded.unsqueeze(0),
                boxes_resized,
                features,
                torch.tensor(self.obj_detect.preprocessed_images.image_sizes[0]).repeat(last_pos_padded.shape[0], 1),
                torch.tensor([last_pos_padded.shape[0]]),
                output_length=1
            )

            if self.motion_cfg['use_box_coding']:
                if self.motion_cfg['predict_coded_a']:
                    # prediction is the acceleration in encoding space
                    last_offset = padded[[-1], :4]
                    pred_offset = last_offset + prediction[0, 0, :4]
                    pred_pos = self.box_coder.decode([pred_offset], [last_pos[-1].unsqueeze(0)])
                else:
                    # prediction is the absolute encoded offset
                    pred_pos = self.box_coder.decode(list(prediction[:, :, :4]), [last_pos[-1].unsqueeze(0)])
                t.pos = clip_boxes_to_image(pred_pos[0], blob['img'].shape[-2:])
            else:
                scales = torch.tensor(self.obj_detect.original_image_sizes[0]).float() / torch.tensor([1080, 1920])
                scale = scales.max()
                t.pos = clip_boxes_to_image(t.pos + prediction[0, 0, :4] * scale.cuda(), blob['img'].shape[-2:])

    def motion_correlation_batched(self, blob):
        all_padded = []
        all_last_pos_padded = []
        all_boxes_resized = []
        all_last_pos = []

        for t in self.tracks:
            assert len(t.last_pos) > 0
            last_pos = torch.cat(list(t.last_pos), dim=0)

            if len(t.last_pos) < self.min_length - 1:
                raise ValueError('sequence length that the model was not trained for')

            if len(t.last_pos) > 1:
                if self.motion_cfg['use_box_coding']:
                    # [0] because batch size == 0
                    diff = self.box_coder.encode([last_pos[1:, :4]], [last_pos[:-1, :4]])[0]
                    diff = torch.cat([diff, torch.zeros(diff.shape[0], 2).cuda()], dim=1)
                    diff[:, 5] = 1  # not padding flag
                else:
                    scales = torch.tensor([1080, 1920]) / torch.tensor(self.obj_detect.original_image_sizes[0]).float()
                    scale = scales.min()

                    diff = last_pos[1:] - last_pos[:-1]
                    diff = torch.cat([diff, torch.zeros(diff.shape[0], 2).cuda()], dim=1)
                    diff[:, 5] = 1  # not padding flag
                    t.pos = clip_boxes_to_image(t.pos + diff[-1, :4], blob['img'].shape[-2:])

                    diff[:, :4] = diff[:, :4] * scale
            else:
                diff = torch.zeros(0, 6).cuda()

            padding = torch.zeros(self.motion_cfg['model_args']['input_length'] - diff.shape[0], 6).cuda()
            padded = torch.cat([padding, diff], dim=0)

            # boxes_resized has to have same length as box features, pad with zero
            last_pos_padded = torch.cat([last_pos, torch.zeros(1, 4).cuda()])
            # convert position to resized image
            boxes_resized = resize_boxes(last_pos_padded, blob['img'].shape[-2:],
                                         self.obj_detect.preprocessed_images.image_sizes[0])

            # save values to be batched afterwards
            all_padded.append(padded)
            all_last_pos_padded.append(last_pos_padded)
            all_last_pos.append(last_pos)
            all_boxes_resized.append(boxes_resized)

        lengths = torch.tensor([l.shape[0] for l in all_last_pos_padded])

        # make prediction
        prediction = self.motion_network.predict(
            torch.stack(all_padded),
            torch.cat(all_boxes_resized),
            torch.cat(list(self.last_features)[-lengths.max():]),
            torch.tensor(self.obj_detect.preprocessed_images.image_sizes[0]).repeat(sum([len(l) for l in all_last_pos_padded]), 1),
            lengths,
            output_length=1,
            batched=True
        )

        # apply predictions to individual tracks
        for i, t in enumerate(self.tracks):
            if self.motion_cfg['use_box_coding']:
                if self.motion_cfg['predict_coded_a']:
                    # prediction is the acceleration in encoding space
                    last_offset = all_padded[i][[-1], :4]
                    pred_offset = last_offset + prediction[i, 0, :4]
                    pred_pos = self.box_coder.decode([pred_offset], [all_last_pos[i][-1].unsqueeze(0)])
                else:
                    # prediction is the absolute encoded offset
                    pred_pos = self.box_coder.decode(list(prediction[[i], :, :4]), [all_last_pos[i][-1].unsqueeze(0)])
                t.pos = clip_boxes_to_image(pred_pos[0], blob['img'].shape[-2:])
            else:
                scales = torch.tensor(self.obj_detect.original_image_sizes[0]).float() / torch.tensor([1080, 1920])
                scale = scales.max()
                t.pos = clip_boxes_to_image(t.pos + prediction[i, 0, :4] * scale.cuda(), blob['img'].shape[-2:])

    def step(self, blob):
        """This function should be called every time step to perform tracking with a blob
        containing the image information.
        """
        for t in self.tracks:
            # add current position to last_pos list
            t.last_pos.append(t.pos.clone())
            t.last_pos_aligned.append(t.pos.clone())
            t.last_pos_inact.append(t.pos.clone())

        for t in self.inactive_tracks:
            t.last_pos_inact.append(t.pos.clone())

        ###########################
        # Look for new detections #
        ###########################

        self.obj_detect.load_image(blob['img'])
        if self.motion_model_enabled and self.motion_cfg['use_correlation_model']:
            if self.motion_cfg['reduce_features'] is not None:
                # use CVA feature reduction
                if self.motion_cfg['feature_level'] is not None:
                    features = self.obj_detect.features[self.motion_cfg['feature_level']].permute(0, 2, 3, 1)
                    transformed = torch.mm(features.view(-1, 256) - self.pca_mean, self.pca_components.t()).float()
                    transformed = transformed.view(*features.shape[:-1], -1).permute(0, 3, 1, 2)
                    self.last_features.append(transformed)
                else:
                    new_features = OrderedDict()
                    for key, feat in self.obj_detect.features.items():
                        feat = feat.permute(0, 2, 3, 1)
                        transformed = torch.mm(feat.view(-1, 256) - self.pca_mean, self.pca_components.t()).float()
                        transformed = transformed.view(*feat.shape[:-1], -1).permute(0, 3, 1, 2)
                        new_features[key] = transformed
                    self.last_features.append(new_features)
            else:
                if self.motion_cfg['model'] == 'FRCNNSeq2Seq':
                    # save whole image as feature
                    self.last_features.append(self.obj_detect.preprocessed_images.tensors)
                else:
                    if self.motion_cfg['feature_level'] is not None:
                        self.last_features.append(self.obj_detect.features[self.motion_cfg['feature_level']])
                    else:
                        self.last_features.append(self.obj_detect.features)

        if self.public_detections:
            dets = blob['dets'].squeeze(dim=0)
            if dets.nelement() > 0:
                boxes, scores = self.obj_detect.predict_boxes(dets)
            else:
                boxes = scores = torch.zeros(0).cuda()
        else:
            boxes, scores = self.obj_detect.detect(blob['img'])

        if boxes.nelement() > 0:
            boxes = clip_boxes_to_image(boxes, blob['img'].shape[-2:])

            # Filter out tracks that have too low person score
            inds = torch.gt(scores, self.detection_person_thresh).nonzero().view(-1)
        else:
            inds = torch.zeros(0).cuda()

        if inds.nelement() > 0:
            det_pos = boxes[inds]

            det_scores = scores[inds]
        else:
            det_pos = torch.zeros(0).cuda()
            det_scores = torch.zeros(0).cuda()

        ##################
        # Predict tracks #
        ##################

        # report the times that the motion model took
        mm_time = None
        warp_time = None

        if len(self.tracks):
            t_start = time.time()
            # align
            if self.do_align:
                warp_time = self.align(blob)

            self.tracks = [t for t in self.tracks if t.has_positive_area()]

            # apply motion model
            if self.motion_model_enabled:
                if self.motion_cfg['use_correlation_model'] and self.motion_cfg.get('batched', False):
                    self.motion_correlation_batched(blob)
                elif self.motion_cfg['use_correlation_model']:
                    self.motion_correlation(blob)
                elif self.motion_cfg['use_cva_model']:
                    self.motion_cva(blob)
                elif self.motion_cfg['use_pos_model']:
                    self.motion(blob)
                else:
                    raise ValueError('Motion modeling enabled but no model specified.')

                self.tracks = [t for t in self.tracks if t.has_positive_area()]

                if self.motion_cfg['cva_reid']:
                    for t in self.inactive_tracks:
                        # use CVA-5 to achieve a more stable trajectory
                        # TODO this can be optimized by pre-calculating `mean` once
                        if len(t.last_pos_aligned) > 1:
                            last_pos = torch.cat(list(t.last_pos_aligned), dim=0)
                            diff = last_pos[1:] - last_pos[:-1]
                        else:
                            diff = torch.zeros(1, 4).cuda()

                        mean = diff[-5:, :4].mean(0)
                        t.pos = clip_boxes_to_image(t.pos + mean, blob['img'].shape[-2:])

            mm_time = time.time() - t_start

            # for visualization, save position as predicted by motion model
            for t in self.tracks:
                t.mm_pos = t.pos.clone()

            # regress
            if len(self.tracks):
                person_scores = self.regress_tracks(blob)

            if len(self.tracks):
                # create nms input

                # nms here if tracks overlap
                keep = nms(self.get_pos(), person_scores, self.regression_nms_thresh)

                self.tracks_to_inactive([self.tracks[i] for i in list(range(len(self.tracks))) if i not in keep])

                if keep.nelement() > 0 and self.do_reid:
                        new_features = self.get_appearances(blob)
                        self.add_features(new_features)

        #####################
        # Create new tracks #
        #####################

        # !!! Here NMS is used to filter out detections that are already covered by tracks. This is
        # !!! done by iterating through the active tracks one by one, assigning them a bigger score
        # !!! than 1 (maximum score for detections) and then filtering the detections with NMS.
        # !!! In the paper this is done by calculating the overlap with existing tracks, but the
        # !!! result stays the same.
        if det_pos.nelement() > 0:
            keep = nms(det_pos, det_scores, self.detection_nms_thresh)
            det_pos = det_pos[keep]
            det_scores = det_scores[keep]

            # check with every track in a single run (problem if tracks delete each other)
            for t in self.tracks:
                nms_track_pos = torch.cat([t.pos, det_pos])
                nms_track_scores = torch.cat(
                    [torch.tensor([2.0]).to(det_scores.device), det_scores])
                keep = nms(nms_track_pos, nms_track_scores, self.detection_nms_thresh)

                keep = keep[torch.ge(keep, 1)] - 1

                det_pos = det_pos[keep]
                det_scores = det_scores[keep]
                if keep.nelement() == 0:
                    break

        if det_pos.nelement() > 0:
            new_det_pos = det_pos
            new_det_scores = det_scores

            # try to reidentify tracks
            new_det_pos, new_det_scores, new_det_features = self.reid(blob, new_det_pos, new_det_scores)

            # add new
            if new_det_pos.nelement() > 0:
                self.add(new_det_pos, new_det_scores, new_det_features)

        ####################
        # Generate Results #
        ####################

        for t in self.tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            self.results[t.id][self.im_index] = np.concatenate([t.pos[0].cpu().numpy(), np.array([t.score, 0]), t.mm_pos[0].cpu().numpy()])

        if self.write_inactives:
            for t in self.inactive_tracks:
                # flag inactive tracks with a "1" in last column
                self.results[t.id][self.im_index] = np.concatenate([t.pos[0].cpu().numpy(), np.array([t.score, 1]), t.mm_pos[0].cpu().numpy()])

        ###########
        # Cleanup #
        ###########

        for t in self.inactive_tracks:
            t.count_inactive += 1

        self.inactive_tracks = [
            t for t in self.inactive_tracks if t.has_positive_area() and t.count_inactive <= self.inactive_patience
        ]

        self.im_index += 1
        self.last_image = blob['img'][0]

        return mm_time, warp_time

    def get_results(self):
        return self.results


class Track(object):
    """This class contains all necessary for every individual track."""

    def __init__(self, pos, score, track_id, features, inactive_patience, max_features_num, mm_steps):
        self.id = track_id
        self.pos = pos
        self.score = score
        self.features = deque([features])
        self.ims = deque([])
        self.count_inactive = 0
        self.inactive_patience = inactive_patience
        self.max_features_num = max_features_num
        self.last_pos = deque([], maxlen=mm_steps + 1)
        self.last_pos_inact = deque([], maxlen=mm_steps + 1)
        self.last_pos_aligned = deque([], maxlen=mm_steps + 1)
        self.last_v = torch.zeros(4).cuda()
        self.mm_pos = torch.zeros(1, 4).cuda()
        self.before_align_pos = None
        self.gt_id = None
        self.feature_translations = []

    def has_positive_area(self):
        return self.pos[0, 2] > self.pos[0, 0] and self.pos[0, 3] > self.pos[0, 1]

    def add_features(self, features):
        """Adds new appearance features to the object."""
        self.features.append(features)
        if len(self.features) > self.max_features_num:
            self.features.popleft()

    def test_features(self, test_features):
        """Compares test_features to features of this Track object"""
        if len(self.features) > 1:
            features = torch.cat(list(self.features), dim=0)
        else:
            features = self.features[0]
        features = features.mean(0, keepdim=True)
        dist = F.pairwise_distance(features, test_features, keepdim=True)
        return dist

    def reset_last_pos(self):
        self.last_pos.clear()
        self.last_pos.append(self.pos.clone())

        self.last_pos_aligned.clear()
        self.last_pos_aligned.append(self.pos.clone())
