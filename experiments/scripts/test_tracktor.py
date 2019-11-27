from __future__ import absolute_import, division, print_function

import os
from os import path as osp
import time
import yaml

import numpy as np
import torch
from torch.utils.data import DataLoader
from sacred import Experiment

from tracktor.config import get_output_dir
from tracktor.datasets.factory import Datasets
from tracktor.oracle_tracker import OracleTracker
from tracktor.resnet import resnet50
from tracktor.motion import Seq2Seq
from tracktor.tracker import Tracker
from tracktor.utils import interpolate, plot_sequence


# Monkey-patch because I trained with a newer version.
# This can be removed once PyTorch 0.4.x is out.
# See https://discuss.pytorch.org/t/question-about-rebuild-tensor-v2/14560
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


ex = Experiment()

ex.add_config('experiments/cfgs/tracktor.yaml')

# hacky workaround to load the corresponding configs and not having to hardcode paths here
ex.add_config(ex.configurations[0]._conf['tracktor']['reid_network_config'])
ex.add_config(ex.configurations[0]._conf['tracktor']['obj_detect_config'])
ex.add_named_config('oracle', 'experiments/cfgs/oracle_tracktor.yaml')

# Tracker = ex.capture(Tracker, prefix='tracker.tracker')

@ex.automain
def my_main(tracktor, siamese, _config):
    # set all seeds
    torch.manual_seed(tracktor['seed'])
    torch.cuda.manual_seed(tracktor['seed'])
    np.random.seed(tracktor['seed'])
    torch.backends.cudnn.deterministic = True

    output_dir = osp.join(get_output_dir(tracktor['module_name']), tracktor['name'])
    sacred_config = osp.join(output_dir, 'sacred_config.yaml')

    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    with open(sacred_config, 'w') as outfile:
        yaml.dump(_config, outfile, default_flow_style=False)

    ##########################
    # Initialize the modules #
    ##########################

    # object detection
    print("[*] Building object detector")
    if tracktor['network'].startswith('frcnn'):
        # FRCNN
        from tracktor.frcnn import FRCNN
        from frcnn.model import config

        if _config['frcnn']['cfg_file']:
            config.cfg_from_file(_config['frcnn']['cfg_file'])
        if _config['frcnn']['set_cfgs']:
            config.cfg_from_list(_config['frcnn']['set_cfgs'])

        obj_detect = FRCNN(num_layers=101)
        obj_detect.create_architecture(2, tag='default',
            anchor_scales=config.cfg.ANCHOR_SCALES,
            anchor_ratios=config.cfg.ANCHOR_RATIOS)
        obj_detect.load_state_dict(torch.load(tracktor['obj_detect_weights']))
    elif tracktor['network'].startswith('fpn'):
        # FPN
        from tracktor.fpn import FPN
        from fpn.model.utils import config
        config.cfg.TRAIN.USE_FLIPPED = False
        config.cfg.CUDA = True
        config.cfg.TRAIN.USE_FLIPPED = False
        checkpoint = torch.load(tracktor['obj_detect_weights'])

        if 'pooling_mode' in checkpoint.keys():
            config.cfg.POOLING_MODE = checkpoint['pooling_mode']

        set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]',
                    'ANCHOR_RATIOS', '[0.5,1,2]']
        config.cfg_from_file(_config['tracktor']['obj_detect_config'])
        config.cfg_from_list(set_cfgs)

        obj_detect = FPN(('__background__', 'pedestrian'), 101, pretrained=False)
        obj_detect.create_architecture()

        obj_detect.load_state_dict(checkpoint['model'])
    else:
        raise NotImplementedError(f"Object detector type not known: {tracktor['network']}")

    obj_detect.eval()
    obj_detect.cuda()

    # reid
    reid_network = resnet50(pretrained=False, **siamese['cnn'])
    reid_network.load_state_dict(torch.load(tracktor['reid_network_weights']))
    reid_network.eval()
    reid_network.cuda()

    # motion network
    motion_network = Seq2Seq(**tracktor['motion']['network_args'])
    motion_network.load_state_dict(torch.load(tracktor['motion']['network_weights'])['model'])
    motion_network.eval().cuda()

    # tracktor
    if 'oracle' in tracktor:
        tracker = OracleTracker(obj_detect, reid_network, tracktor['tracker'], tracktor['oracle'])
    else:
        tracker = Tracker(obj_detect, reid_network, motion_network, tracktor['tracker'], tracktor['motion'])

    print("[*] Beginning evaluation...")

    time_total = 0
    for sequence in Datasets(tracktor['dataset'], **tracktor['dataset_args']):
        tracker.reset()

        now = time.time()

        print("[*] Evaluating: {}".format(sequence))

        data_loader = DataLoader(sequence, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        for i, frame in enumerate(data_loader):
            if len(sequence) * tracktor['frame_split'][0] <= i <= len(sequence) * tracktor['frame_split'][1]:
                tracker.step(frame)
        results = tracker.get_results()

        time_total += time.time() - now

        print("[*] Tracks found: {}".format(len(results)))
        print("[*] Time needed for {} evaluation: {:.3f} s".format(sequence, time.time() - now))

        if tracktor['interpolate']:
            results = interpolate(results)

        sequence.write_results(results, osp.join(output_dir))

        if tracktor['write_images']:
            plot_sequence(results, sequence, osp.join(output_dir, tracktor['dataset'], str(sequence)))

    print("[*] Evaluation for all sets (without image generation): {:.3f} s".format(time_total))
