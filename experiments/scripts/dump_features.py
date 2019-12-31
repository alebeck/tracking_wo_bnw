from __future__ import absolute_import, division, print_function

import os
from os import path as osp
import time
import yaml

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sacred import Experiment
import h5py

from tracktor.config import get_output_dir
from tracktor.datasets.factory import Datasets


ex = Experiment()

ex.add_config('experiments/cfgs/tracktor.yaml')

# hacky workaround to load the corresponding configs and not having to hardcode paths here
ex.add_config(ex.configurations[0]._conf['tracktor']['reid_network_config'])
ex.add_config(ex.configurations[0]._conf['tracktor']['obj_detect_config'])
ex.add_named_config('oracle', 'experiments/cfgs/oracle_tracktor.yaml')


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

    print("[*] Beginning operation...")

    layers = ['p2', 'p3', 'p4', 'p5']

    f_hdf5 = h5py.File('/usr/stud/beckera/tracking_wo_bnw/data/motion/im_features.hdf5', 'w')
    i_hdf5 = h5py.File('/usr/stud/beckera/tracking_wo_bnw/data/motion/images.hdf5', 'w')

    for sequence in Datasets(tracktor['dataset']):
        print("[*] Storing sequence: {}".format(sequence))
        f_group = f_hdf5.create_group(sequence._seq_name)
        i_group = i_hdf5.create_group(sequence._seq_name)

        data_loader = DataLoader(sequence, batch_size=1, shuffle=False)
        for i, frame in enumerate(data_loader):
            if i == 0:
                i_group.create_dataset('data', shape=(len(data_loader), *frame['data'][0].shape[1:]), dtype='float16')
                i_group.create_dataset('app_data', shape=(len(data_loader), *frame['app_data'][0].shape[1:]), dtype='float16')
                i_group.create_dataset('im_info', shape=(len(data_loader), 3), dtype='float16')
            i_group['data'][i] = frame['data'][0].cpu().numpy()
            i_group['app_data'][i] = frame['app_data'][0].cpu().numpy()
            i_group['im_info'][i] = frame['im_info'].cpu().numpy()

            image = Variable(frame['data'][0].permute(0, 3, 1, 2).cuda(), volatile=True)
            features = obj_detect.get_features(image)

            for j, layer in enumerate(layers):
                if i == 0:
                    f_group.create_dataset(layer, shape=(len(data_loader), *features[j].shape[1:]), dtype='float16')
                f_group[layer][i] = features[j].data.cpu().numpy().astype('float16')

    f_hdf5.close()
    i_hdf5.close()
