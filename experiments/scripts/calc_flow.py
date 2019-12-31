from __future__ import absolute_import, division, print_function

import os
from os import path as osp
import yaml

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sacred import Experiment
import cv2
from PIL import Image

from tracktor.config import get_output_dir
from tracktor.datasets.factory import Datasets
#from tracktor.utils import write_optical_flow

ex = Experiment()

ex.add_config('experiments/cfgs/tracktor.yaml')

@ex.automain
def my_main(tracktor, _config):
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

    print("[*] Beginning process...")

    for seq in Datasets(tracktor['dataset']):

        print(f"[*] Processing sequence {seq}")

        img_output_dir = osp.join(output_dir, tracktor['dataset'], str(seq))
        if tracktor['write_images'] and not osp.exists(img_output_dir):
            os.makedirs(img_output_dir)

        data_loader = DataLoader(seq, batch_size=1, shuffle=False)
        flows = []
        for i, frame in enumerate(tqdm(data_loader)):
            current_img = np.transpose(frame['img'][0].cpu().numpy(), (1, 2, 0))

            if i == 0:
                prev_img = current_img

            current_gray = cv2.cvtColor(current_img, cv2.COLOR_RGB2GRAY)
            prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            flows.append(flow)

            if tracktor['write_images']:
                mask = np.zeros_like(current_img)
                mask[..., 1] = 255
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                mask[..., 0] = angle * 180 / np.pi / 2
                mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
                rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2RGB)
                save_path = osp.join(img_output_dir, osp.basename(frame['img_path'][0]))
                Image.fromarray(rgb.astype('uint8')).save(save_path)

            prev_img = current_img


        #write_optical_flow(results, warps, sequence, osp.join(output_dir, tracktor['dataset'], str(sequence)))

    print("[*] Evaluation for all sets (without image generation): {:.3f} s".format(time_total))
