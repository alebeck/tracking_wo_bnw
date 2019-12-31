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

        feature_params = dict(maxCorners=3000, qualityLevel=0.01, minDistance=2, blockSize=7)
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        color = (0, 255, 0)

        data_loader = DataLoader(seq, batch_size=1, shuffle=False)
        flows = []
        for i, frame in enumerate(tqdm(data_loader)):
            current_img = np.transpose(frame['img'][0].cpu().numpy(), (1, 2, 0))
            save_path = osp.join(img_output_dir, osp.basename(frame['img_path'][0]))
            Image.fromarray(frame['img'][0].cpu().numpy().astype('uint8')).save(save_path)
            current_gray = cv2.cvtColor(current_img, cv2.COLOR_RGB2GRAY)

            if i == 0:
                mask = np.zeros_like(current_img)
                prev_gray = current_gray.copy()
                prev = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

            next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray.astype('uint8'), current_gray.astype('uint8'), prev, None, **lk_params)
            good_old = prev[status == 1]
            good_new = next[status == 1]
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                # Returns a contiguous flattened array as (x, y) coordinates for new point
                a, b = new.ravel()
                # Returns a contiguous flattened array as (x, y) coordinates for old point
                c, d = old.ravel()
                # Draws line between new and old position with green color and 2 thickness
                mask = cv2.line(mask, (a, b), (c, d), color, 2)
                # Draws filled circle (thickness of -1) at new position with green color and radius of 3
                current_img = cv2.circle(current_img, (a, b), 3, color, -1)

            output = cv2.add(current_img, mask)
            prev_gray = current_gray.copy()
            prev = good_new.reshape(-1, 1, 2)

            if tracktor['write_images']:
                save_path = osp.join(img_output_dir, osp.basename(frame['img_path'][0]))
                cv2.imwrite(save_path, current_img)

        #write_optical_flow(results, warps, sequence, osp.join(output_dir, tracktor['dataset'], str(sequence)))

    print("[*] Evaluation for all sets (without image generation): {:.3f} s".format(time_total))
