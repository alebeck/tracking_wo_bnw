from pathlib import Path
import configparser
import math

import numpy as np
import torch
from torch import nn
from PIL import Image
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip

from tracktor.frcnn_fpn import FRCNN_FPN
from tqdm import tqdm

MIN_SIZE = 800
MAX_SIZE = 1333
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
DATA_PATH = Path("/usr/stud/beckera/tracking_wo_bnw/data/MOT17Det")
WEIGHTS_PATH = "output/faster_rcnn_fpn_training_mot_17/model_epoch_27.model"
OUTPUT_PATH = Path('data/features-fp16-flip')
SEQUENCES = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']
FEATURE_LEVEL = 1
DO_FLIP = True
FP16 = True

rcnn_transform: GeneralizedRCNNTransform = GeneralizedRCNNTransform(MIN_SIZE, MAX_SIZE, IMAGE_MEAN, IMAGE_STD)
if DO_FLIP:
    transform = Compose([RandomHorizontalFlip(1.), ToTensor()])
else:
    transform = ToTensor()

obj_detect: nn.Module = FRCNN_FPN(num_classes=2)
obj_detect.load_state_dict(torch.load(WEIGHTS_PATH))
obj_detect.eval().cuda()


# inspired by torchvision.models.detection.transform.GeneralizedRCNNTransform
def batch_images(images, max_size, size_divisible=32):
    stride = size_divisible
    max_size = list(max_size)
    max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
    max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)
    max_size = tuple(max_size)

    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).zero_()
    for img, pad_img in zip(images, batched_imgs):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    return batched_imgs


# read one image for each sequence to determine max_size
image_shapes = []
for seq in SEQUENCES:
    seq_path = DATA_PATH / 'train' / seq
    config_file = seq_path / 'seqinfo.ini'
    assert config_file.exists(), f'Config file does not exist: {config_file}'

    config = configparser.ConfigParser()
    config.read(config_file)
    im_dir = seq_path / config['Sequence']['imDir']
    gt_file = seq_path / 'gt' / 'gt.txt'

    im_path = im_dir / '000001.jpg'
    img = transform(Image.open(im_path).convert('RGB'))
    img = rcnn_transform.normalize(img)
    img, _ = rcnn_transform.resize(img, None)

    image_shapes.append(img.shape)

max_size = tuple(max(s) for s in zip(*image_shapes))

for seq in SEQUENCES:
    seq_path = DATA_PATH / 'train' / seq
    config_file = seq_path / 'seqinfo.ini'
    assert config_file.exists(), f'Config file does not exist: {config_file}'

    config = configparser.ConfigParser()
    config.read(config_file)
    seq_length = int(config['Sequence']['seqLength'])
    im_dir = seq_path / config['Sequence']['imDir']
    gt_file = seq_path / 'gt' / 'gt.txt'

    print(f'Serializing sequence {seq} with {seq_length} frames...')

    original_image_sizes = []
    image_sizes = []
    feature_maps = []
    with torch.no_grad():
        for i in tqdm(range(1, seq_length + 1)):
            im_path = im_dir / "{:06d}.jpg".format(i)
            img = transform(Image.open(im_path).convert('RGB'))
            original_image_sizes.append(img.shape[-2:])
            img = rcnn_transform.normalize(img)
            img, _ = rcnn_transform.resize(img, None)
            image_sizes.append(img.shape[-2:])
            img = batch_images([img], max_size)[0]
            features = obj_detect.backbone(img.cuda().unsqueeze(0))[FEATURE_LEVEL].detach().cpu()
            feature_maps.append(features.half() if FP16 else features)

    feature_maps = torch.cat(feature_maps)
    np.save(OUTPUT_PATH / f'{seq}-features', feature_maps.numpy())
    np.save(OUTPUT_PATH / f'{seq}-origsizes', torch.tensor(original_image_sizes).numpy())
    np.save(OUTPUT_PATH / f'{seq}-sizes', torch.tensor(image_sizes).numpy())
