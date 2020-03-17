from pathlib import Path
import pickle
import sys

import numpy as np


assert len(sys.argv) == 4

PCA_PATH = Path(sys.argv[1])
INPUT_PATH = Path(sys.argv[2])
OUTPUT_PATH = Path(sys.argv[3])
STEP_SIZE = 50
TYPE = 'float16'

SEQS = [
    'MOT17-02',
    'MOT17-04',
    'MOT17-05',
    'MOT17-09',
    'MOT17-10',
    'MOT17-11',
    'MOT17-13',
]


OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

with PCA_PATH.open('rb') as fh:
    ipca = pickle.load(fh)

for seq in SEQS:
    f = np.load(INPUT_PATH / f'{seq}-features.npy', mmap_mode='r')
    out = []

    idx = 0
    while idx < f.shape[0] - 1:
        win = f[idx:idx + STEP_SIZE].copy()

        win = win.transpose(0, 2, 3, 1)  # [B,H,W,C]
        original_shape = win.shape[:-1]
        win = win.reshape(-1, 256)

        transformed = ipca.transform(win).astype(TYPE)
        transformed = transformed.reshape(*original_shape, -1)
        transformed = transformed.transpose(0, 3, 1, 2)
        out.append(transformed)

        idx += STEP_SIZE

    out = np.concatenate(out)
    assert out.shape[0] == f.shape[0] and out.shape[2] == f.shape[2] and out.shape[3] == f.shape[3]
    np.save(OUTPUT_PATH / f'{seq}-features', out)
