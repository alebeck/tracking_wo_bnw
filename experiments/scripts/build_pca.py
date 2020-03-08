from pathlib import Path
import pickle
import sys

import numpy as np
from sklearn.decomposition import IncrementalPCA


N_COMPONENTS = int(sys.argv[1])
SAVE_PATH = Path(f'output/ipca_{N_COMPONENTS}.pkl')
assert SAVE_PATH.parent.exists()

SEQS = [
    'MOT17-02',
    'MOT17-04',
    'MOT17-05',
    'MOT17-09',
    'MOT17-10',
    'MOT17-11',
    'MOT17-13',
]


ipca = IncrementalPCA(n_components=N_COMPONENTS)

for seq in SEQS:
    f = np.load(f'data/features/{seq}-features.npy', mmap_mode='r')
    # go through images in 150 frame steps
    idx = 0
    while idx < f.shape[0] - 1:
        win = f[idx:idx + 150].copy()
        win = win.transpose((0, 2, 3, 1))
        win = win.reshape((-1, 256))

        # partial fit on window
        ipca.partial_fit(win)

        idx += 150

print(f'Samples seen: {ipca.n_samples_seen_}')
print(f'Explained variance: {ipca.explained_variance_ratio_.sum()}')

with SAVE_PATH.open('wb') as fh:
    pickle.dump(ipca, fh)
