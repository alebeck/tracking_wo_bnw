from pathlib import Path
import pickle
import sys

import numpy as np
from sklearn.decomposition import IncrementalPCA


assert len(sys.argv) == 5

INPUT_PATH = Path(sys.argv[1])
N_COMPONENTS = int(sys.argv[2])
SEQS = eval(sys.argv[3])
NO = sys.argv[4]
SAVE_PATH = Path(f'output/ipca_{N_COMPONENTS}_{NO}.pkl')
assert SAVE_PATH.parent.exists()

print(f'Using sequences {SEQS}')


ipca = IncrementalPCA(n_components=N_COMPONENTS)

for seq in SEQS:
    print(f'Processing sequence {seq}')
    f = np.load(INPUT_PATH / f'{seq}-features.npy', mmap_mode='r')
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
