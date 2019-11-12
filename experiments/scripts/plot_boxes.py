from pathlib import Path
import pickle

from sacred import Experiment

from tracktor.config import get_output_dir
from tracktor.datasets.factory import Datasets
from tracktor.utils import plot_sequence


ex = Experiment()
ex.add_config('experiments/cfgs/plotter.yaml')

@ex.automain
def my_main(plotter, _config):
	output_dir = Path(get_output_dir(plotter['module_name'])) / plotter['name']

	for sequence in Datasets(plotter['dataset']):
		for file in Path(plotter['boxes_dir']).glob('*.pkl'):
			with file.open('rb') as fh:
				data = pickle.load(fh)[sequence._seq_name + '-FRCNN']

			plot_sequence(data, sequence, output_dir / plotter['dataset'] / str(sequence) / str(file.stem))
