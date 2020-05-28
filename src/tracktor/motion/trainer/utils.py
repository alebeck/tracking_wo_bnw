import sys
import logging
from pathlib import Path
from importlib import reload
import yaml

from colorama import Fore, Style

from .config import TrainingConfig


class _TrainProgressFormatter(logging.Formatter):

	def format(self, record):
		msg = f'{Fore.CYAN}[{record.levelname}]{Style.RESET_ALL} {Fore.GREEN if record.fwpass == "Train" else Fore.BLUE}' + \
			f'[Epoch {record.epoch} / {record.fwpass}]{Style.RESET_ALL} {record.msg}'
		return msg


class _ColorFormatter(logging.Formatter):

	def format(self, record):
		if record.levelname == 'INFO':
			color = Fore.CYAN
		elif record.levelname in ['ERROR', 'CRITICAL']:
			color = Fore.RED
		elif record.levelname == 'WARNING':
			color = Fore.YELLOW
		else:
			color = Style.RESET_ALL
		return f'{color}[{record.levelname}]{Style.RESET_ALL} {record.msg}'


def setup_logging():
	# workaround to make logging work in notebooks (https://stackoverflow.com/a/21475297)
	reload(logging)

	stdout_handler = logging.StreamHandler(sys.stdout)
	stdout_handler.setFormatter(_ColorFormatter())
	logging.basicConfig(
		handlers=[stdout_handler],
		level=logging.INFO
	)


def get_train_logger(log_path: Path):
	file_handler = logging.FileHandler(log_path / 'log.txt')
	file_handler.setFormatter(logging.Formatter(f'[Epoch %(epoch)s / %(fwpass)s] %(message)s'))

	stdout_handler = logging.StreamHandler(sys.stdout)
	stdout_handler.setFormatter(_TrainProgressFormatter())

	train_logger = logging.getLogger('train_logger')
	train_logger.handlers = []
	train_logger.propagate = False
	train_logger.setLevel(logging.INFO)
	train_logger.addHandler(file_handler)
	train_logger.addHandler(stdout_handler)

	return train_logger


def dump_yaml_config(obj: object, path: Path):
	string = yaml.dump(vars(obj), sort_keys=False)
	with path.open('w+') as f:
		f.write(string)


def load_yaml_config(path: Path):
	with path.open() as f:
		string = f.read()
	return TrainingConfig(**yaml.load(string, Loader=yaml.FullLoader))
