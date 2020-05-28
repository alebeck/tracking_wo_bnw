import logging
from pathlib import Path
from datetime import datetime
from typing import Union
from abc import abstractmethod

import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler, DataLoader
from torch import optim

from .config import TrainingConfig
from .context import TrainingContext
from .utils import setup_logging, get_train_logger, dump_yaml_config, load_yaml_config


# initializations
setup_logging()
_context = TrainingContext()


class TorchTrainer:

	@abstractmethod
	def epoch(self):
		raise NotImplementedError

	def __init__(self, config: Union[TrainingConfig, str]):
		if isinstance(config, str):
			# load yaml config from file
			config = load_yaml_config(Path(config))

		self._config = config
		self._model = self._config.model(**config.model_args)
		self._epoch_start = 0

		# load checkpoint
		if self._config.resume:
			self._checkpoint = torch.load(self._config.resume)
			self._model.load_state_dict(self._checkpoint['model'])
			self._epoch_start = self._checkpoint['epoch']

		# check CUDA availability
		self._use_cuda = self._config.use_cuda and torch.cuda.is_available()
		if self._use_cuda:
			self._model.cuda()

		# update _context
		_context.cfg = self._config
		_context.model = self._model
		_context.use_cuda = self._use_cuda

	def train(self):
		log_path = Path(self._config.log_path) / str(datetime.now())
		log_path.mkdir(parents=True, exist_ok=True)
		_context.log_path = log_path

		logging.info('Instantiating data set')

		if isinstance(self._config.batch_size, tuple):
			batch_size_train, batch_size_val = self._config.batch_size
		else:
			batch_size_train, batch_size_val = (self._config.batch_size,) * 2

		if isinstance(self._config.data, type) and isinstance(self._config.data_args, dict):
			if not isinstance(self._config.val_size, float):
				err_str = '"val_size" has to be specified when using automated train/val split.'
				logging.error(err_str)
				raise ValueError(err_str)

			ds = self._config.data(**self._config.data_args)
			idx = np.arange(len(ds))
			np.random.shuffle(idx)
			split = int(self._config.val_size * len(ds))
			train_idx, val_idx = idx[split:], idx[:split]
			train_loader = DataLoader(ds, batch_size_train, sampler=SubsetRandomSampler(train_idx),
									num_workers=self._config.num_workers, collate_fn=self._config.collate_fn,
									pin_memory=self._config.pin_memory)
			val_loader = DataLoader(ds, batch_size_val, sampler=SubsetRandomSampler(val_idx),
									num_workers=self._config.num_workers, collate_fn=self._config.collate_fn,
									pin_memory=self._config.pin_memory)

		elif isinstance(self._config.data, list) and isinstance(self._config.data_args, list):
			train_ds = self._config.data[0](**self._config.data_args[0])
			val_ds = self._config.data[1](**self._config.data_args[1])
			train_loader = DataLoader(train_ds, batch_size_train, num_workers=self._config.num_workers, shuffle=True,
									collate_fn=self._config.collate_fn, pin_memory=self._config.pin_memory)
			val_loader = DataLoader(val_ds, batch_size_val, num_workers=self._config.num_workers, shuffle=True,
									collate_fn=self._config.collate_fn, pin_memory=self._config.pin_memory)
		else:
			err_str = 'Received invalid argument combination for "data" and "data_args". "data" can be one of ' + \
						'[type, tuple(type, type)] and "data_args" one of [dict, tuple(dict, dict)], respectively.'
			logging.error(err_str)
			raise ValueError(err_str)

		if self._config.resume:
			logging.info('Resuming training from checkpoint')
			# resume optimizer state
			if self._config.resume_optimizer:
				self._set_optim_states(self._checkpoint['optim'])
		else:
			logging.info('Training started')

		if self._use_cuda:
			logging.info('Using CUDA')

		# dump config
		dump_yaml_config(self._config, log_path / 'config.yaml')

		train_logger = get_train_logger(log_path)
		best_val = float('inf') if self._config.smaller_is_better else 0.

		for epoch in range(self._epoch_start, self._config.epochs):
			# train model
			_context.epoch = epoch
			_context.data_loader = train_loader
			_context.validate = False

			self._model.train()
			train_metrics = self.epoch()
			train_logger.info(
				' '.join([f'{k}: {v}' for k, v in train_metrics.items()]),
				extra={'epoch': epoch, 'fwpass': 'Train'}
			)

			# validate model
			if epoch % self._config.val_every == 0:
				_context.data_loader = val_loader
				_context.validate = True

				self._model.eval()
				with torch.no_grad():
					val_metrics = self.epoch()
				train_logger.info(
					' '.join([f'{k}: {v}' for k, v in val_metrics.items()]),
					extra={'epoch': epoch, 'fwpass': 'Val'}
				)

				if self._config.primary_metric is None and len(val_metrics.keys()) == 1:
					metric = list(val_metrics.keys())[0]
					logging.info(f'Using "{metric}" as primary metric')
					self._config.primary_metric = metric
				elif self._config.primary_metric is None:
					err_str = 'Received multiple metrics but no primary metric is defined.'
					logging.error(err_str)
					raise ValueError(err_str)

			# save checkpoint
			if self._config.save_every or \
				(self._config.smaller_is_better and val_metrics[self._config.primary_metric] < best_val) or \
				(not self._config.smaller_is_better and val_metrics[self._config.primary_metric] > best_val):

				logging.info(f'Saving checkpoint at epoch {epoch}')

				checkpoint = {
					'model': self._model.state_dict(),
					'optim': self._get_optim_states(),
					'epoch': epoch + 1
				}
				torch.save(checkpoint, log_path / f'checkpoint_{epoch}.pt')
				best_val = val_metrics[self._config.primary_metric]

		logging.info(f'Training finished')

	def _get_optim_states(self):
		return { k: v.state_dict() for k, v in self.__dict__.items() if issubclass(type(v), optim.Optimizer) }

	def _set_optim_states(self, state_dict):
		for k in state_dict:
			getattr(self, k).load_state_dict(state_dict[k])
