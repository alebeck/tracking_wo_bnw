from typing import Optional, List, Type, Union, Tuple

from torch import nn
from torch.utils.data import Dataset


class TrainingConfig:

	def __init__(
			self,
			data: Union[Type[Dataset], List[Type[Dataset]]],
			data_args: Union[dict, List[dict]],
			model: Type[nn.Module],
			model_args: dict,
			batch_size: Union[int, Tuple[int, int]],
			epochs: int,
			val_size: Union[float, None] = None,
			val_every: int = 1,
			# Use CUDA if available
			use_cuda: bool = True,
			# number of CPUs to use for data loading
			num_workers: int = 2,
			# Should be None if not resuming training, otherwise a string path to a checkpoint
			resume: Optional[str] = None,
			# when resuming, also resume optimizer state
			resume_optimizer: bool = True,
			log_path: str = './train_logs',
			# Decide to save model checkpoint based on the following validation metric. Can be None if only one metric is used.
			primary_metric: str = None,
			save_every: bool = False,
			smaller_is_better: bool = True,
			collate_fn=None,
			pin_memory=False,
			**kwargs
	):
		self.data = data
		self.data_args = data_args
		self.model = model
		self.model_args = model_args
		self.batch_size = batch_size
		self.epochs = epochs
		self.val_size = val_size
		self.val_every = val_every
		self.use_cuda = use_cuda
		self.num_workers = num_workers
		self.resume = resume
		self.resume_optimizer = resume_optimizer
		self.log_path = log_path
		self.primary_metric = primary_metric
		self.save_every = save_every
		self.smaller_is_better = smaller_is_better
		self.collate_fn = collate_fn
		self.pin_memory = pin_memory

		for k, v in kwargs.items():
			setattr(self, k, v)

	def __repr__(self):
		return f'{__class__.__name__}(' + ', '.join([f'{k}={repr(v)}' for k, v in vars(self).items()]) + ')'
