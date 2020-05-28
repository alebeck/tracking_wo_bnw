from torch import nn
from torch.utils.data import DataLoader

from .config import TrainingConfig


class TrainingContext:
	config: TrainingConfig
	model: nn.Module
	data_loader: DataLoader
	epoch: int
	use_cuda: bool
	validate: bool
	log_path: str
