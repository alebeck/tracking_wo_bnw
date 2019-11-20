import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class Seq2Seq(nn.Module):
	GO = torch.Tensor([[0., 0, 0, 0, 1, 0]])

	def __init__(
			self,
			input_size,
			hidden_size,
			output_size,
			input_length,
			n_layers=1,
			dropout=0.
	):
		super().__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.input_length = input_length
		self.n_layers = n_layers

		self.encoder = nn.LSTM(self.input_size, self.hidden_size, batch_first=True, num_layers=self.n_layers, dropout=dropout)

		self.attn = nn.Linear(self.hidden_size + self.output_size, self.input_length)
		self.attn_combine = nn.Linear(self.hidden_size + self.output_size, self.hidden_size)

		self.decoder = nn.LSTM(self.input_size + self.hidden_size, self.hidden_size, batch_first=True, num_layers=self.n_layers, dropout=dropout)
		self.linear1 = nn.Linear(self.hidden_size, self.hidden_size)
		self.linear2 = nn.Linear(self.hidden_size, self.output_size)

	def forward(self, x, target, teacher_forcing=False):
		B = x.shape[0]
		encoder_out = self.encoder(x)  # encoder_out[0]: 32, 60, 48
		last_h = encoder_out[1][0]
		last_c = torch.zeros(self.n_layers, B, self.hidden_size).cuda()
		decoder_in = torch.cat([self.GO.unsqueeze(0)] * B).cuda()
		out_seq = []

		for i in range(target.shape[1]):
			attn_weights = F.softmax(
				self.attn(torch.cat([last_h[0], decoder_in.squeeze(1)], dim=1)), dim=1)  # 32, 60
			attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_out[0])  # 32, 1, 48
			decoder_in = torch.cat([decoder_in, attn_applied], dim=2)  # 32, 1, 54

			_, (last_h, last_c) = self.decoder(decoder_in, (last_h, last_c))
			out = self.linear2(F.relu(self.linear1(last_h.sum(dim=0))))
			out = out.unsqueeze(1)  # add sequence dimension
			out_seq.append(out)

			if teacher_forcing:
				decoder_in = target[:, i, :].unsqueeze(1)
			else:
				decoder_in = out.detach()

		return torch.cat(out_seq, dim=1)

	def predict(self, x, output_length):
		x = Variable(x)
		B = x.shape[0]
		encoder_out = self.encoder(x)  # encoder_out[0]: 32, 60, 48
		last_h = encoder_out[1][0]
		last_c = Variable(torch.zeros(self.n_layers, B, self.hidden_size).cuda())
		decoder_in = Variable(torch.cat([self.GO.unsqueeze(0)] * B).cuda())
		out_seq = []

		for i in range(output_length):
			attn_weights = F.softmax(
				self.attn(torch.cat([last_h[0], decoder_in.squeeze(1)], dim=1)), dim=1)  # 32, 60
			attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_out[0])  # 32, 1, 48
			decoder_in = torch.cat([decoder_in, attn_applied], dim=2)  # 32, 1, 54

			_, (last_h, last_c) = self.decoder(decoder_in, (last_h, last_c))
			out = self.linear2(F.relu(self.linear1(last_h.sum(dim=0))))
			out = out.unsqueeze(1)  # add sequence dimension
			out_seq.append(out)

			decoder_in = out

		return torch.cat(out_seq, dim=1)
