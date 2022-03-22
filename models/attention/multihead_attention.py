# -*- coding: utf-8 -*-
# ---------------------

import torch
import torch.nn as nn
from math import sqrt

from models.attention.attention_heads import *
from conf import Conf


class VanillaAttention(nn.Module):

	def __init__(self, cnf):
		# type: (Conf) -> ()
		super().__init__()

		self.head_dim = cnf.attention.head_dim
		dropout = cnf.attention.dropout
		self.dropout = nn.Dropout(dropout)
		atn_softmax = nn.Softmax(dim=-1)
		self.use_autocast = cnf.attention.get("autocast", False)
		self.add_module('atn_softmax', atn_softmax)

	def forward(self, q, k, v, mask=None):
		# type: (torch.tensor, torch.tensor, torch.tensor, torch.tensor) -> torch.tensor

		# E = (Q^T)*K -> (bs, h, q, head_dim) * (bs, h, head_dim, k) = (bs. h, q, k)
		energy = torch.matmul(q, torch.transpose(k, -2, -1))
		energy = energy / sqrt(self.head_dim)

		#with torch.cuda.amp.autocast(enabled=False):
		# Mask to avoid attending padding or future elements (in decoder attn)
		if mask is not None:
			energy = energy - 1e6 * (1 - mask.float()[:, None, None, :])

		attention = self.atn_softmax(energy)

		# E*V -> (bs. h, q, k) * (bs, h, v, head_dim) -> (bs, h, q, head_dim)
		# (note: k=v always!)
		x = torch.matmul(self.dropout(attention), v.float())

		return x


class MultiHeadAttention(nn.Module):
	def __init__(self, cnf):
		# type: (Conf) -> None
		"""
		:param cnf: Configuration object
		"""

		super().__init__()
		self.cnf = cnf

		self.n_heads = cnf.attention.n_heads
		self.head_dim = cnf.attention.head_dim
		self.hid_dim = self.head_dim * self.n_heads
		emb_dim = cnf.attention.emb_dim

		kv_emb_dim = cnf.attention.get("kv_emb_dim", None)
		if kv_emb_dim is None:
			kv_emb_dim = emb_dim

		self.fc_q = nn.Linear(emb_dim, self.hid_dim)
		self.fc_k = nn.Linear(kv_emb_dim, self.hid_dim)
		self.fc_v = nn.Linear(kv_emb_dim, self.hid_dim)
		self.fc_o = nn.Linear(self.hid_dim, self.hid_dim)

		self.atn_head = eval(cnf.attention.type)(cnf)

		#self._reset_parameters()

	def _reset_parameters(self):
		# type: () -> None
		"""
		https://github.com/pytorch/fairseq/blob/master/fairseq/modules/multihead_attention.py
		:return:
		"""
		# scaled initialization
		nn.init.xavier_uniform_(self.fc_k.weight)
		nn.init.xavier_uniform_(self.fc_v.weight)
		nn.init.xavier_uniform_(self.fc_q.weight)
		nn.init.constant_(self.fc_q.bias, 0.0)
		nn.init.constant_(self.fc_k.bias, 0.0)
		nn.init.constant_(self.fc_v.bias, 0.0)
		nn.init.constant_(self.fc_o.bias, 0.0)

	def forward(self, X, mask=None, enc_ebs=None):
		# type: (torch.tensor, torch.tensor, torch.tensor) -> torch.tensor
		"""
		:param X: (bs, q, emb_dim)
		:param enc_ebs: (bs, e, kv_emb_dim)
		:return: (bs, q, head_dim*n_heads)
		"""

		batch_size = X.shape[0]
		dtype = X.dtype

		# (bs, q/k/v_len, emb_dim) -> (bs, q/k/v_len, hid_dim)
		q = self.fc_q(X)
		if enc_ebs is None:
			k = self.fc_k(X)
			v = self.fc_v(X)
		else:
			k = self.fc_k(enc_ebs)
			v = self.fc_v(enc_ebs)

		# (bs, q/v_len, hid_dim) -> (bs, h, q/v_len, head_dim)
		q = q.reshape(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
		v = v.reshape(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
		k = k.reshape(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

		with torch.cuda.amp.autocast(enabled=False):
			x = self.atn_head(q.float(), k.float(), v.float(), mask.float() if mask is not None else mask)
		x = x.to(dtype)

		# (bs, h, q, head_dim) -> (bs, q, h, head_dim) -> (bs, q, hid_dim)
		x = x.transpose(1, 2).reshape(batch_size, -1, self.hid_dim)

		# (bs, q, hid_dim) -> (bs, q, hid_dim)
		x = self.fc_o(x)

		return x


if __name__ == '__main__':

	from time import time
	from pytorch_memlab import MemReporter

	cnf = Conf(exp_name='pretrain_small_dct_2', log=False)
	#cnf.attention.type = "LinformerAttention"
	#cnf.attention.linformer_k = 12

	model = MultiHeadAttention(cnf, ).cuda()
	model.eval()

	with torch.no_grad():
		times = []
		for _ in range(10):
			q = torch.rand((4, 128, 512), dtype=torch.float32).cuda()
			t = time()
			mask = torch.ones([4,128], dtype=torch.float32).cuda()
			out = model(q, mask=mask)
			times.append(time() - t)

		print(sum(times) / len(times))
		print(out.shape)

	q = torch.rand((4, 128, 512), dtype=torch.float32).to(cnf.device)

	reporter = MemReporter(model)
	out = model(q)
	reporter.report(verbose=True)
