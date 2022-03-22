# -*- coding: utf-8 -*-
# ---------------------

import torch
import torch.nn as nn

from math import sqrt, floor

from utils.dct import dct, idct
from models.attention import VanillaAttention
from utils.dct import create_dct
from conf import Conf


class DCT_MHSA_Naive(nn.Module):

	Q_dct = None

	def __init__(self, cnf):
		# type: (Conf) -> None
		"""
		:param cnf: Configuration object
		"""

		super().__init__()
		self.cnf = cnf

		# This can be removed by inheriting from MultiHeadAttention
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

		self.atn_head = VanillaAttention(cnf)
		# -----------------

		# Only Vanilla Attention head
		assert cnf.attention.type == "VanillaAttention"

		# DCT stuff
		self.max_n = cnf.attention.dct.get("maxN", None)
		self.max_m = cnf.attention.dct.get("maxM", None)

		# Initialize class variable
		if DCT_MHSA_Naive.Q_dct is None and self.max_n is not None:
			DCT_MHSA_Naive.Q_dct = create_dct(n=self.max_n, m=self.max_m).to(cnf.device)


	def forward(self, X, mask=None):
		# type: (torch.tensor, torch.tensor) -> torch.tensor
		"""
		:param X: (bs, q, emb_dim)
		:return: (bs, q, head_dim*n_heads)
		"""

		x_shape = X.shape
		dtype = X.dtype
		batch_size = x_shape[0]

		# Encode with DCT and downsample
		pad = 0
		if self.max_n is not None:
			X_dct = torch.matmul(self.Q_dct, X.float()).to(dtype)
		else:
			pad = max(0, X.shape[-2] - self.max_m)
			X_dct = dct(X.float(), dim=-2)[:, :self.max_m, :].to(dtype)

		# (bs, q/k/v_len, emb_dim) -> (bs, q/k/v_len, hid_dim)
		q = self.fc_q(X_dct)
		k = self.fc_k(X_dct)
		v = self.fc_v(X_dct)

		# (bs, q/v_len, hid_dim) -> (bs, h, q/v_len, head_dim)
		q = q.reshape(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
		v = v.reshape(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
		k = k.reshape(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

		with torch.cuda.amp.autocast(enabled=False):
			X_dct = self.atn_head(q.float(), k.float(), v.float(), mask=None)
		X_dct = X_dct.to(dtype)

		# (bs, h, q, head_dim) -> (bs, q, h, head_dim) -> (bs, q, hid_dim)
		X_dct = X_dct.transpose(1, 2).reshape(batch_size, -1, self.hid_dim).contiguous()

		# Upsample and decode with inverse DCT
		if self.max_n is not None:
			x = torch.matmul(self.Q_dct.t(), X_dct.float()).to(dtype)
		else:
			x = torch.nn.ConstantPad1d((0, pad), 0)(X_dct.transpose(1,2)).transpose(1,2)
			x = idct(x.float(), dim=-2).to(dtype)

		# (bs, q, hid_dim) -> (bs, q, hid_dim)
		x = self.fc_o(x)

		return x

	def __del__(self):
		DCT_MHSA_Naive.Q_dct = None

if __name__ == '__main__':

	from time import time
	from pytorch_memlab import MemReporter

	cnf = Conf(exp_name='pretrain_small_dct_0', log=False)
	cnf.attention.dct.n_scale_method = "sqrt"

	model = DCT_MHSA_Naive(cnf).to(cnf.device)
	model.eval()

	with torch.no_grad():
		times = []
		for _ in range(10):
			q = torch.rand((4, 128, 512), dtype=torch.float32).to(cnf.device)
			t = time()
			out = model(q)
			times.append(time() - t)

		print(sum(times) / len(times))
		print(out.shape)

	q = torch.rand((4, 128, 512), dtype=torch.float32).to(cnf.device)

	reporter = MemReporter(model)
	out = model(q)
	reporter.report(verbose=True)

