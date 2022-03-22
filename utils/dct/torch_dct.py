# -*- coding: utf-8 -*-
# ---------------------

"""
based on: https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py
4d14d54decb70ea7f922bedaeaf4f4617fad218c

replaced calls to torch with torch.fft..... to support PyTorch >=1.10
"""

import numpy as np
import torch
import math

def create_dct(n, m=None):

    I = torch.eye(n)
    Q = dct(I, dim=0)

    if m is not None:
        Q = Q[:m,:]

    return Q

def create_dct_2(n, m):
    x = torch.arange(float(n))
    u = torch.arange(float(m)).unsqueeze(1)
    dct = torch.cos((math.pi * ((2 * x) + 1) * u) / (2*n))  # size (n_mfcc, n_mels)

    dct[0] *= math.sqrt(1/n)
    dct[1:] *= math.sqrt(2/n)

    return dct

def dct(src, dim=-1, norm='ortho'):
    # type: (torch.tensor, int, str) -> torch.tensor

    x = src.clone()
    N = x.shape[dim]

    x = x.transpose(dim, -1)
    x_shape = x.shape
    x = x.contiguous().view(-1, N)

    v = torch.empty_like(x, device=x.device)
    v[..., :(N - 1) // 2 + 1] = x[..., ::2]

    if N % 2:  # odd length
        v[..., (N - 1) // 2 + 1:] = x.flip(-1)[..., 1::2]
    else:  # even length
        v[..., (N - 1) // 2 + 1:] = x.flip(-1)[..., ::2]

    V = torch.fft.fft(v, dim=-1)

    k = torch.arange(N, device=x.device)
    V = 2 * V * torch.exp(-1j * np.pi * k / (2 * N))

    if norm == 'ortho':
        V[..., 0] *= math.sqrt(1/(4*N))
        V[..., 1:] *= math.sqrt(1/(2*N))

    V = V.real
    V = V.view(*x_shape).transpose(-1, dim)

    return V

def idct(src, dim=-1, norm='ortho'):
    # type: (torch.tensor, int, str) -> torch.tensor

    X = src.clone()
    N = X.shape[dim]

    X = X.transpose(dim, -1)
    X_shape = X.shape
    X = X.contiguous().view(-1, N)

    if norm == 'ortho':
        X[..., 0] *= 1 / math.sqrt(2)
        X *= N*math.sqrt((2 / N))
    else:
        raise Exception("idct with norm=None is buggy A.F")

    k = torch.arange(N, device=X.device)

    X = X * torch.exp(1j * np.pi * k / (2 * N))
    X = torch.fft.ifft(X, dim=-1).real
    v = torch.empty_like(X, device=X.device)

    v[..., ::2] = X[..., :(N - 1) // 2 + 1]
    v[..., 1::2] = X[..., (N - 1) // 2 + 1:].flip(-1)

    v = v.view(*X_shape).transpose(-1, dim)

    return v

def dct_2d(x, norm='ortho'):

    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)

def idct_2d(X, norm='ortho'):

    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


if __name__ == '__main__':

    from time import time
    import matplotlib.pyplot as plt

    from torch.nn.functional import softmax

    times = []
    A = torch.rand(128, 128) * 100
    Q = create_dct_2(128, 128)
    Q2 = create_dct(128, 128)

    for _ in range(100):
        t = time()

        a = torch.matmul(Q, A)
        b = torch.matmul(Q.t(), a)

        error = torch.abs(A-b)
        print(error.mean())
        assert error.max() < 1e-3, (error, error.max())

        d = time() - t
        times.append(d)
        break

    print(sum(times) / 100)

"""
    from time import time
    import scipy.fftpack as fftpack

    times = []
    A = torch.rand(128,128)
    Q = create_dct(128, 32)

    for _ in range(100):
        t = time()

        a = torch.matmul(Q,torch.matmul(A, Q.t()))
        b = fftpack.dctn(A.numpy(), norm='ortho')[:32,:32]

        error = np.abs(a.numpy()-b)
        print(error.mean())
        assert error.max() < 1e-3, (error, error.max())

        d = time() - t
        times.append(d)
        break

    print(sum(times) / 100)
"""
