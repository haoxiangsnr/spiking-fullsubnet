import math

import numpy as np
import torch


EPS = torch.as_tensor(torch.finfo(torch.get_default_dtype()).eps)
PI = math.pi


def db2mag(fdB):
    """converts dB to magnitude"""
    return 10 ** (fdB / 20)


class STFTTorch:
    """
    class used to simplify handling of STFT & iSTFT
    """

    def __init__(
        self,
        frame_length=64,
        overlap_length=48,
        window=torch.hann_window,
        fs=16000,
    ):
        self.num_bins = int((frame_length / 2) + 1)
        self.fs = fs
        self.frame_length = frame_length
        self.overlap_length = overlap_length
        self.window = window(self.frame_length)
        self.shift_length = self.frame_length - self.overlap_length

        self.params = {
            "frame_length": frame_length,
            "overlap_length": overlap_length,
            "window": window,
        }

    def get_stft(self, wave):
        return torch.stft(
            wave,
            window=self.window.to(device=wave.device),
            n_fft=self.frame_length,
            hop_length=self.shift_length,
            win_length=self.frame_length,
            normalized=False,
            center=True,
            pad_mode="constant",
        )

    def get_istft(self, stft, length=None):
        return torch.istft(
            stft,
            window=self.window.to(device=stft.device),
            n_fft=self.frame_length,
            hop_length=self.shift_length,
            win_length=self.frame_length,
            normalized=False,
            center=True,
            length=length,
        )


def complex_tensor_exponential(t):
    """return complex exponential of input with shape ... x 2"""
    exp_real = torch.exp(t[..., 0])
    real = exp_real * torch.cos(t[..., 1])
    imag = exp_real * torch.sin(t[..., 1])
    return torch.stack([real, imag], dim=-1)


def vector_to_Hermitian(vec):
    """
    this function constructs a ... x N x N x 2-dim. Hermitian matrix from a vector of N**2 independent real-valued components
    broadcasting enabled
    input:
        vec: ... x N**2
    output:
        mat: ... x N x N x 2
    """
    N = int(np.sqrt(vec.shape[-1]))
    mat = torch.zeros(size=vec.shape[:-1] + (N, N, 2), device=vec.device)

    # real component
    triu = np.triu_indices(N, 0)
    triu2 = np.triu_indices(N, 1)  # above main diagonal
    tril = (triu2[1], triu2[0])  # below main diagonal; for symmetry
    mat[(...,) + triu + (np.zeros(triu[0].shape[0]),)] = vec[..., : triu[0].shape[0]]
    start = triu[0].shape[0]
    mat[(...,) + tril + (np.zeros(tril[0].shape[0]),)] = mat[(...,) + triu2 + (np.zeros(triu2[0].shape[0]),)]

    # imaginary component
    mat[(...,) + triu2 + (np.ones(triu2[0].shape[0]),)] = vec[..., start : start + triu2[0].shape[0]]
    mat[(...,) + tril + (np.ones(tril[0].shape[0]),)] = -mat[(...,) + triu2 + (np.ones(triu2[0].shape[0]),)]

    return mat


def complex_tensor_hermitian(t: torch.Tensor) -> torch.Tensor:
    """return Hermitian of t

    Args:
        t (torch.Tensor): ... x N x N x 2

    Returns:
        torch.Tensor: Hermitian of t; ... x N x N x 2
    """
    return complex_tensor_conj(t.transpose(-3, -2))


def get_mvdr(gammax, Phi):
    """
    compute conventional MPDR/MVDR filter
    :param gammax:
    :param Phi:
    :return: filter coefficients filter
    computes filter coefficients;
    gammax: KxMx2, Phi: KxMxMx2
    """
    b = complex_solve_matrix_vector(Phi, gammax)
    denom = complex_tensor_inner_product(gammax, b)
    return complex_tensor_division(b, denom[..., None, :] + EPS)


def complex_solve_matrix_vector(A, b):
    """
    solves a complex system of linear equations
    07.10.19: validated against numpy
    """
    A_big = torch.cat(
        (
            torch.cat((A[..., 0], -A[..., 1]), dim=-1),
            torch.cat((A[..., 1], A[..., 0]), dim=-1),
        ),
        dim=-2,
    )
    b_big = torch.cat((b[..., 0], b[..., 1]), dim=-1)
    x_big = torch.solve(b_big[..., None], A_big)[0][..., 0]  # ignore singleton dimension
    length = int(x_big.shape[-1] / 2)
    return torch.stack((x_big[..., :length], x_big[..., length:]), dim=-1)


def complex_solve_matrix_matrix(mat1, mat2):
    """
    solve system of linear equations specified by
    mat1 X = mat2 <-> X = mat1^-1 mat2
    Args:
        mat1 ([torch.tensor]): ... m x n x 2 (real and imaginary components in last dimension)
        mat2 ([torch.tensor]): ... n x o x 2

    Returns:
        [torch.tensor]: X = mat1^-1 mat2; ... x m x o
    """
    n = mat1.shape[-2]
    mat1_big = torch.cat(
        (
            torch.cat((mat1[..., 0], -mat1[..., 1]), dim=-1),
            torch.cat((mat1[..., 1], mat1[..., 0]), dim=-1),
        ),
        dim=-2,
    )
    mat2_big = torch.cat((mat2[..., 0], mat2[..., 1]), dim=-2)
    solution_big = torch.solve(mat2_big, mat1_big)[0]
    return torch.stack((solution_big[..., :n, :], solution_big[..., n:, :]), dim=-1)


def filter_(mWeights, mSTFTNoisyAdj):
    """
    performs filtering on STFT using mWeights
    mWeights: batch_size x K x time_steps x AdjFrames
    """
    return complex_tensor_inner_product(mWeights, mSTFTNoisyAdj)


def filter_minimum_gain_like(G_min, w, y, alpha=None, k=10.0):
    """
    approximate a minimum gain operation as
    speech_estimate = alpha w^H y + (1 - alpha) G_min Y,
    where alpha = 1 / (1 + exp(-2 k x)), x = w^H y - G_min Y
    inputs:
        - G_min:    minimum gain, float
        - w:        complex-valued filter coefficients, ... x L x N x 2
        - y:        buffered and stacked input, ... x L x N x 2
        - k:        scaling in tanh-like function
        - alpha:    mixing factor
    outputs:
        - minimum gain-filtered output
        - (optional) alpha
    """
    filtered_input = complex_tensor_inner_product(w, y)
    Y = y[..., -1, :]
    return minimum_gain_like(G_min, Y, filtered_input, alpha, k)


def minimum_gain_like(G_min, Y, filtered_input, alpha=None, k=10.0):
    if alpha is None:
        alpha = 1.0 / (1.0 + torch.exp(-2 * k * (complex_tensor_abs(filtered_input) - complex_tensor_abs(G_min * Y))))
        alpha = alpha[..., None]
        return_alpha = True
    else:
        return_alpha = False
    output = alpha * filtered_input + (1 - alpha) * G_min * Y
    if return_alpha:
        return output, alpha
    else:
        return output


@torch.jit.script
def complex_tensor_hadamard_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    perform multiplication of two complex tensors a and b, with real and imag components in last dimensions
    :param a:
    :param b:
    :return:
    """
    return torch.stack(
        (
            a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1],
            a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0],
        ),
        dim=-1,
    )


@torch.jit.script
def safe_sqrt(tensor, eps=EPS):
    """
    safer version of sqrt, adding EPS to all values to avoid nan
    :param input:
    :param kwargs:
    :return:
    """
    return torch.sqrt(tensor + eps)


@torch.jit.script
def complex_tensor_conj(t: torch.Tensor) -> torch.Tensor:
    """
    return complex conjugate, assuming that real and imag parts lie in last dimension
    :param mat:
    :return:
    """
    return torch.stack((t[..., 0], -t[..., 1]), dim=-1)


@torch.jit.script
def complex_tensor_matrix_vector_product(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """perform multiplication of complex tensor matrix and vector with real and imag components in last dimensions"""
    return torch.stack(
        [
            matrix[..., 0] @ vector[..., 0][..., None] - matrix[..., 1] @ vector[..., 1][..., None],
            matrix[..., 0] @ vector[..., 1][..., None] + matrix[..., 1] @ vector[..., 0][..., None],
        ],
        dim=-1,
    )[..., 0, :]


@torch.jit.script
def complex_tensor_matrix_matrix_product(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """matrix multiplication of complex matrices mat1 and mat2
    Args:
        mat1 ([tensor]): ... x M x N x 2 complex tensor
        mat2 ([tensor]): ... x N x O x 2 complex tensor
    """
    return torch.stack(
        [
            mat1[..., 0] @ mat2[..., 0] - mat1[..., 1] @ mat2[..., 1],
            mat1[..., 0] @ mat2[..., 1] + mat1[..., 1] @ mat2[..., 0],
        ],
        dim=-1,
    )


def trace(mat: torch.Tensor) -> torch.Tensor:
    """
    returns the trace of mat, taken over the last two dimensions
    :param mat:
    :return:
    """
    return torch.einsum("...ii->...", mat)


@torch.jit.script
def complex_tensor_abs(tIn: torch.Tensor) -> torch.Tensor:
    """
    returns the absolute value of tIn
    tIn: ... x 2, with real and imaginary component on last axis
    """
    return safe_sqrt(tIn[..., 0] ** 2 + tIn[..., 1] ** 2)


@torch.jit.script
def tik_reg(mat: torch.Tensor, reg: float = 0.001) -> torch.Tensor:
    """
    performs Tikhonov regularization
    only modifies real part
    mat: ... x iAdj x 2
    """
    iAdj = mat.shape[-2]  # number of adjacent frames
    temp = ((reg * trace(complex_tensor_abs(mat))) / iAdj)[..., None, None] * torch.eye(iAdj, device=mat.device)[
        None, None, ...
    ]
    return mat + torch.stack([temp, temp.new_zeros(size=(1,)).expand(temp.shape)], dim=-1)


@torch.jit.script
def complex_tensor_division(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    perform division a/b, with a and b complex tensors, with real and imaginary components sitting in last dim.
    :param a:
    :param b:
    :return:
    """
    factor = 1.0 / (b[..., 0] ** 2 + b[..., 1] ** 2)
    result = torch.stack(
        (
            a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1],
            a[..., 1] * b[..., 0] - a[..., 0] * b[..., 1],
        ),
        dim=-1,
    )
    return factor[..., None] * result


@torch.jit.script
def complex_tensor_inner_product(x, y):
    """
    perform inner product of two complex-valued vectors
    """
    xHy = torch.empty(size=x[..., 0, :].shape, device=x.device)
    xHy[..., 0] = torch.sum(x[..., 0] * y[..., 0] + x[..., 1] * y[..., 1], dim=-1)
    xHy[..., 1] = torch.sum(x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0], dim=-1)
    return xHy


class Bunch(object):
    """
    transforms adict into an object such that adict values can be accessed using "." notation
    """

    def __init__(self, adict) -> None:
        self.__dict__.update(adict)
