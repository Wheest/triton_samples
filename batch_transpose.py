#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import triton
import triton.language as tl
import unittest
import numpy as np


@triton.jit
def batch_transpose_kernel(
    y_ptr,  # Batch x N x M
    x_ptr,  # Batch x M x N
    stride_ybatch,
    stride_yn,
    stride_ym,
    stride_xbatch,
    stride_xm,
    stride_xn,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr = 16,
    BLOCK_SIZE_N: tl.constexpr = 16,
    GROUP_SIZE_M: tl.constexpr = 8,
):
    # Compute the input and output pointers
    pid = tl.program_id(1)
    batch_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # `x_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_N] pointers
    # `y_ptrs` is a block of [BLOCK_SIZE_N, BLOCK_SIZE_M] pointers
    offs_xbatch = batch_pid * stride_xbatch
    offs_ybatch = batch_pid * stride_ybatch
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    x_ptrs = (x_ptr + (offs_am[:, None] * stride_xm + offs_n[None, :] * stride_xn) + offs_xbatch)
    y_ptrs = (y_ptr + (offs_n[:, None] * stride_yn + offs_bm[None, :] * stride_ym) + offs_ybatch)

    a = tl.load(x_ptrs)
    a_t = tl.trans(a)  # transpose just a block of data
    tl.store(y_ptrs, a_t)


def triton_transpose(x):
    Batch, M, N = x.shape

    # Move tensors to GPU (if they're not already there)
    x = x.cuda()

    grid = lambda META: (
        Batch,
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    # Allocate output
    y = torch.empty_like(x.transpose(1, 2))

    batch_transpose_kernel[grid](
        y,
        x,
        y.stride(0),
        y.stride(1),
        y.stride(2),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        M,
        N,
        num_warps=1,
        num_stages=1,
    )
    return y


class TestTritonTranspose(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.atol = 1e-2

    def test_batch_one_tranpose(self):
        rows, cols = 16, 16
        batch = 1

        a = torch.arange(batch * rows * cols, device="cuda", dtype=torch.int16).reshape(batch, rows, cols)
        triton_output = triton_transpose(a)
        torch_output = a.transpose(1, 2)

        self.assertTrue(torch.allclose(triton_output, torch_output, atol=self.atol))

    def test_batch_two_tranpose(self):
        rows, cols = 16, 16
        batch = 2

        a = torch.arange(batch * rows * cols, device="cuda", dtype=torch.int16).reshape(batch, rows, cols)
        triton_output = triton_transpose(a)
        torch_output = a.transpose(1, 2)

        self.assertTrue(torch.allclose(triton_output, torch_output, atol=self.atol))

    def test_varying_size_tranpose(self):
        for batch in [16, 32, 64, 128, 256, 512]:
            for M in [16, 32, 64, 128, 256, 512]:
                for N in [16, 32, 64, 128, 256, 512]:
                    a = torch.randn((batch, M, N), device="cuda", dtype=torch.float16)
                    triton_output = triton_transpose(a)
                    torch_output = a.transpose(1, 2)
                    self.assertTrue(torch.allclose(triton_output, torch_output, atol=self.atol))


if __name__ == "__main__":
    unittest.main()
