#!/usr/bin/env python3
import torch
import triton
import triton.language as tl
import unittest


@triton.jit
def transpose_kernel(
    y_ptr,  # N x M
    x_ptr,  # M x N
    stride_yn,
    stride_ym,
    stride_xm,
    stride_xn,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr = 16,
    BLOCK_SIZE_N: tl.constexpr = 16,
    GROUP_SIZE_M: tl.constexpr = 8,
):
    # Compute the input and output pointers
    pid = tl.program_id(0)
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
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    x_ptrs = x_ptr + (offs_am[:, None] * stride_xm + offs_n[None, :] * stride_xn)
    y_ptrs = y_ptr + (offs_n[:, None] * stride_yn + offs_bm[None, :] * stride_ym)

    a = tl.load(x_ptrs)
    a_t = tl.trans(a)  # transpose just a block of data
    tl.store(y_ptrs, a_t)


def triton_transpose(x):
    n_rows, n_cols = x.shape
    M, N = n_rows, n_cols
    # Move tensors to GPU (if they're not already there)
    x = x.cuda()

    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    # Allocate output
    y = torch.empty_like(x.transpose(0, 1))

    transpose_kernel[grid](
        y,
        x,
        y.stride(0),
        y.stride(1),
        x.stride(0),
        x.stride(1),
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

    def test_one_tile_tranpose(self):
        rows, cols = 16, 16

        a = torch.arange(rows * cols, device="cuda", dtype=torch.int16).reshape(rows, cols)
        triton_output = triton_transpose(a)
        torch_output = a.transpose(0, 1)

        self.assertTrue(torch.allclose(triton_output, torch_output, atol=self.atol))

    def test_two_tile_tranpose(self):
        rows, cols = 32, 16
        a = torch.arange(rows * cols, device="cuda", dtype=torch.int16).reshape(rows, cols)
        triton_output = triton_transpose(a)
        torch_output = a.transpose(0, 1)

        self.assertTrue(torch.allclose(triton_output, torch_output, atol=self.atol))

    def test_varying_size_tranpose(self):
        for M in [16, 32, 64, 128, 256, 512]:
            for N in [16, 32, 64, 128, 256, 512]:
                a = torch.randn((M, N), device="cuda", dtype=torch.float16)
                triton_output = triton_transpose(a)
                torch_output = a.transpose(0, 1)
                self.assertTrue(torch.allclose(triton_output, torch_output, atol=self.atol))


if __name__ == "__main__":
    unittest.main()
