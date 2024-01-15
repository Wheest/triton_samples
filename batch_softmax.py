#!/usr/bin/env python3
# Batch softmax kernel
# Adapted from the fused softmax tutorial: https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import triton
import triton.language as tl
import unittest


@triton.jit
def softmax_kernel_batch(
    output_ptr,
    input_ptr,
    input_row_stride,
    input_batch_stride,
    output_row_stride,
    output_batch_stride,
    n_cols,
    n_rows,
    BLOCK_SIZE: tl.constexpr,
):
    # The rows of the softmax are independent, so we parallelize across those
    batch_pid = tl.program_id(axis=0)
    row_idx = tl.program_id(1)

    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride + (batch_pid * input_batch_stride)
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))
    # Subtract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride + (batch_pid * output_batch_stride)
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


def triton_batch_softmax(x):
    batch, n_rows, n_cols = x.shape

    # Move tensors to GPU (if they're not already there)
    x = x.cuda()

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    # Allocate output
    y = torch.empty_like(x)

    # Enqueue kernel. The 2D launch grid is simple: we have one kernel instance per row of
    # the input matrix
    softmax_kernel_batch[(batch, n_rows)](
        y,
        x,
        x.stride(1),
        x.stride(0),
        y.stride(1),
        y.stride(0),
        n_cols,
        n_rows,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


class TestTritonSoftmax(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.atol = 1e-2

    def test_one_batched_softmax(self):
        batch_size = 1
        size = 16
        a = torch.randn((batch_size, size, size), device="cuda", dtype=torch.float16)
        triton_output = triton_batch_softmax(a)
        torch_output = F.softmax(a, dim=-1)
        self.assertTrue(torch.allclose(triton_output, torch_output, atol=self.atol))

    def test_one_batched_asym_softmax(self):
        batch_size = 1
        N, M = 4, 16
        a = torch.randn((batch_size, N, M), device="cuda", dtype=torch.float16)
        triton_output = triton_batch_softmax(a)
        torch_output = F.softmax(a, dim=-1)
        tensor = torch_output - triton_output
        nonzero_values = tensor[tensor != 0]
        self.assertTrue(torch.allclose(triton_output, torch_output, atol=self.atol))

    def test_two_batched_softmax(self):
        batch_size = 2
        size = 16
        a = torch.randn((batch_size, size, size), device="cuda", dtype=torch.float16)
        triton_output = triton_batch_softmax(a)
        torch_output = F.softmax(a, dim=-1)
        self.assertTrue(torch.allclose(triton_output, torch_output, atol=self.atol))

    def test_three_batched_softmax(self):
        batch_size = 3
        size = 16
        a = torch.randn((batch_size, size, size), device="cuda", dtype=torch.float16)
        triton_output = triton_batch_softmax(a)
        torch_output = F.softmax(a, dim=-1)
        self.assertTrue(torch.allclose(triton_output, torch_output, atol=self.atol))

    def test_varying_sizes(self):
        for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            for N in [8, 32, 64, 128, 256]:
                for M in [8, 32, 64, 128, 256]:
                    a = torch.randn((batch_size, N, M), device="cuda", dtype=torch.float16)
                    triton_output = triton_batch_softmax(a)
                    torch_output = F.softmax(a, dim=-1)
                    self.assertTrue(torch.allclose(triton_output, torch_output, atol=self.atol))


batch_sizes = [1, 4, 8, 16, 32]
nm_values = [64, 128, 256, 512]
x_vals = ((batch, n, m) for batch in batch_sizes for n in nm_values for m in nm_values)
x_vals_list = list(x_vals)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=[
            "Batch",
            "N",
            "M",
        ],  # Argument names to use as an x-axis for the plot
        # Different possible values for `x_name`
        x_vals=x_vals_list,  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=["cublas", "triton"],
        # Label name for the lines
        line_names=["cuBLAS", "Triton"],
        # Line styles
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="batch-matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))
def benchmark(Batch, N, M, provider):
    torch.cuda.empty_cache()
    a = torch.randn((Batch, N, M), device="cuda", dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "cublas":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: F.softmax(a, dim=-1), quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_batch_softmax(a), quantiles=quantiles)
    perf = lambda ms: 2 * Batch * M * N * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == '__main__':
    # Comment out tests or benchmark as desired
    # unittest.main()
    benchmark.run(show_plots=False, print_data=True, save_path="results/")
