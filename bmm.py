#!/usr/bin/env python3
# Batch matmul kernel
# Adapted from the 2D matmul tutorial: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
import torch

import triton
import triton.language as tl
import unittest


@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr = 16, BLOCK_SIZE_N: tl.constexpr = 16, BLOCK_SIZE_K: tl.constexpr = 16,  #
        GROUP_SIZE_M: tl.constexpr = 8,  #
        #    ACTIVATION: tl.constexpr = "",  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    # pid = tl.program_id(axis=1)
    # Batch = tl.program_id(axis=0)

    batch_pid = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_abatch = batch_pid * M * K
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = (a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak) + offs_abatch)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cbatch = batch_pid * M * N
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = (c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :] + offs_cbatch)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b, activation=""):
    # Check constraints.
    assert a.shape[2] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"

    Batch, M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((Batch, M, N), device=a.device, dtype=a.dtype)

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        Batch,
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        c.stride(1),
        c.stride(2),
        num_warps=1,
        num_stages=1,
    )
    return c


class TestTritonMatmul(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.atol = 1.6e-2
        self.rtol = 1e-7

    def test_one_batched_matmul(self):
        batch_size = 1
        size = 16
        a = torch.randn((batch_size, size, size), device="cuda", dtype=torch.float16)
        b = torch.randn((size, size), device="cuda", dtype=torch.float16)
        triton_output = matmul(a, b)
        torch_output = torch.matmul(a, b)
        self.assertTrue(torch.allclose(triton_output, torch_output, atol=self.atol, rtol=self.rtol))

    def test_two_batched_matmul(self):
        batch_size = 2
        size = 16
        a = torch.randn((batch_size, size, size), device="cuda", dtype=torch.float16)
        b = torch.randn((size, size), device="cuda", dtype=torch.float16)
        triton_output = matmul(a, b)
        torch_output = torch.matmul(a, b)
        self.assertTrue(torch.allclose(triton_output, torch_output, atol=self.atol, rtol=self.rtol))

    def test_three_batched_matmul(self):
        batch_size = 3
        size = 16
        a = torch.randn((batch_size, size, size), device="cuda", dtype=torch.float16)
        b = torch.randn((size, size), device="cuda", dtype=torch.float16)
        triton_output = matmul(a, b)
        torch_output = torch.matmul(a, b)
        self.assertTrue(torch.allclose(triton_output, torch_output, atol=self.atol, rtol=self.rtol))

    def test_varying_sizes(self):
        for batch_size in [1, 2, 4]:
            for M in [8, 32, 64]:
                for N in [8, 32, 64]:
                    for K in [8, 32, 64]:
                        a = torch.randn((batch_size, M, K), device="cuda", dtype=torch.float16)
                        b = torch.randn((K, N), device="cuda", dtype=torch.float16)
                        triton_output = matmul(a, b)
                        torch_output = torch.matmul(a, b)

                        passed = torch.allclose(
                            triton_output,
                            torch_output,
                            atol=self.atol,
                            rtol=self.rtol,
                        )
                        if not passed:
                            print(f"oh snap we failed: batch_size={batch_size}, M={M}, N={N}, K={K}")
                            print("triton_output:", triton_output)
                            print("torch_output:", torch_output)
                            import numpy as np

                            np.save("/tmp/array.npy", triton_output.cpu().numpy())
                            np.save("/tmp/target.npy", torch_output.cpu().numpy())
                            tensor = torch_output - triton_output
                            non_zero_values = tensor[tensor != 0]
                            print(non_zero_values)
                        self.assertTrue(
                            torch.allclose(
                                triton_output,
                                torch_output,
                                atol=self.atol,
                                rtol=self.rtol,
                            ),
                            f"Failed for batch_size={batch_size}, N={N}, M={M}, K={K}",
                        )


batch_sizes = [1, 4, 8, 16, 32]
mnk_values = [64, 128, 256, 512]
# batch_sizes = [1, 4]
# mnk_values = [64, 128]
x_vals = ((batch, m, n, k) for batch in batch_sizes for m in mnk_values for n in mnk_values for k in mnk_values)
x_vals_list = list(x_vals)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=[
            "Batch",
            "M",
            "N",
            "K",
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
def benchmark(Batch, M, N, K, provider):
    torch.cuda.empty_cache()
    a = torch.randn((Batch, M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "cublas":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * Batch * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    # Comment out tests of benchmark as desired
    unittest.main()
    # benchmark.run(show_plots=False, print_data=True, save_path="results/")

    # individual test
    # torch.manual_seed(0)
    # atol = 1e-2
    # batch_size = 4
    # N = 32
    # M = 64
    # K = 64

    # a = torch.randn((batch_size, M, K), device="cuda", dtype=torch.float16)
    # b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    # triton_output = matmul(a, b)
    # torch_output = torch.matmul(a, b)
    # passed = torch.allclose(triton_output, torch_output, atol=atol)
    # print("pass!", passed)
