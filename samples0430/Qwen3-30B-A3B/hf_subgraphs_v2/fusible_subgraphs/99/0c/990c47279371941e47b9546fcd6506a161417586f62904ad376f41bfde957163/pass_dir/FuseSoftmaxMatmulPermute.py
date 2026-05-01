import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    scaled = 0.0625 * in_0
    softmax = torch.nn.functional.softmax(scaled, dim=-1)
    matmul = torch.matmul(softmax, in_1)
    permuted = matmul.permute(0, 2, 1)
    return permuted

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_softmax_matmul_kernel(in_0_ptr, in_1_ptr, out_ptr, B, S, K, D, BLOCK_SIZE: tl.constexpr):
    # Each program handles a single (batch, sequence) position
    batch_id = tl.program_id(0)
    seq_id = tl.program_id(1)

    # Load in_0 for this (batch, seq)
    in_0_addr = in_0_ptr + batch_id * S * K + seq_id * K
    in_0_vals = tl.load(in_0_addr, shape=(K,), mask=tl.arange(0, K) < K)

    # Apply scaling
    in_0_scaled = in_0_vals * 0.0625

    # Compute softmax normalization
    max_val = tl.max(in_0_scaled)
    exp_vals = tl.exp(in_0_scaled - max_val)
    exp_sum = tl.sum(exp_vals)
    weights = exp_vals / exp_sum

    # Load in_1 for this batch
    in_1_addr = in_1_ptr + batch_id * K * D
    in_1_vals = tl.load(in_1_addr, shape=(K, D), mask=tl.arange(0, K)[:, None] < K)

    # Compute dot product: [K] dot [K, D] = [D]
    out_vals = tl.dot(weights, in_1_vals)

    # Store output
    out_addr = out_ptr + batch_id * D * S + seq_id
    tl.store(out_addr + tl.arange(0, D) * S, out_vals)

@torch.fx.wrap
def fused_softmax_matmul_wrapper(in_0, in_1):
    B, S, K = in_0.shape
    _, _, D = in_1.shape

    # Output shape: [B, D, S]
    out = torch.empty((B, D, S), dtype=in_0.dtype, device=in_0.device)

    # Choose optimal block size based on work (K is small)
    BLOCK_SIZE = 32

    # Grid setup: one block per (batch, sequence) position
    grid = (B, (S + BLOCK_SIZE - 1) // BLOCK_SIZE)

    # Launch kernel
    fused_softmax_matmul_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        B=B,
        S=S,
        K=K,
        D=D,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

def replacement_func():
    return fused_softmax_matmul_wrapper