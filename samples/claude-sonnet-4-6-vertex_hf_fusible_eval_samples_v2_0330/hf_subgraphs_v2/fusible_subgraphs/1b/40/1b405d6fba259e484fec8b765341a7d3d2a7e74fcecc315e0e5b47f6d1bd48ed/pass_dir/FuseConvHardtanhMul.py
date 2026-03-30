import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: match hardtanh(in_3, 0, 6) * conv_out
# conv_out is produced by an upstream conv2d node (treated as an opaque input)
# ---------------------------------------------------------------------------

def pattern(conv_out, in_3):
    """
    Match:
        tmp_3 = hardtanh(in_3, 0.0, 6.0, inplace=False)
        tmp_4 = tmp_3 * conv_out
    and return tmp_4.
    """
    tmp_3 = torch.nn.functional.hardtanh(in_3, 0.0, 6.0, False)
    tmp_4 = tmp_3 * conv_out
    return tmp_4


def replacement_args(conv_out, in_3):
    return (conv_out, in_3)


# ---------------------------------------------------------------------------
# Triton kernel: fused hardtanh(a, 0, 6) * b  in a single memory pass
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def relu6_mul_kernel(
    a_ptr,        # pointer to in_3   (flat, contiguous)
    b_ptr,        # pointer to conv_out (flat, contiguous)
    out_ptr,      # pointer to output
    N,            # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)   # in_3
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)   # conv_out

    # hardtanh(a, 0, 6)  ≡  clamp(a, 0, 6)
    a_clamped = tl.minimum(tl.maximum(a, 0.0), 6.0)

    tl.store(out_ptr + offsets, a_clamped * b, mask=mask)


# ---------------------------------------------------------------------------
# Replacement wrapper – @torch.fx.wrap prevents FX from tracing into it
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_relu6_mul(conv_out, in_3):
    """
    Fused replacement for hardtanh(in_3, 0, 6) * conv_out.

    Dispatch strategy:
    • Large tensors (N >= 5 M elements): single Triton kernel fuses clamp + mul,
      saving ~40 % memory bandwidth vs two separate PyTorch kernels.
    • Small tensors (N < 5 M elements): Triton launch overhead (~50 µs) exceeds
      the kernel savings; fall back to lightweight PyTorch elementwise ops.
    """
    N = conv_out.numel()

    if N < 5_000_000:
        # Small-batch path: low-overhead PyTorch ops (same cost as original)
        clamped = in_3.clamp(min=0.0, max=6.0)
        return clamped.mul_(conv_out)

    # Large-batch path: fused Triton kernel (saves ~40% elementwise bandwidth)
    out    = torch.empty_like(conv_out)
    in3_c  = in_3.contiguous()
    conv_c = conv_out.contiguous()
    grid   = lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    relu6_mul_kernel[grid](in3_c, conv_c, out, N)
    return out


def replacement_func():
    return fused_relu6_mul