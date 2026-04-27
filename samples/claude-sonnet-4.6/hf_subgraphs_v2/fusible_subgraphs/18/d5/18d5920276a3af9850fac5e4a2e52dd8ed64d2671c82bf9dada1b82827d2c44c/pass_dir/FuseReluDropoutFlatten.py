import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Strategy: FX traces INTO F.dropout(p=0.0, training=False) statically and
# eliminates the node. We match ONLY relu; the Triton replacement fuses
# relu + flatten into one kernel, returning [B, C].
# Downstream dropout(identity) and flatten(no-op on 2D) leave the result
# unchanged, so the final [B, C] output is semantically correct.
# ---------------------------------------------------------------------------

def pattern(x):
    # Match dropout(identity) + flatten together so their replacement
    # receives x = relu output (the INPUT to dropout).
    tmp = torch.nn.functional.dropout(x, 0.0, False, False)
    return tmp.flatten(1, -1)


def replacement_args(x):
    # x = relu output (the tensor fed into dropout, since dropout is identity)
    return (x,)


# ---------------------------------------------------------------------------
# Triton kernel: element-wise ReLU (idempotent on relu output) + flatten.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _relu_flatten_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # ReLU is idempotent on the relu output; this is effectively a copy.
    out = tl.where(x > 0, x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def relu_flatten_wrapper(in_0):
    B = in_0.shape[0]
    n_elements = in_0.numel()
    out = torch.empty((B, n_elements // B), dtype=in_0.dtype, device=in_0.device)
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _relu_flatten_kernel[grid](in_0, out, n_elements)
    return out


def replacement_func():
    return relu_flatten_wrapper