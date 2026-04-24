import torch
import triton
import triton.language as tl
import inspect as _inspect


# ---------------------------------------------------------------------------
# Pattern: relu -> dropout(p=0,training=False) -> flatten(1,-1)
#
# The compiled graph (symbolic_trace) has these exact call signatures:
#   torch.nn.functional.relu(in_0, inplace=False)  → args=(in_0,), kwargs={'inplace':False}
#   torch.nn.functional.dropout(tmp_0, 0.0, False, False) → args=(tmp_0,0.0,F,F), kwargs={}
#   tmp_1.flatten(1, -1) → call_method('flatten', args=(tmp_1,1,-1))
#
# ForceArgsTracer on relu: inplace is keyword-only → args=(proxy,), kwargs={}  (same as compiled!)
# ForceArgsTracer on dropout: all args positional → args=(proxy,0.0,F,F), kwargs={}  (same as compiled!)
# ForceArgsTracer on flatten: call_method → args=(proxy,1,-1), kwargs={}  (same as compiled!)
# ---------------------------------------------------------------------------

def pattern(in_0):
    # Test: just dropout + flatten to isolate whether the issue is relu
    # The compiled graph's flatten has dropout as its direct predecessor
    tmp_1 = torch.nn.functional.dropout(in_0, 0.0, False, False)
    tmp_2 = tmp_1.flatten(1, -1)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


# ---------------------------------------------------------------------------
# Triton kernel: elementwise ReLU over flat input, output written to a buffer
# shaped as the flattened result.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _fused_relu_flatten_kernel(
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
    # ReLU: max(x, 0)
    out = tl.where(x > 0, x, tl.zeros_like(x))
    tl.store(out_ptr + offsets, out, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper (marked as a leaf for torch.fx tracing)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_relu_flatten(in_0):
    batch = in_0.shape[0]
    # flatten dims 1 to -1: output shape = [batch, C*H*W]
    out = torch.empty((batch, in_0.numel() // batch), dtype=in_0.dtype, device=in_0.device)
    n_elements = in_0.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _fused_relu_flatten_kernel[grid](in_0, out, n_elements)
    return out


# ---------------------------------------------------------------------------
# replacement_func: returns the callable (NOT a call result)
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_relu_flatten