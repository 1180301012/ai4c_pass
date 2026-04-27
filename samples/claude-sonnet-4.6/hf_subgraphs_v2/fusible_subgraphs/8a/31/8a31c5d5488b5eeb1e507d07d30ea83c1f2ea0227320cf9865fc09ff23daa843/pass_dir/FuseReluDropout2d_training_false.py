import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# The compiled (dynamo) graph for:
#   tmp_0 = relu(in_0, inplace=True)
#   tmp_1 = dropout2d(tmp_0, 0.1, training=False, inplace=False)
#   return (tmp_1, tmp_0)
#
# After torch.compile:
#   • dropout2d(training=False) is a no-op → eliminated from the graph
#   • Both outputs equal the relu result
#   • The graph is: relu(arg0) → result; output = (result, result)
#
# Pattern strategy:
#   • Match ONLY the relu node (since dropout is eliminated)
#   • Return a SINGLE value so returning_nodes has exactly one entry
#   • The replacement also returns a single tensor; replace_all_uses_with
#     automatically updates both references in the output tuple
# ---------------------------------------------------------------------------

def pattern(x):
    # relu(inplace=True) → after functionalization in dynamo → aten.relu.default
    # We try the out-of-place form first; it covers the functionalized case.
    result = torch.nn.functional.relu(x)
    return result


def replacement_args(x):
    return (x,)


# ---------------------------------------------------------------------------
# Triton kernel – memory-bandwidth-bound ReLU
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


# ---------------------------------------------------------------------------
# Kernel wrapper – @torch.fx.wrap makes it a leaf in the replacement graph.
# Returns a SINGLE tensor; the graph's output tuple is reconstructed
# automatically when replace_all_uses_with replaces both references.
# ---------------------------------------------------------------------------

@torch.fx.wrap
def relu_fused(x):
    N = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _relu_kernel[grid](x, out, N)
    return out


# ---------------------------------------------------------------------------
# replacement_func: zero-argument factory that returns the callable
# ---------------------------------------------------------------------------

def replacement_func():
    return relu_fused