import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: relu (non-inplace aten) → sigmoid
# _decomposed graphs normalize relu(inplace=True) → aten.relu.default
# and torch.sigmoid → aten.sigmoid.default
# ---------------------------------------------------------------------------
def pattern(in_0):
    tmp_0 = torch.ops.aten.relu.default(in_0)
    tmp_1 = torch.ops.aten.sigmoid.default(tmp_0)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


# ---------------------------------------------------------------------------
# Fused Triton kernel: relu + sigmoid in one pass over memory
# sigmoid(relu(x)) = sigmoid(max(0, x))
#   x >= 0  →  1 / (1 + exp(-x))
#   x <  0  →  1 / (1 + exp(0))  = 0.5
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128},  num_warps=2),
        triton.Config({"BLOCK_SIZE": 256},  num_warps=2),
        triton.Config({"BLOCK_SIZE": 512},  num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def fused_relu_sigmoid_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)

    # ReLU: max(0, x)
    x = tl.maximum(x, 0.0)

    # Sigmoid: 1 / (1 + exp(-x))
    out = 1.0 / (1.0 + tl.exp(-x))

    tl.store(out_ptr + offsets, out, mask=mask)


# ---------------------------------------------------------------------------
# Kernel wrapper – must be decorated with @torch.fx.wrap
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_relu_sigmoid(in_0):
    n = in_0.numel()
    out = torch.empty_like(in_0)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    fused_relu_sigmoid_kernel[grid](in_0, out, n)

    return out


# ---------------------------------------------------------------------------
# Replacement entry-point (zero-argument, returns callable)
# ---------------------------------------------------------------------------
def replacement_func():
    return fused_relu_sigmoid