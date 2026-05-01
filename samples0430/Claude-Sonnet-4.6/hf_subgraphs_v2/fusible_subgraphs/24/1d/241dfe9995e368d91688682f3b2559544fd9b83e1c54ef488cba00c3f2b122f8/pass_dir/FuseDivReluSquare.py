import torch
import triton
import triton.language as tl
import inspect as _inspect

# ForceArgsTracer normalizes function args by calling inspect.signature(target).bind(*args, **kwargs).
# torch.nn.functional.relu has signature (input, inplace=False).
# handle_torch_function passes inplace=False as a kwarg → after normalization: args=(proxy, False).
# But dynamo's gm has args=(tmp_0,) for relu (no inplace kwarg).
# Fix: override relu's signature so that inplace goes into **kwargs (VAR_KEYWORD),
# which is excluded from bound.args, giving normalized_args=(proxy,) to match the model.
_relu_sig_fixed = _inspect.Signature([
    _inspect.Parameter("input",  _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    _inspect.Parameter("_kw",    _inspect.Parameter.VAR_KEYWORD),   # absorbs inplace=False
])
torch.nn.functional.relu.__signature__ = _relu_sig_fixed


# ── pattern ──────────────────────────────────────────────────────────────────
def pattern(in_0):
    tmp_0 = in_0 / 11.313708498984761
    tmp_1 = torch.nn.functional.relu(tmp_0)
    tmp_2 = torch.square(tmp_1)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


# ── Triton kernel ─────────────────────────────────────────────────────────────
@triton.jit
def _fused_relu_sq_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # divide  (1/11.313708498984761 = 1/sqrt(128))
    x = x * 0.08838834764831843

    # relu
    x = tl.maximum(x, 0.0)

    # square
    x = x * x

    tl.store(out_ptr + offsets, x, mask=mask)


# ── wrapper ───────────────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_relu_sq(in_0):
    N = in_0.numel()
    out = torch.empty_like(in_0)
    # Use a fixed BLOCK_SIZE of 1024 to minimize autotune overhead;
    # for small tensors this still saturates the GPU adequately.
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _fused_relu_sq_kernel[grid](
        x_ptr=in_0,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
# ── replacement entry-point ───────────────────────────────────────────────────
def replacement_func():
    return fused_relu_sq