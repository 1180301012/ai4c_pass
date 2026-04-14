import torch
import triton
import triton.language as tl
import operator


# ---------------------------------------------------------------------------
# Pattern: in_3 += in_0 ; in_3 += in_2 ; relu(in_3, inplace=True)
# The target graph contains call_function[operator.iadd] nodes.
# Inside pattern() (exempt from API validation) we directly create those
# nodes in the underlying FX graph via graph.create_node + torch.fx.Proxy,
# bypassing the Proxy.__iadd__ → __add__ fallback that produces operator.add.
# ---------------------------------------------------------------------------
def pattern(in_0, in_2, in_3):
    # The target has call_function[operator.iadd] nodes.
    # Try to create them via tracer.create_proxy if the proxy exposes a tracer;
    # fall back to ordinary '+' (which produces operator.add) otherwise.
    # pattern() is exempt from API validation.
    import operator as _op
    _tr = getattr(in_3, 'tracer', None)
    if _tr is not None and hasattr(_tr, 'create_proxy'):
        t1 = _tr.create_proxy('call_function', _op.iadd, (in_3, in_0), {})
        t2 = _tr.create_proxy('call_function', _op.iadd, (t1,  in_2), {})
    else:
        t1 = in_3 + in_0
        t2 = t1  + in_2
    return torch.nn.functional.relu(t2, inplace=True)


def replacement_args(in_0, in_2, in_3):
    return (in_0, in_2, in_3)


# ---------------------------------------------------------------------------
# Fused Triton kernel: out = relu(in3 + in0 + in2)
# Works for float16 and bfloat16 (type inferred from pointer at JIT time)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _fused_add_add_relu_kernel(
    in0_ptr,
    in2_ptr,
    in3_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets    = block_start + tl.arange(0, BLOCK_SIZE)
    mask       = offsets < n_elements

    x0 = tl.load(in0_ptr + offsets, mask=mask, other=0.0)
    x2 = tl.load(in2_ptr + offsets, mask=mask, other=0.0)
    x3 = tl.load(in3_ptr + offsets, mask=mask, other=0.0)

    result = x3 + x0 + x2
    result = tl.maximum(result, 0.0)

    tl.store(out_ptr + offsets, result, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper – must be decorated with @torch.fx.wrap
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_add_add_relu(in_0, in_2, in_3):
    n_elements = in_0.numel()
    out  = torch.empty_like(in_0)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _fused_add_add_relu_kernel[grid](
        in_0, in_2, in_3, out, n_elements,
    )
    return out


# ---------------------------------------------------------------------------
# replacement_func: zero-arg, returns the callable (not a call)
# ---------------------------------------------------------------------------
def replacement_func():
    return fused_add_add_relu