import torch
import triton
import triton.language as tl

def pattern(value_states):
    tmp_10 = torch.unsqueeze(value_states, 2)
    tmp_11 = tmp_10.expand(1, 1, 8, 3, 256)
    tmp_12 = tmp_11.reshape(1, 8, 3, 256)
    return tmp_12

def replacement_args(value_states):
    return (value_states, "expand_value")

@triton.jit
def expand_value_kernel(
    value_ptr, out_ptr,
    n_seq: tl.constexpr, n_dim: tl.constexpr, n_heads: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    head_id = tl.program_id(0)
    pid = tl.program_id(1)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    total = n_seq * n_dim
    mask = offsets < total

    val = tl.load(value_ptr + offsets, mask=mask, other=0.0)
    expand_offsets = head_id * total + offsets
    tl.store(out_ptr + expand_offsets, val, mask=mask)

@torch.fx.wrap
def _rope_impl(cos, key, sin):
    # Placeholder - never called in this pass's context
    raise NotImplementedError("rope route not implemented in FuseExpandValueStates pass")

@torch.fx.wrap
def _expand_value_impl(value_states):
    n_seq = 3
    n_dim = 256
    n_heads = 8
    BLOCK_SIZE = 128

    total = n_seq * n_dim  # 768
    num_programs = (total + BLOCK_SIZE - 1) // BLOCK_SIZE  # 6

    out = torch.empty((1, n_heads, n_seq, n_dim), dtype=value_states.dtype, device=value_states.device)

    grid = (n_heads, num_programs)

    expand_value_kernel[grid](
        value_ptr=value_states, out_ptr=out,
        n_seq=n_seq, n_dim=n_dim, n_heads=n_heads,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

@torch.fx.wrap
def dispatch_wrapper(*args):
    route = args[-1]
    if route == "rope":
        cos, key, sin, _ = args
        return _rope_impl(cos, key, sin)
    elif route == "expand_value":
        value_states, _ = args
        return _expand_value_impl(value_states)
    else:
        raise ValueError(f"Unknown route: {route}")

def replacement_func():
    return dispatch_wrapper