import torch
import triton
import triton.language as tl

# Simple pattern that just matches float conversion of position_ids
def pattern(in_3):
    tmp_19 = in_3[(slice(None, None, None), None, slice(None, None, None))]
    tmp_20 = tmp_19.float()
    tmp_22 = tmp_20.float()
    return tmp_22

def replacement_args(in_3):
    return (in_3, "route_pos_cast")

@triton.jit
def position_ids_cast_kernel(
    pos_ids_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    pos_ids = tl.load(pos_ids_ptr + offsets, mask=mask, other=0)
    pos_ids_f = pos_ids.to(tl.float32)
    tl.store(out_ptr + offsets, pos_ids_f, mask=mask)

@torch.fx.wrap
def _position_ids_cast(in_3):
    out_pos_shape = list(in_3.shape)
    out_pos_shape.insert(1, 1)
    out_pos = torch.empty(out_pos_shape, dtype=torch.float32, device='cuda:0')
    n_pos = in_3.numel()
    BLOCK_SIZE = 256
    num_pos_programs = (n_pos + BLOCK_SIZE - 1) // BLOCK_SIZE
    position_ids_cast_kernel[(num_pos_programs,)](
        pos_ids_ptr=in_3, out_ptr=out_pos, n_elements=n_pos, BLOCK_SIZE=BLOCK_SIZE,
    )
    return out_pos

@torch.fx.wrap
def dispatch_wrapper(in_3, route):
    if route == "route_pos_cast":
        return _position_ids_cast(in_3)
    else:
        raise ValueError(f"Unknown route: {route}")

def replacement_func():
    return dispatch_wrapper