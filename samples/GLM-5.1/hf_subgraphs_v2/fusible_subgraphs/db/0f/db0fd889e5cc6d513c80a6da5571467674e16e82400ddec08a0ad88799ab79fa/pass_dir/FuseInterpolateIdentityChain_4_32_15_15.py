import torch
import triton
import triton.language as tl


# Match the second interpolate chain (for in_6) that transforms:
# slice -> transpose(2,3) -> view -> interpolate(identity) -> flatten -> transpose(1,2) -> contiguous -> view
def pattern(in_6):
    tmp_28 = in_6[(slice(None, None, None), slice(None, None, None), slice(1, -10, None), slice(None, None, None))]
    tmp_29 = tmp_28.transpose(2, 3)
    tmp_30 = tmp_29.view(4, 32, 15, 15)
    tmp_31 = torch.nn.functional.interpolate(tmp_30, size = (15, 15), mode = 'bicubic', align_corners = False)
    tmp_32 = tmp_31.flatten(2)
    tmp_33 = tmp_32.transpose(1, 2)
    tmp_34 = tmp_33.contiguous()
    tmp_35 = tmp_34.view(4, 1, 225, 32)
    return tmp_35


def replacement_args(in_6):
    return (in_6, "route_interp_chain_2")


@triton.jit
def identity_copy_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x, mask=mask)


@triton.jit
def interp_chain_1_kernel(
    in_ptr,
    out_ptr,
    N,
    H: tl.constexpr,
    W: tl.constexpr,
    SLICE_START: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    batch_idx = offsets // (225 * W)
    seq_idx = (offsets // W) % 225
    feat_idx = offsets % W
    
    in_offsets = batch_idx * H * W + (seq_idx + SLICE_START) * W + feat_idx
    
    x = tl.load(in_ptr + in_offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x, mask=mask)


@triton.jit
def interp_chain_2_kernel(
    in_ptr,
    out_ptr,
    N,
    H: tl.constexpr,
    W: tl.constexpr,
    SLICE_START: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    batch_idx = offsets // (225 * W)
    seq_idx = (offsets // W) % 225
    feat_idx = offsets % W
    
    in_offsets = batch_idx * (1 * H * W) + (seq_idx + SLICE_START) * W + feat_idx
    
    x = tl.load(in_ptr + in_offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x, mask=mask)


def _dropout_identity_impl(tmp_23):
    n_elements = tmp_23.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(tmp_23)
    
    identity_copy_kernel[(num_programs,)](
        in_ptr=tmp_23,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def _interp_chain_1_impl(in_5):
    N = 1 * 225 * 32
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty(1, 225, 32, dtype=in_5.dtype, device=in_5.device)
    
    interp_chain_1_kernel[(num_programs,)](
        in_ptr=in_5,
        out_ptr=out,
        N=N,
        H=236,
        W=32,
        SLICE_START=1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def _interp_chain_2_impl(in_6):
    N = 4 * 1 * 225 * 32
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty(4, 1, 225, 32, dtype=in_6.dtype, device=in_6.device)
    
    interp_chain_2_kernel[(num_programs,)](
        in_ptr=in_6,
        out_ptr=out,
        N=N,
        H=236,
        W=32,
        SLICE_START=1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


@torch.fx.wrap
def shared_dispatch_wrapper(*args):
    route = args[-1]
    if route == "route_dropout_identity":
        return _dropout_identity_impl(args[0])
    elif route == "route_interp_chain_1":
        return _interp_chain_1_impl(args[0])
    elif route == "route_interp_chain_2":
        return _interp_chain_2_impl(args[0])
    else:
        raise RuntimeError(f"Unknown route: {route}")


def replacement_func():
    return shared_dispatch_wrapper