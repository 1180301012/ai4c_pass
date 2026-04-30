import torch
import triton
import triton.language as tl


def pattern(x):
    tmp_6 = torch.nn.functional.dropout(x, 0.0, False, False)
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    return tmp_7


def replacement_args(x):
    return (x, "eliminate_double_dropout")


@torch.fx.wrap
def _identity(x):
    return x


@torch.fx.wrap
def _fused_add_mean(in_4, in_5):
    B, C, H, W = in_4.shape
    HW = H * W
    total_bc = B * C

    BLOCK_HW = triton.next_power_of_2(HW)
    if BLOCK_HW > 512:
        BLOCK_HW = 512

    out = torch.empty((B, C), dtype=in_4.dtype, device=in_4.device)

    grid = (total_bc,)

    fused_add_mean_kernel[grid](
        in_4_ptr=in_4, in_5_ptr=in_5,
        out_ptr=out,
        total_bc=total_bc, HW=HW,
        BLOCK_HW=BLOCK_HW,
    )

    return out


@triton.jit
def fused_add_mean_kernel(
    in_4_ptr, in_5_ptr,
    out_ptr,
    total_bc, HW,
    BLOCK_HW: tl.constexpr,
):
    bc_idx = tl.program_id(0)

    if bc_idx >= total_bc:
        return

    base_offset = bc_idx * HW

    acc = 0.0

    for start in range(0, HW, BLOCK_HW):
        offsets = start + tl.arange(0, BLOCK_HW)
        mask = offsets < HW
        val_4 = tl.load(in_4_ptr + base_offset + offsets, mask=mask, other=0.0).to(tl.float32)
        val_5 = tl.load(in_5_ptr + base_offset + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum((val_4 + val_5) * mask.to(tl.float32))

    mean_val = acc / HW

    tl.store(out_ptr + bc_idx, mean_val)


@torch.fx.wrap
def dispatch_wrapper(*args):
    route = args[-1]
    if route == "eliminate_double_dropout":
        return _identity(args[0])
    elif route == "fuse_add_mean":
        return _fused_add_mean(args[0], args[1])
    else:
        raise ValueError(f"Unknown route: {route}")


def replacement_func():
    return dispatch_wrapper