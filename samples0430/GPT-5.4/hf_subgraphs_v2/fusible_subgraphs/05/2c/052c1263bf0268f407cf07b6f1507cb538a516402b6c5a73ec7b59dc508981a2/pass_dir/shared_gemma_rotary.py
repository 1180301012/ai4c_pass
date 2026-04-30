import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rotary_key_kernel(
    key_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    key_stride_s,
    key_stride_d,
    cos_stride_s,
    cos_stride_d,
    sin_stride_s,
    sin_stride_d,
    out_stride_s,
    out_stride_d,
    BLOCK_D: tl.constexpr,
    HALF_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)

    key_ptrs = key_ptr + pid * key_stride_s + offs_d * key_stride_d
    cos_ptrs = cos_ptr + pid * cos_stride_s + offs_d * cos_stride_d
    sin_ptrs = sin_ptr + pid * sin_stride_s + offs_d * sin_stride_d

    rot_offs_d = tl.where(offs_d < HALF_D, offs_d + HALF_D, offs_d - HALF_D)
    rot_sign = tl.where(offs_d < HALF_D, -1.0, 1.0)
    rot_ptrs = key_ptr + pid * key_stride_s + rot_offs_d * key_stride_d

    key = tl.load(key_ptrs).to(tl.float32)
    cos = tl.load(cos_ptrs).to(tl.float32)
    sin = tl.load(sin_ptrs).to(tl.float32)
    rot = tl.load(rot_ptrs).to(tl.float32)

    out = key * cos + rot * sin * rot_sign

    out_ptrs = out_ptr + pid * out_stride_s + offs_d * out_stride_d
    tl.store(out_ptrs, out)


@torch.fx.wrap
def shared_dispatch(*args):
    route = args[-1]
    if route != "rotary_key":
        raise RuntimeError(f"Unknown route: {route}")

    in_1, in_2, in_4 = args[0], args[1], args[2]
    out = torch.empty_like(in_2)
    n_seq = in_2.shape[2]
    _fused_rotary_key_kernel[(n_seq,)](
        in_2,
        in_1,
        in_4,
        out,
        in_2.stride(2),
        in_2.stride(3),
        in_1.stride(2),
        in_1.stride(3),
        in_4.stride(2),
        in_4.stride(3),
        out.stride(2),
        out.stride(3),
        BLOCK_D=256,
        HALF_D=128,
        num_warps=4,
        num_stages=1,
    )
    return out