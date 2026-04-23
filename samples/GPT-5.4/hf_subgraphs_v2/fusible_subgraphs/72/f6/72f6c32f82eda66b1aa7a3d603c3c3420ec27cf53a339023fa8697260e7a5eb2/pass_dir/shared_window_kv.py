import torch
import triton
import triton.language as tl


@triton.jit
def _window_kv_partition_kernel(
    x_ptr,
    q_ptr,
    v_ptr,
    BLOCK_P: tl.constexpr,
    C_GROUP: tl.constexpr,
    VCH: tl.constexpr,
):
    pid_p = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_g = tl.program_id(2)

    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    mask_p = offs_p < 144

    win_h = pid_w // 2
    win_w = pid_w % 2

    kh = offs_p // 12
    kw = offs_p % 12

    y_pad = win_h * 8 + kh
    x_pad = win_w * 8 + kw

    valid = mask_p & (y_pad >= 2) & (y_pad < 18) & (x_pad >= 2) & (x_pad < 18)
    spatial = (y_pad - 2) * 16 + (x_pad - 2)
    spatial = tl.where(valid, spatial, 0)

    base_c = pid_g * C_GROUP

    q_ch = tl.arange(0, 16)[:, None]
    q_vals = tl.load(
        x_ptr + (base_c + q_ch) * 256 + spatial[None, :],
        mask=valid[None, :],
        other=0.0,
    )
    q_base = ((pid_g * 4 + pid_w) * 16) * 144
    tl.store(
        q_ptr + q_base + q_ch * 144 + offs_p[None, :],
        q_vals,
        mask=mask_p[None, :],
    )

    v_ch = tl.arange(0, VCH)[None, :]
    v_vals = tl.load(
        x_ptr + (base_c + 16 + v_ch) * 256 + spatial[:, None],
        mask=valid[:, None],
        other=0.0,
    )
    v_base = ((pid_g * 4 + pid_w) * 144) * VCH
    tl.store(
        v_ptr + v_base + offs_p[:, None] * VCH + v_ch,
        v_vals,
        mask=mask_p[:, None],
    )


@torch.fx.wrap
def fused_window_kv_partition(x, route):
    if route == 'kv48':
        c_group = 48
        vch = 32
        num_warps = 4
    elif route == 'kv80':
        c_group = 80
        vch = 64
        num_warps = 4
    else:
        raise RuntimeError(f'Unsupported route: {route}')

    q = torch.empty((8, 4, 16, 144), device=x.device, dtype=x.dtype)
    v = torch.empty((8, 4, 144, vch), device=x.device, dtype=x.dtype)

    block_p = 64
    grid = (triton.cdiv(144, block_p), 4, 8)
    _window_kv_partition_kernel[grid](
        x,
        q,
        v,
        BLOCK_P=block_p,
        C_GROUP=c_group,
        VCH=vch,
        num_warps=num_warps,
        num_stages=1,
    )
    return (q, v)