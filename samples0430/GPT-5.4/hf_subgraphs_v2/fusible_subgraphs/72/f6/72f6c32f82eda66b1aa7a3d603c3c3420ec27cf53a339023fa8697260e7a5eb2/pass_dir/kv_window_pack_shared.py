import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_TOK": 16}, num_warps=2),
        triton.Config({"BLOCK_TOK": 32}, num_warps=4),
        triton.Config({"BLOCK_TOK": 64}, num_warps=4),
    ],
    key=["C_GROUP", "V_DIM"],
)
@triton.jit
def _fused_kv_from_conv_kernel(
    inp_ptr,
    out_k_ptr,
    out_v_ptr,
    stride_ic,
    stride_ih,
    stride_iw,
    stride_ok0,
    stride_ok1,
    stride_ok2,
    stride_ok3,
    stride_ov0,
    stride_ov1,
    stride_ov2,
    stride_ov3,
    C_GROUP,
    V_DIM,
    BLOCK_TOK: tl.constexpr,
    MAX_C: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_tok = tl.program_id(1)

    head = pid_hw // 4
    win = pid_hw % 4
    win_h = win // 2
    win_w = win % 2

    tok = pid_tok * BLOCK_TOK + tl.arange(0, BLOCK_TOK)
    tok_mask = tok < 144
    kh = tok // 12
    kw = tok % 12

    ih = win_h * 8 + kh - 2
    iw = win_w * 8 + kw - 2
    hw_mask = tok_mask & (ih >= 0) & (ih < 16) & (iw >= 0) & (iw < 16)

    c = tl.arange(0, MAX_C)
    chan_mask = c < C_GROUP
    global_c = head * C_GROUP + c

    in_ptrs = (
        inp_ptr
        + global_c[None, :] * stride_ic
        + ih[:, None] * stride_ih
        + iw[:, None] * stride_iw
    )
    vals = tl.load(in_ptrs, mask=hw_mask[:, None] & chan_mask[None, :], other=0.0)

    k = tl.arange(0, 16)
    out_k_ptrs = (
        out_k_ptr
        + head * stride_ok0
        + win * stride_ok1
        + k[None, :] * stride_ok2
        + tok[:, None] * stride_ok3
    )
    tl.store(out_k_ptrs, vals[:, :16], mask=tok_mask[:, None])

    v = tl.arange(0, 64)
    out_v_ptrs = (
        out_v_ptr
        + head * stride_ov0
        + win * stride_ov1
        + tok[:, None] * stride_ov2
        + v[None, :] * stride_ov3
    )
    tl.store(out_v_ptrs, vals[:, 16:80], mask=tok_mask[:, None] & (v[None, :] < V_DIM))


@torch.fx.wrap
def fused_kv_from_conv(conv2d_out, route):
    if route == "shape48":
        c_group = 48
        v_dim = 32
    elif route == "shape80":
        c_group = 80
        v_dim = 64
    else:
        raise ValueError("unknown route")

    out_k = torch.empty((8, 4, 16, 144), device=conv2d_out.device, dtype=conv2d_out.dtype)
    out_v = torch.empty((8, 4, 144, v_dim), device=conv2d_out.device, dtype=conv2d_out.dtype)

    grid = lambda META: (32, triton.cdiv(144, META["BLOCK_TOK"]))
    _fused_kv_from_conv_kernel[grid](
        conv2d_out,
        out_k,
        out_v,
        conv2d_out.stride(1),
        conv2d_out.stride(2),
        conv2d_out.stride(3),
        out_k.stride(0),
        out_k.stride(1),
        out_k.stride(2),
        out_k.stride(3),
        out_v.stride(0),
        out_v.stride(1),
        out_v.stride(2),
        out_v.stride(3),
        c_group,
        v_dim,
        MAX_C=80,
    )
    return out_k, out_v