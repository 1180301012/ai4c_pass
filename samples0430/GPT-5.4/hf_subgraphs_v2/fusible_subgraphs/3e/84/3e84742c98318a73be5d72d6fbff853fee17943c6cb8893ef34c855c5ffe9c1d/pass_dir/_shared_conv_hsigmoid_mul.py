import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    ],
    key=["M", "CIN", "COUT"],
)
@triton.jit
def _conv1x1_hsigmoid_mul_kernel(
    xgate_ptr,
    w_ptr,
    b_ptr,
    xmul_ptr,
    out_ptr,
    M,
    CIN,
    COUT,
    W,
    HW,
    stride_xg_n,
    stride_xg_c,
    stride_w_co,
    stride_w_ci,
    stride_xm_n,
    stride_xm_c,
    stride_xm_h,
    stride_xm_w,
    stride_o_n,
    stride_o_c,
    stride_o_h,
    stride_o_w,
    ADD_CONST: tl.constexpr,
    DIV_CONST: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_co = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_co = pid_co * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    co_mask = offs_co < COUT

    hw_idx = offs_m % HW
    n_idx = offs_m // HW
    h_idx = hw_idx // W
    w_idx = hw_idx % W

    acc = tl.load(b_ptr + offs_co, mask=co_mask, other=0.0).to(tl.float32)
    acc = tl.broadcast_to(acc[None, :], [BLOCK_M, BLOCK_N])

    k = 0
    while k < CIN:
        offs_k = k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < CIN

        x = tl.load(
            xgate_ptr + n_idx[:, None] * stride_xg_n + offs_k[None, :] * stride_xg_c,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(
            w_ptr + offs_co[:, None] * stride_w_co + offs_k[None, :] * stride_w_ci,
            mask=co_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(x, tl.trans(w))
        k += BLOCK_K

    gate = (acc + ADD_CONST) / DIV_CONST
    gate = tl.maximum(0.0, tl.minimum(1.0, gate))

    mul = tl.load(
        xmul_ptr
        + n_idx[:, None] * stride_xm_n
        + offs_co[None, :] * stride_xm_c
        + h_idx[:, None] * stride_xm_h
        + w_idx[:, None] * stride_xm_w,
        mask=m_mask[:, None] & co_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    out = mul * gate

    tl.store(
        out_ptr
        + n_idx[:, None] * stride_o_n
        + offs_co[None, :] * stride_o_c
        + h_idx[:, None] * stride_o_h
        + w_idx[:, None] * stride_o_w,
        out,
        mask=m_mask[:, None] & co_mask[None, :],
    )


@torch.fx.wrap
def shared_dispatch(in_0, in_1, in_2, in_3, route: str):
    if route == "plus1_div2":
        add_const = 1.0
        div_const = 2.0
    elif route == "plus3_div6":
        add_const = 3.0
        div_const = 6.0
    else:
        raise RuntimeError(f"Unknown route: {route}")

    xmul_shape = in_2.shape
    xgate_shape = in_3.shape
    w_shape = in_1.shape

    N = xmul_shape[0]
    COUT = xmul_shape[1]
    H = xmul_shape[2]
    W = xmul_shape[3]
    M = N * H * W
    HW = H * W
    CIN = xgate_shape[1]

    xgate_stride = in_3.stride()
    w_stride = in_1.stride()
    xmul_stride = in_2.stride()

    out = torch.empty_like(in_2)
    out_stride = out.stride()

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(COUT, META["BLOCK_N"]),
    )

    _conv1x1_hsigmoid_mul_kernel[grid](
        in_3,
        in_1,
        in_0,
        in_2,
        out,
        M,
        CIN,
        COUT,
        W,
        HW,
        xgate_stride[0],
        xgate_stride[1],
        w_stride[0],
        w_stride[1],
        xmul_stride[0],
        xmul_stride[1],
        xmul_stride[2],
        xmul_stride[3],
        out_stride[0],
        out_stride[1],
        out_stride[2],
        out_stride[3],
        ADD_CONST=add_const,
        DIV_CONST=div_const,
    )
    return out