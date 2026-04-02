import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """
    Match: einsum('bchw,bchj->bhwj', in_2, in_1) + cat + softmax + slice.
    in_2 (query):  (B, C, H, W)
    in_1 (key):    (B, C, H, J)
    in_0 (energy): (B, H, W, J)
    Returns tmp_3=(B,H,W,128), tmp_4=(B,H,W,64).
    """
    einsum = torch.functional.einsum('bchw,bchj->bhwj', in_2, in_1)
    tmp_2 = torch.cat([in_0, einsum], dim=-1)
    tmp_3 = torch.nn.functional.softmax(tmp_2, dim=-1)
    tmp_4 = tmp_3[(Ellipsis, slice(None, 64, None))]
    return (tmp_3, tmp_4)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4,  num_stages=2),
        triton.Config({}, num_warps=8,  num_stages=2),
        triton.Config({}, num_warps=16, num_stages=2),
        triton.Config({}, num_warps=4,  num_stages=3),
        triton.Config({}, num_warps=8,  num_stages=3),
    ],
    key=['B', 'H'],
)
@triton.jit
def fused_matmul_cat_softmax_kernel(
    # query in_2: (B, C, H, W)
    in2_ptr, stride_in2_b, stride_in2_c, stride_in2_h, stride_in2_w,
    # key   in_1: (B, C, H, J)
    in1_ptr, stride_in1_b, stride_in1_c, stride_in1_h, stride_in1_j,
    # energy in_0: (B, H, W, J)  – last dim stride is always 1 (contiguous)
    in0_ptr, stride_in0_b, stride_in0_h, stride_in0_w,
    # output: (B, H, W, 2*J)    – last dim stride is always 1 (contiguous)
    out_ptr, stride_out_b, stride_out_h, stride_out_w,
    B, H, W, C, J,
    BLOCK_W: tl.constexpr,   # == W == 64
    BLOCK_C: tl.constexpr,   # == C == 64
    BLOCK_J: tl.constexpr,   # == J == 64
):
    """
    One program per (b, h) pair.

    For each (b, h):
      A[w, c] = in_2[b, c, h, w]          shape (W, C)
      B[c, j] = in_1[b, c, h, j]          shape (C, J)
      R        = tl.dot(A, B)              shape (W, J)  [fp32 accum]
      E[w, j]  = in_0[b, h, w, j]         shape (W, J)

    Then row-wise softmax over [E, R] (128 elements) and write
      out[b, h, w, 0:J]   = softmax_E
      out[b, h, w, J:2J]  = softmax_R
    """
    bh_idx = tl.program_id(0)
    b_idx  = bh_idx // H
    h_idx  = bh_idx %  H

    w_offs = tl.arange(0, BLOCK_W)   # (W,)
    c_offs = tl.arange(0, BLOCK_C)   # (C,)
    j_offs = tl.arange(0, BLOCK_J)   # (J,)

    # ---- Load A = in_2[b, :, h, :].T  ->  (W, C) ----
    # A[w, c] = in_2[b, c, h, w]
    in2_base = b_idx * stride_in2_b + h_idx * stride_in2_h
    a_ptrs = (in2_ptr + in2_base
              + c_offs[None, :] * stride_in2_c
              + w_offs[:, None] * stride_in2_w)
    A = tl.load(a_ptrs)   # (W, C)  in original dtype

    # ---- Load B = in_1[b, :, h, :]    ->  (C, J) ----
    # B[c, j] = in_1[b, c, h, j]
    in1_base = b_idx * stride_in1_b + h_idx * stride_in1_h
    b_ptrs = (in1_ptr + in1_base
              + c_offs[:, None] * stride_in1_c
              + j_offs[None, :] * stride_in1_j)
    B_mat = tl.load(b_ptrs)   # (C, J)  in original dtype

    # ---- Matmul: R = A @ B  ->  (W, J),  fp32 accumulator ----
    R = tl.dot(A, B_mat, out_dtype=tl.float32)   # (W, J)

    # ---- Load E = in_0[b, h, :, :]   ->  (W, J) ----
    # E[w, j] = in_0[b, h, w, j]   (last dim stride = 1 since contiguous)
    in0_base = b_idx * stride_in0_b + h_idx * stride_in0_h
    e_ptrs = (in0_ptr + in0_base
              + w_offs[:, None] * stride_in0_w
              + j_offs[None, :])
    E = tl.load(e_ptrs).to(tl.float32)   # (W, J)

    # ---- Numerically stable softmax over [E ‖ R]  (128 elems / row) ----
    max_val = tl.maximum(tl.max(E, axis=1), tl.max(R, axis=1))  # (W,)

    exp_E = tl.exp(E - max_val[:, None])   # (W, J)
    exp_R = tl.exp(R - max_val[:, None])   # (W, J)

    inv_sum = 1.0 / (tl.sum(exp_E, axis=1) + tl.sum(exp_R, axis=1))  # (W,)

    out_E = exp_E * inv_sum[:, None]   # (W, J)
    out_R = exp_R * inv_sum[:, None]   # (W, J)

    # ---- Store ----
    # Triton auto-casts fp32 values to the pointer's element type (fp16/bf16/fp32)
    out_base = b_idx * stride_out_b + h_idx * stride_out_h

    out_e_ptrs = (out_ptr + out_base
                  + w_offs[:, None] * stride_out_w
                  + j_offs[None, :])
    tl.store(out_e_ptrs, out_E)

    out_r_ptrs = (out_ptr + out_base
                  + w_offs[:, None] * stride_out_w
                  + (BLOCK_J + j_offs[None, :]))
    tl.store(out_r_ptrs, out_R)


@torch.fx.wrap
def fused_einsum_cat_softmax_slice(in_0, in_1, in_2):
    """
    Fully-fused replacement for:
      einsum('bchw,bchj->bhwj', in_2, in_1)
      -> cat([in_0, ...], dim=-1)
      -> softmax(dim=-1)
      -> slice [..., :64]
    All computation is done in a single Triton kernel per (b, h) pair.
    """
    B, C, H, W = in_2.shape
    J = in_1.shape[3]   # 64

    out = torch.empty(B, H, W, J + J, dtype=in_0.dtype, device=in_0.device)

    # Make inputs contiguous so strides are predictable
    in_2_c = in_2.contiguous()
    in_1_c = in_1.contiguous()
    in_0_c = in_0.contiguous()

    fused_matmul_cat_softmax_kernel[(B * H,)](
        in_2_c,
        in_2_c.stride(0), in_2_c.stride(1), in_2_c.stride(2), in_2_c.stride(3),
        in_1_c,
        in_1_c.stride(0), in_1_c.stride(1), in_1_c.stride(2), in_1_c.stride(3),
        in_0_c,
        in_0_c.stride(0), in_0_c.stride(1), in_0_c.stride(2),
        out,
        out.stride(0), out.stride(1), out.stride(2),
        B, H, W, C, J,
        BLOCK_W=W, BLOCK_C=C, BLOCK_J=J,
    )

    tmp_4 = out[..., :J]
    return (out, tmp_4)


def replacement_func():
    return fused_einsum_cat_softmax_slice