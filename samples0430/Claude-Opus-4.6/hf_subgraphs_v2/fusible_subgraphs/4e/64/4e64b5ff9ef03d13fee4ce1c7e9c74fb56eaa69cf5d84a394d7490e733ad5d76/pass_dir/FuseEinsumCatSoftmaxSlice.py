import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    einsum = torch.functional.einsum('bchw,bchj->bhwj', in_2, in_1)
    tmp_2 = torch.cat([in_0, einsum], dim=-1)
    tmp_3 = torch.nn.functional.softmax(tmp_2, dim=-1)
    tmp_4 = tmp_3[(Ellipsis, slice(None, 64, None))]
    return (tmp_3, tmp_4)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 16, 'BLOCK_C': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 32, 'BLOCK_C': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 64, 'BLOCK_C': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 32, 'BLOCK_C': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_W': 16, 'BLOCK_C': 32}, num_warps=4, num_stages=3),
    ],
    key=['B'],
)
@triton.jit
def fused_einsum_cat_softmax_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, out_ptr,
    B,
    stride_in0_b, stride_in0_h, stride_in0_w, stride_in0_last,
    stride_in1_b, stride_in1_c, stride_in1_h, stride_in1_j,
    stride_in2_b, stride_in2_c, stride_in2_h, stride_in2_w,
    stride_out_b, stride_out_h, stride_out_w, stride_out_last,
    H: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    J: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)

    num_w_blocks = W // BLOCK_W
    bh = pid // num_w_blocks
    w_block = pid % num_w_blocks
    b = bh // H
    h = bh % H
    w_start = w_block * BLOCK_W

    w_offs = w_start + tl.arange(0, BLOCK_W)
    j_offs = tl.arange(0, J)

    # Accumulator for einsum result: [BLOCK_W, J]
    acc = tl.zeros([BLOCK_W, J], dtype=tl.float32)

    # Compute einsum: result[w, j] = sum_c query[b, c, h, w] * key[b, c, h, j]
    for c_start in range(0, C, BLOCK_C):
        c_offs = c_start + tl.arange(0, BLOCK_C)

        # Load query[b, c, h, w] as [BLOCK_W, BLOCK_C]
        q_ptrs = (in_2_ptr + b * stride_in2_b + h * stride_in2_h +
                  w_offs[:, None] * stride_in2_w + c_offs[None, :] * stride_in2_c)
        q = tl.load(q_ptrs)  # [BLOCK_W, BLOCK_C]

        # Load key[b, c, h, j] as [BLOCK_C, J]
        k_ptrs = (in_1_ptr + b * stride_in1_b + h * stride_in1_h +
                  c_offs[:, None] * stride_in1_c + j_offs[None, :] * stride_in1_j)
        k = tl.load(k_ptrs)  # [BLOCK_C, J]

        # Matmul: [BLOCK_W, BLOCK_C] @ [BLOCK_C, J] -> [BLOCK_W, J]
        acc += tl.dot(q, k)

    # Load in_0[b, h, w, :J]: [BLOCK_W, J]
    in0_ptrs = (in_0_ptr + b * stride_in0_b + h * stride_in0_h +
                w_offs[:, None] * stride_in0_w + j_offs[None, :] * stride_in0_last)
    in0 = tl.load(in0_ptrs).to(tl.float32)  # [BLOCK_W, J]

    # Softmax over concatenated [in0, acc] (2*J = 128 elements per row)
    max_in0 = tl.max(in0, axis=1)  # [BLOCK_W]
    max_acc = tl.max(acc, axis=1)  # [BLOCK_W]
    max_val = tl.maximum(max_in0, max_acc)  # [BLOCK_W]

    exp_in0 = tl.exp(in0 - max_val[:, None])  # [BLOCK_W, J]
    exp_acc = tl.exp(acc - max_val[:, None])  # [BLOCK_W, J]

    sum_exp = tl.sum(exp_in0, axis=1) + tl.sum(exp_acc, axis=1)  # [BLOCK_W]

    softmax_in0 = exp_in0 / sum_exp[:, None]  # [BLOCK_W, J]
    softmax_acc = exp_acc / sum_exp[:, None]  # [BLOCK_W, J]

    # Store first half (from in_0)
    out_ptrs_first = (out_ptr + b * stride_out_b + h * stride_out_h +
                      w_offs[:, None] * stride_out_w + j_offs[None, :] * stride_out_last)
    tl.store(out_ptrs_first, softmax_in0.to(out_ptr.dtype.element_ty))

    # Store second half (from einsum)
    out_ptrs_second = (out_ptr + b * stride_out_b + h * stride_out_h +
                       w_offs[:, None] * stride_out_w + (j_offs[None, :] + J) * stride_out_last)
    tl.store(out_ptrs_second, softmax_acc.to(out_ptr.dtype.element_ty))


@torch.fx.wrap
def fused_einsum_cat_softmax(in_0, in_1, in_2):
    B, C, H, W = in_2.shape
    J = in_1.shape[3]

    # Output: [B, H, W, 2*J]
    out = torch.empty(B, H, W, 2 * J, dtype=in_0.dtype, device=in_0.device)

    grid = lambda META: (B * H * (W // META['BLOCK_W']),)

    fused_einsum_cat_softmax_kernel[grid](
        in_0, in_1, in_2, out,
        B,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        H=H, W=W, C=C, J=J,
    )

    tmp_4 = out[(Ellipsis, slice(None, J, None))]
    return (out, tmp_4)


def replacement_func():
    return fused_einsum_cat_softmax