import torch
import triton
import triton.language as tl
from graph_net_bench.torch.posion_dispatch_tensor import unwrap_tensor


@triton.jit
def _rope_q_cat_kernel(
    q_ptr,
    cos_ptr,
    sin_ptr,
    cls_ptr,
    out_ptr,
    H,
    N,
    q_s0,
    q_s1,
    q_s2,
    q_s3,
    cos_s0,
    cos_s1,
    sin_s0,
    sin_s1,
    cls_s0,
    cls_s1,
    cls_s2,
    cls_s3,
    out_s0,
    out_s1,
    out_s2,
    out_s3,
):
    row = tl.program_id(0)
    pairs = tl.arange(0, 32)

    rows_per_b = H * (N + 1)
    b = row // rows_per_b
    rem = row % rows_per_b
    h = rem // (N + 1)
    to = rem % (N + 1)

    is_cls = to == 0
    tok = tl.maximum(to - 1, 0)

    out_base = out_ptr + b * out_s0 + h * out_s1 + to * out_s2
    out_even_ptrs = out_base + (2 * pairs) * out_s3
    out_odd_ptrs = out_base + (2 * pairs + 1) * out_s3

    cls_base = cls_ptr + b * cls_s0 + h * cls_s1
    cls_even = tl.load(cls_base + (2 * pairs) * cls_s3)
    cls_odd = tl.load(cls_base + (2 * pairs + 1) * cls_s3)

    q_base = q_ptr + b * q_s0 + h * q_s1 + tok * q_s2
    q_even = tl.load(q_base + (2 * pairs) * q_s3)
    q_odd = tl.load(q_base + (2 * pairs + 1) * q_s3)

    cos_base = cos_ptr + tok * cos_s0
    sin_base = sin_ptr + tok * sin_s0
    cos_even = tl.load(cos_base + (2 * pairs) * cos_s1)
    cos_odd = tl.load(cos_base + (2 * pairs + 1) * cos_s1)
    sin_even = tl.load(sin_base + (2 * pairs) * sin_s1)
    sin_odd = tl.load(sin_base + (2 * pairs + 1) * sin_s1)

    rot_even = q_even * cos_even + (-q_odd) * sin_even
    rot_odd = q_odd * cos_odd + q_even * sin_odd

    out_even = tl.where(is_cls, cls_even, rot_even)
    out_odd = tl.where(is_cls, cls_odd, rot_odd)

    tl.store(out_even_ptrs, out_even)
    tl.store(out_odd_ptrs, out_odd)


@triton.jit
def _rope_k_cat_kernel(
    pos_ptr,
    k_ptr,
    out_ptr,
    H,
    N,
    pos_s0,
    pos_s1,
    k_s0,
    k_s1,
    k_s2,
    k_s3,
    out_s0,
    out_s1,
    out_s2,
    out_s3,
):
    row = tl.program_id(0)
    pairs = tl.arange(0, 32)

    rows_per_b = H * (N + 1)
    b = row // rows_per_b
    rem = row % rows_per_b
    h = rem // (N + 1)
    to = rem % (N + 1)

    is_cls = to == 0
    pos_tok = tl.maximum(to - 1, 0)
    k_tok = tl.maximum(to, 1)

    out_base = out_ptr + b * out_s0 + h * out_s1 + to * out_s2
    out_even_ptrs = out_base + (2 * pairs) * out_s3
    out_odd_ptrs = out_base + (2 * pairs + 1) * out_s3

    cls_base = k_ptr + b * k_s0 + h * k_s1
    cls_even = tl.load(cls_base + (2 * pairs) * k_s3)
    cls_odd = tl.load(cls_base + (2 * pairs + 1) * k_s3)

    k_base = k_ptr + b * k_s0 + h * k_s1 + k_tok * k_s2
    k_even = tl.load(k_base + (2 * pairs) * k_s3)
    k_odd = tl.load(k_base + (2 * pairs + 1) * k_s3)

    pos_base = pos_ptr + pos_tok * pos_s0
    pos1_even = tl.load(pos_base + (2 * pairs) * pos_s1)
    pos1_odd = tl.load(pos_base + (2 * pairs + 1) * pos_s1)
    pos2_even = tl.load(pos_base + (64 + 2 * pairs) * pos_s1)
    pos2_odd = tl.load(pos_base + (64 + 2 * pairs + 1) * pos_s1)

    rot_even = k_even * pos2_even + (-k_odd) * pos1_even
    rot_odd = k_odd * pos2_odd + k_even * pos1_odd

    out_even = tl.where(is_cls, cls_even, rot_even)
    out_odd = tl.where(is_cls, cls_odd, rot_odd)

    tl.store(out_even_ptrs, out_even)
    tl.store(out_odd_ptrs, out_odd)


@torch.fx.wrap
def fused_rope_cat_leaf(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    try:
        in_0 = unwrap_tensor(in_0)
        in_1 = unwrap_tensor(in_1)
        in_2 = unwrap_tensor(in_2)
        in_3 = unwrap_tensor(in_3)
        in_4 = unwrap_tensor(in_4)
        in_5 = unwrap_tensor(in_5)
        in_6 = unwrap_tensor(in_6)

        B = in_3.shape[0]
        H = in_3.shape[1]
        N = in_3.shape[2]
        D = in_3.shape[3]

        out_shape = (B, H, N + 1, D)
        out_k = torch.empty(out_shape, device=in_6.device, dtype=in_6.dtype)
        out_q = torch.empty(out_shape, device=in_6.device, dtype=in_6.dtype)

        total_rows = B * H * (N + 1)
        grid = (total_rows,)

        _rope_q_cat_kernel[grid](
            in_3,
            in_1,
            in_5,
            in_2,
            out_q,
            H,
            N,
            in_3.stride(0),
            in_3.stride(1),
            in_3.stride(2),
            in_3.stride(3),
            in_1.stride(0),
            in_1.stride(1),
            in_5.stride(0),
            in_5.stride(1),
            in_2.stride(0),
            in_2.stride(1),
            in_2.stride(2),
            in_2.stride(3),
            out_q.stride(0),
            out_q.stride(1),
            out_q.stride(2),
            out_q.stride(3),
            num_warps=1,
            num_stages=1,
        )

        _rope_k_cat_kernel[grid](
            in_0,
            in_4,
            out_k,
            H,
            N,
            in_0.stride(0),
            in_0.stride(1),
            in_4.stride(0),
            in_4.stride(1),
            in_4.stride(2),
            in_4.stride(3),
            out_k.stride(0),
            out_k.stride(1),
            out_k.stride(2),
            out_k.stride(3),
            num_warps=1,
            num_stages=1,
        )

        return out_k, out_q
    except Exception as e:
        raise RuntimeError(f"fused_rope_cat_leaf_failed: {type(e).__name__}: {e}")


def fused_rope_cat(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    outs = fused_rope_cat_leaf(in_0, in_1, in_2, in_3, in_4, in_5, in_6)
    return outs[0], outs[1]


def replacement_func():
    return fused_rope_cat