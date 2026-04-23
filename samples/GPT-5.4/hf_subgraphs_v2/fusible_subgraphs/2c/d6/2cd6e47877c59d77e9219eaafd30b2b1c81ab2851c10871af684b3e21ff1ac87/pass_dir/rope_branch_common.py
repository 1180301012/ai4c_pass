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
def rope_q_branch(cos, cls, q, sin, ref):
    cos = unwrap_tensor(cos)
    cls = unwrap_tensor(cls)
    q = unwrap_tensor(q)
    sin = unwrap_tensor(sin)
    ref = unwrap_tensor(ref)

    B = q.shape[0]
    H = q.shape[1]
    N = q.shape[2]
    D = q.shape[3]
    out = torch.empty((B, H, N + 1, D), device=ref.device, dtype=ref.dtype)

    _rope_q_cat_kernel[(B * H * (N + 1),)](
        q,
        cos,
        sin,
        cls,
        out,
        H,
        N,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        cls.stride(0),
        cls.stride(1),
        cls.stride(2),
        cls.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        num_warps=1,
        num_stages=1,
    )
    return out


@torch.fx.wrap
def rope_k_branch(pos, k, ref):
    pos = unwrap_tensor(pos)
    k = unwrap_tensor(k)
    ref = unwrap_tensor(ref)

    B = k.shape[0]
    H = k.shape[1]
    N = k.shape[2] - 1
    D = k.shape[3]
    out = torch.empty((B, H, N + 1, D), device=ref.device, dtype=ref.dtype)

    _rope_k_cat_kernel[(B * H * (N + 1),)](
        pos,
        k,
        out,
        H,
        N,
        pos.stride(0),
        pos.stride(1),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        num_warps=1,
        num_stages=1,
    )
    return out


def rope_branch_dispatch(*args):
    route = args[-1]
    if route == "q":
        return rope_q_branch(args[0], args[1], args[2], args[3], args[4])
    if route == "k":
        return rope_k_branch(args[0], args[1], args[2])
    raise RuntimeError(f"unknown_route: {route}")


def replacement_func():
    return rope_branch_dispatch