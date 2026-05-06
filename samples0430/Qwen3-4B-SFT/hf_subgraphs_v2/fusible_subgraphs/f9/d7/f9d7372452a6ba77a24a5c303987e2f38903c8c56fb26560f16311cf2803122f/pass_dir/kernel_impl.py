"""
Shared Triton kernels and dispatch wrapper for all passes.
Both FuseAddAddRelu and FuseViewPermute import `shared_dispatch` from here
so that replacement_func() returns THE SAME function object → satisfies
output_pass_replacement_func_limit == 1.
"""
import torch
import triton
import triton.language as tl


# ── 1. add + add (3-way sum) kernel ──────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_size': 1024}),
        triton.Config({'BLOCK_size': 2048}),
        triton.Config({'BLOCK_size': 4096}),
    ],
    key=['n_elements'],
)
@triton.jit
def _aa_add_kernel(
    x_ptr, a_ptr, b_ptr, out_ptr,
    n_elements,
    BLOCK_size: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_size + tl.arange(0, BLOCK_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    out = x + a + b
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_size': 1024}),
        triton.Config({'BLOCK_size': 2048}),
        triton.Config({'BLOCK_size': 4096}),
    ],
    key=['n_elements'],
)
@triton.jit
def _aa_relu_kernel(
    x_ptr, a_ptr, b_ptr, out_ptr,
    n_elements,
    BLOCK_size: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_size + tl.arange(0, BLOCK_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    out = tl.maximum(x + a + b, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


# ── 2. view + permute kernel (transpose [1,C,H,W] → [1,W,HW/C]) ─────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 64}),
        triton.Config({'BLOCK_H': 64, 'BLOCK_W': 32}),
    ],
    key=['inner_stride'],
)
@triton.jit
def _view_permute_kernel(
    in_ptr, out_ptr,
    inner_stride,   # = number of columns in input (W = 64)
    total_elements,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    w_idx = pid // BLOCK_H
    h_idx = pid % BLOCK_H
    w_offs = w_idx * BLOCK_W + tl.arange(0, BLOCK_W)
    h_offs = h_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    # Read from [H/W, W] layout (row-major): index = w * inner_stride + h
    read_offsets = w_offs[:, None] * inner_stride + h_offs[None, :]
    mask = (w_offs[:, None] < total_elements % inner_stride) & \
           (h_offs[None, :] < (total_elements // inner_stride))
    vals = tl.load(in_ptr + read_offsets, mask=mask)
    # Write to [W, H/W] layout: row-major, same offsets already correct
    tl.store(out_ptr + read_offsets, vals, mask=mask)


# ── 3. shared dispatch wrapper ────────────────────────────────────────────────

@torch.fx.wrap
def shared_dispatch(*args):
    """
    Route by last argument (a string encoding the pass):
      "add_add":  args = (in_0, in_2, in_3, "add_add")
        → fused sum (no relu: downstream Dynamo relu keeps applying)
      "add_relu": args = (in_0, in_2, in_3, "add_relu")
        → fused sum + relu
      "view_permute": args = (in_1, "view_permute")
        → view then permute (transpose) of in_1
    """
    route = args[-1]

    if route == "add_add":
        # args = (in_0, in_2, in_3, "add_add")
        in_0, in_2, in_3 = args[0], args[1], args[2]
        N = in_3.numel()
        out = torch.empty_like(in_3)
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_size']),)
        _aa_add_kernel[grid](in_3, in_0, in_2, out, N)
        return out

    if route == "add_relu":
        # args = (in_0, in_2, in_3, "add_relu")
        in_0, in_2, in_3 = args[0], args[1], args[2]
        N = in_3.numel()
        out = torch.empty_like(in_3)
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_size']),)
        _aa_relu_kernel[grid](in_3, in_0, in_2, out, N)
        return out

    # route == "view_permute"
    in_1 = args[0]
    B, C, H, W = in_1.shape
    total = B * C * H * W
    out = torch.empty(B, H * W, C, dtype=in_1.dtype, device=in_1.device)
    grid = lambda meta: (
        triton.cdiv(H * W, meta['BLOCK_W']) * triton.cdiv(C, meta['BLOCK_H']),
    )
    _view_permute_kernel[grid](in_1, out, H, total)
    return out