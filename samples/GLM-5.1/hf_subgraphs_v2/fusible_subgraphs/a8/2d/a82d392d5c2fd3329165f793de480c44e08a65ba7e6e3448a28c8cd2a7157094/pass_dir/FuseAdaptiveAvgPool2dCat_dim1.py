import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.adaptive_avg_pool2d(in_0, (32, 24))
    tmp_1 = torch.cat([tmp_0, in_1], dim=1)
    return (tmp_1,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.autotune(configs=[
    triton.Config({'BLOCK': 512, 'num_warps': 4, 'num_stages': 2}),
    triton.Config({'BLOCK': 1024, 'num_warps': 4, 'num_stages': 3}),
    triton.Config({'BLOCK': 2048, 'num_warps': 4, 'num_stages': 3}),
    triton.Config({'BLOCK': 4096, 'num_warps': 8, 'num_stages': 3}),
], key=['total_elements'])
@triton.jit
def fused_pool_cat_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    C0, C1, H_out, W_out,
    H_in, W_in,
    total_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    mask = offsets < total_elements

    # Decode flat index into (b, c_out, h_out, w_out) for output
    # All tensors are contiguous, so we can compute offsets directly
    C_total = C0 + C1
    HW_out = H_out * W_out
    CHW_out = C_total * HW_out

    w_out = offsets % W_out
    remainder = offsets // W_out
    h_out = remainder % H_out
    remainder = remainder // H_out
    c_out = remainder % C_total
    b = remainder // C_total

    # Determine which path each element takes
    is_pool = c_out < C0
    is_copy = c_out >= C0

    # Output offset - contiguous, so offset = flat index
    # out[b, c, h, w] = out_ptr + b * CHW_out + c * HW_out + h * W_out + w
    # But since it's contiguous, offset = b*CHW_out + c*HW_out + h*W_out + w = offsets
    out_offset = offsets

    # Pooling path: average 2x2 block from in_0 for channels < C0
    # in_0 is contiguous [B, C0, H_in, W_in]
    # offset = b * C0 * H_in * W_in + c * H_in * W_in + h_in * W_in + w_in
    HW_in = H_in * W_in
    pool_mask = mask & is_pool
    val = tl.zeros([BLOCK], dtype=tl.float32)
    for kh in tl.static_range(2):
        for kw in tl.static_range(2):
            h_in = h_out * 2 + kh
            w_in = w_out * 2 + kw
            h_in_ok = h_in < H_in
            w_in_ok = w_in < W_in
            load_mask = pool_mask & h_in_ok & w_in_ok
            in0_offset = b * C0 * HW_in + c_out * HW_in + h_in * W_in + w_in
            in_val = tl.load(in_0_ptr + in0_offset, mask=load_mask, other=0.0)
            val += in_val
    avg_val = val * 0.25
    avg_val = tl.where(is_pool, avg_val, 0.0)

    # Copy path: read from in_1 for channels >= C0
    # in_1 is contiguous [B, C1, H_out, W_out]
    # offset = b * C1 * H_out * W_out + c1 * H_out * W_out + h_out * W_out + w_out
    c1 = c_out - C0
    copy_mask = mask & is_copy
    in1_offset = b * C1 * HW_out + c1 * HW_out + h_out * W_out + w_out
    in1_val = tl.load(in_1_ptr + in1_offset, mask=copy_mask, other=0.0)

    # Select the appropriate value
    result = tl.where(is_pool, avg_val, in1_val)
    tl.store(out_ptr + out_offset, result, mask=mask)


@torch.fx.wrap
def fused_pool_cat(in_0, in_1):
    B = in_0.shape[0]
    C0 = in_0.shape[1]
    C1 = in_1.shape[1]
    H_in = in_0.shape[2]
    W_in = in_0.shape[3]
    H_out = in_1.shape[2]
    W_out = in_1.shape[3]

    C_total = C0 + C1
    out = torch.empty((B, C_total, H_out, W_out), dtype=in_0.dtype, device=in_0.device)

    total_elements = B * C_total * H_out * W_out
    grid = ((total_elements + 4096 - 1) // 4096,)

    fused_pool_cat_kernel[grid](
        in_0, in_1, out,
        C0, C1, H_out, W_out,
        H_in, W_in,
        total_elements,
    )

    return out

def replacement_func():
    return fused_pool_cat