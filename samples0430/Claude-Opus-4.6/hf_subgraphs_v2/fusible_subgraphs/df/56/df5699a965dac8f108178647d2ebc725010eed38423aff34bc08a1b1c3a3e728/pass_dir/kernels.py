import torch
import triton
import triton.language as tl


@triton.jit
def channel_interleave_2out_kernel(
    A_ptr, B_ptr, out1_ptr, out2_ptr,
    N_half,      # elements per output tensor = B * C_in * HW
    C_in,        # input channels per tensor (20 for path1, 40 for path2)
    HW,          # H * W
    BLOCK_SIZE: tl.constexpr,
):
    """Writes interleaved channels directly into two output buffers (first half and second half)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_half

    # Map flat offset to (batch, out_ch_within_half, hw)
    hw_idx = offs % HW
    rem = offs // HW
    out_ch = rem % C_in  # channel within the half (0..C_in-1)
    batch = rem // C_in

    # In the full interleaved tensor [B, 2*C_in, H, W]:
    # First half (out1): channels 0..C_in-1 of the full tensor
    #   full_ch = out_ch, so in_ch = full_ch // 2, is_odd = full_ch % 2
    # Second half (out2): channels C_in..2*C_in-1 of the full tensor
    #   full_ch = out_ch + C_in, so in_ch = (out_ch+C_in)//2, is_odd = (out_ch+C_in)%2

    # For out1: full_ch = out_ch
    in_ch_1 = out_ch >> 1
    is_odd_1 = out_ch & 1
    src_offset_1 = batch * (C_in * HW) + in_ch_1 * HW + hw_idx
    a1 = tl.load(A_ptr + src_offset_1, mask=mask & (is_odd_1 == 0), other=0.0)
    b1 = tl.load(B_ptr + src_offset_1, mask=mask & (is_odd_1 == 1), other=0.0)
    val1 = a1 + b1
    tl.store(out1_ptr + offs, val1, mask=mask)

    # For out2: full_ch = out_ch + C_in
    full_ch_2 = out_ch + C_in
    in_ch_2 = full_ch_2 >> 1
    is_odd_2 = full_ch_2 & 1
    src_offset_2 = batch * (C_in * HW) + in_ch_2 * HW + hw_idx
    a2 = tl.load(A_ptr + src_offset_2, mask=mask & (is_odd_2 == 0), other=0.0)
    b2 = tl.load(B_ptr + src_offset_2, mask=mask & (is_odd_2 == 1), other=0.0)
    val2 = a2 + b2
    tl.store(out2_ptr + offs, val2, mask=mask)


@triton.jit
def linear_sigmoid_kernel(
    in_6_ptr,   # [B, 10, 1, 1] contiguous -> stride batch=10
    in_1_ptr,   # [40, 10, 1, 1] contiguous -> stride out_ch=10
    in_0_ptr,   # [40] bias
    out_ptr,    # [B, 40]
    B_dim,
    C_out,      # 40
    C_in,       # 10
    BLOCK_C: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C_out

    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for k in range(C_in):
        x_val = tl.load(in_6_ptr + pid_b * C_in + k)
        w_vals = tl.load(in_1_ptr + c_offs * C_in + k, mask=c_mask, other=0.0)
        acc += x_val.to(tl.float32) * w_vals.to(tl.float32)

    bias = tl.load(in_0_ptr + c_offs, mask=c_mask, other=0.0)
    acc += bias.to(tl.float32)

    result = tl.sigmoid(acc)

    tl.store(out_ptr + pid_b * C_out + c_offs, result.to(out_ptr.dtype.element_ty), mask=c_mask)


@triton.jit
def interleave_mul_2out_kernel(
    in_3_ptr,       # [B, C_in, H, W]
    in_5_ptr,       # [B, C_in, H, W]
    sigmoid_ptr,    # [B, C_in]
    out1_ptr,       # [B, C_in, H, W] first half
    out2_ptr,       # [B, C_in, H, W] second half
    N_half,         # B * C_in * HW
    C_in,           # 40
    HW,             # H * W = 768
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_half

    hw_idx = offs % HW
    rem = offs // HW
    out_ch = rem % C_in
    batch = rem // C_in

    # out1: full_ch = out_ch
    in_ch_1 = out_ch >> 1
    is_odd_1 = out_ch & 1
    src_offset_1 = batch * (C_in * HW) + in_ch_1 * HW + hw_idx
    sig_offset_1 = batch * C_in + in_ch_1

    in3_v1 = tl.load(in_3_ptr + src_offset_1, mask=mask & (is_odd_1 == 0), other=0.0)
    in5_v1 = tl.load(in_5_ptr + src_offset_1, mask=mask & (is_odd_1 == 1), other=0.0)
    sig_v1 = tl.load(sigmoid_ptr + sig_offset_1, mask=mask & (is_odd_1 == 1), other=1.0)
    val1 = tl.where(is_odd_1 == 1, in5_v1 * sig_v1, in3_v1)
    tl.store(out1_ptr + offs, val1, mask=mask)

    # out2: full_ch = out_ch + C_in
    full_ch_2 = out_ch + C_in
    in_ch_2 = full_ch_2 >> 1
    is_odd_2 = full_ch_2 & 1
    src_offset_2 = batch * (C_in * HW) + in_ch_2 * HW + hw_idx
    sig_offset_2 = batch * C_in + in_ch_2

    in3_v2 = tl.load(in_3_ptr + src_offset_2, mask=mask & (is_odd_2 == 0), other=0.0)
    in5_v2 = tl.load(in_5_ptr + src_offset_2, mask=mask & (is_odd_2 == 1), other=0.0)
    sig_v2 = tl.load(sigmoid_ptr + sig_offset_2, mask=mask & (is_odd_2 == 1), other=1.0)
    val2 = tl.where(is_odd_2 == 1, in5_v2 * sig_v2, in3_v2)
    tl.store(out2_ptr + offs, val2, mask=mask)


@torch.fx.wrap
def dispatch(*args):
    route = args[-1]
    if route == "p1":
        a, b = args[0], args[1]
        B_dim = a.shape[0]
        C_in = a.shape[1]  # 20
        H = a.shape[2]     # 64
        W = a.shape[3]     # 48
        HW = H * W
        N_half = B_dim * C_in * HW

        out1 = torch.empty(B_dim, C_in, H, W, device=a.device, dtype=a.dtype)
        out2 = torch.empty(B_dim, C_in, H, W, device=a.device, dtype=a.dtype)

        BLOCK_SIZE = 2048
        grid = ((N_half + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        channel_interleave_2out_kernel[grid](
            a, b, out1, out2,
            N_half, C_in, HW,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return (out1, out2)

    else:  # route == "p2"
        in_0, in_1, in_3, in_5, in_6 = args[0], args[1], args[2], args[3], args[4]
        B_dim = in_3.shape[0]
        C_in = in_3.shape[1]   # 40
        H = in_3.shape[2]      # 32
        W = in_3.shape[3]      # 24
        HW = H * W
        IN_C = in_6.shape[1]   # 10

        # Step 1: Linear + Sigmoid -> [B, 40]
        sigmoid_out = torch.empty(B_dim, C_in, device=in_3.device, dtype=in_3.dtype)

        BLOCK_C = 64
        grid_ls = (B_dim, (C_in + BLOCK_C - 1) // BLOCK_C)
        linear_sigmoid_kernel[grid_ls](
            in_6, in_1, in_0, sigmoid_out,
            B_dim, C_in, IN_C,
            BLOCK_C=BLOCK_C,
        )

        # Step 2: Interleave with multiply -> two outputs [B, C_in, H, W] each
        N_half = B_dim * C_in * HW
        out1 = torch.empty(B_dim, C_in, H, W, device=in_3.device, dtype=in_3.dtype)
        out2 = torch.empty(B_dim, C_in, H, W, device=in_3.device, dtype=in_3.dtype)

        BLOCK_SIZE = 2048
        grid = ((N_half + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        interleave_mul_2out_kernel[grid](
            in_3, in_5, sigmoid_out, out1, out2,
            N_half, C_in, HW,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return (out1, out2)