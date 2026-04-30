import torch
import triton
import triton.language as tl


@triton.jit
def fused_softmax_mul_sum_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    total_elements,
    B, C, H, W,
    stride_b0, stride_k0, stride_c0, stride_h0, stride_w0,
    stride_b1, stride_k1, stride_d1, stride_c1,
    stride_b_out, stride_c_out, stride_h_out, stride_w_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Decode flat index to (b, c, h, w)
    chw = C * H * W
    hw = H * W
    b = offsets // chw
    rem = offsets - b * chw
    c = rem // hw
    rem2 = rem - c * hw
    h = rem2 // W
    w = rem2 - h * W

    # Load in_1 values for softmax: in_1[b, k, 0, c] for k=0,1
    # Note: 0 * stride_d1 = 0, so we skip that term
    idx_1_0 = b * stride_b1 + c * stride_c1
    idx_1_1 = idx_1_0 + stride_k1

    v1_0 = tl.load(in_1_ptr + idx_1_0, mask=mask)
    v1_1 = tl.load(in_1_ptr + idx_1_1, mask=mask)

    # Compute softmax over 2 values (numerically stable)
    mx = tl.maximum(v1_0, v1_1)
    e0 = tl.exp(v1_0 - mx)
    e1 = tl.exp(v1_1 - mx)
    s = e0 + e1
    w0 = e0 / s
    w1 = e1 / s

    # Load in_0 values for both channels: in_0[b, k, c, h, w] for k=0,1
    base_idx_0 = b * stride_b0 + c * stride_c0 + h * stride_h0 + w * stride_w0
    v0_0 = tl.load(in_0_ptr + base_idx_0, mask=mask)
    v0_1 = tl.load(in_0_ptr + base_idx_0 + stride_k0, mask=mask)

    # Compute weighted sum
    out = w0 * v0_0 + w1 * v0_1

    # Store output
    idx_out = b * stride_b_out + c * stride_c_out + h * stride_h_out + w * stride_w_out
    tl.store(out_ptr + idx_out, out, mask=mask)


@torch.fx.wrap
def fused_softmax_mul_sum(in_0, in_1):
    """Fused softmax-broadcast-multiply-sum kernel.
    
    Computes: output[b,c,h,w] = sum_k(softmax(in_1, dim=1)[b,k,0,c] * in_0[b,k,c,h,w])
    
    in_0: [B, 2, C, H, W] 5D tensor
    in_1: [B, 2, 1, C] 4D tensor  
    output: [B, C, H, W] 4D tensor
    """
    B = in_0.shape[0]
    C = in_0.shape[2]
    H = in_0.shape[3]
    W = in_0.shape[4]

    out = torch.empty((B, C, H, W), dtype=in_0.dtype, device=in_0.device)

    total = B * C * H * W
    BLOCK_SIZE = 1024
    num_programs = (total + BLOCK_SIZE - 1) // BLOCK_SIZE

    fused_softmax_mul_sum_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        total_elements=total,
        B=B, C=C, H=H, W=W,
        stride_b0=in_0.stride()[0], stride_k0=in_0.stride()[1],
        stride_c0=in_0.stride()[2], stride_h0=in_0.stride()[3], stride_w0=in_0.stride()[4],
        stride_b1=in_1.stride()[0], stride_k1=in_1.stride()[1],
        stride_d1=in_1.stride()[2], stride_c1=in_1.stride()[3],
        stride_b_out=out.stride()[0], stride_c_out=out.stride()[1],
        stride_h_out=out.stride()[2], stride_w_out=out.stride()[3],
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


@torch.fx.wrap
def fused_softmax_mul_sum_dispatch(in_0, in_1, route):
    """Dispatch wrapper shared across all pass files.
    All routes call the same kernel since it's shape-agnostic.
    """
    if route == "route_b8" or route == "route_b1" or route == "route_b2":
        return fused_softmax_mul_sum(in_0, in_1)
    else:
        raise ValueError(f"Unknown route: {route}")