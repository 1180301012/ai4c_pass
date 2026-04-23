import torch
import triton
import triton.language as tl


@triton.jit
def conv2d_view_softmax_unsqueeze_kernel(
    input_ptr,      # [B, C_in, H, W]
    weight_ptr,     # [C_in] (squeezed from [1, C_in, 1, 1])
    bias_ptr,       # [1]
    output_ptr,     # [B, 1, L, 1]
    B, C_in, H, W, L,
    stride_ib, stride_ic, stride_ih, stride_iw,  # input strides
    stride_ob, stride_o1, stride_oL, stride_o1w,  # output strides
    BLOCK_L: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Fused 1x1 conv2d + view + softmax + unsqueeze kernel.
    
    Each program handles one batch element.
    Computes: output[b, 0, hw, 1] = softmax(conv2d(input[b], weight, bias))[hw]
    
    Reads input only once, computes conv output into accumulator,
    then does online softmax in-place.
    """
    b = tl.program_id(0)
    
    # Load bias
    bias_val = tl.load(bias_ptr)
    
    # Spatial position offsets
    l_offsets = tl.arange(0, BLOCK_L)
    l_mask = l_offsets < L
    
    # Compute h and w from l = h * W + w
    h_indices = l_offsets // W
    w_indices = l_offsets % W
    
    # Phase 1: Compute conv output (dot product over C_in channels)
    # Iterate over C_in in chunks to keep 2D load size manageable
    acc = tl.zeros([BLOCK_L], dtype=tl.float32)
    
    for c_start in range(0, C_in, BLOCK_C):
        c_offsets = tl.arange(0, BLOCK_C)
        c_mask = c_offsets < C_in
        
        # Load weight chunk
        w_chunk = tl.load(weight_ptr + c_start + c_offsets, mask=c_mask, other=0.0)
        
        # Load input chunk: [BLOCK_L, BLOCK_C]
        # input[b, c, h, w] where c = c_start + c_offsets
        input_offsets = b * stride_ib + (c_start + c_offsets[None, :]) * stride_ic + h_indices[:, None] * stride_ih + w_indices[:, None] * stride_iw
        input_mask = l_mask[:, None] & c_mask[None, :]
        x = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
        
        # Accumulate: acc[l] += sum_c(x[l, c] * w[c])
        acc += tl.sum(x * w_chunk[None, :], axis=1)
    
    # Add bias
    conv_vals = acc + bias_val
    
    # Mask out-of-bounds positions for softmax
    conv_vals_masked = tl.where(l_mask, conv_vals, -float('inf'))
    
    # Phase 2: Online softmax (in-place, no extra memory reads)
    # Find max
    max_val = tl.max(conv_vals_masked, axis=0)
    
    # Compute exp(conv - max)
    exp_vals = tl.exp(conv_vals_masked - max_val)
    exp_vals = tl.where(l_mask, exp_vals, 0.0)
    
    # Sum of exponentials
    exp_sum = tl.sum(exp_vals, axis=0)
    
    # Normalize
    softmax_vals = exp_vals / exp_sum
    softmax_vals = tl.where(l_mask, softmax_vals, 0.0)
    
    # Phase 3: Store output with unsqueeze(-1)
    # output[b, 0, l, 1] = softmax_vals[l]
    output_offsets = b * stride_ob + l_offsets * stride_oL
    tl.store(output_ptr + output_offsets, softmax_vals, mask=l_mask)


@torch.fx.wrap
def fused_conv2d_view_softmax_unsqueeze(input, weight, bias):
    """
    Fused implementation of: conv2d(1x1) -> view(B,1,L) -> softmax(dim=2) -> unsqueeze(-1)
    
    Args:
        input: [B, C_in, H, W]
        weight: [1, C_in, 1, 1]  
        bias: [1]
    
    Returns:
        output: [B, 1, H*W, 1]
    """
    B = input.shape[0]
    C_in = input.shape[1]
    H = input.shape[2]
    W = input.shape[3]
    L = H * W
    
    # Determine dtype
    dtype = input.dtype
    
    # Create output tensor: [B, 1, L, 1]
    output = torch.empty((B, 1, L, 1), dtype=dtype, device=input.device)
    
    # Compute strides
    stride_ib, stride_ic, stride_ih, stride_iw = input.stride()
    stride_ob, stride_o1, stride_oL, stride_o1w = output.stride()
    
    # Choose block sizes - BLOCK_L must cover entire L for in-place softmax
    BLOCK_L = triton.next_power_of_2(L)
    # Limit BLOCK_L to avoid excessive register pressure
    BLOCK_L = min(BLOCK_L, 8192)
    
    # BLOCK_C for chunked reduction over input channels
    BLOCK_C = 256
    
    # Squeeze weight to [C_in] for efficient loading
    w_squeezed = weight.reshape(C_in).contiguous()
    
    # Launch kernel - one program per batch element
    grid = (B,)
    
    conv2d_view_softmax_unsqueeze_kernel[grid](
        input_ptr=input,
        weight_ptr=w_squeezed,
        bias_ptr=bias,
        output_ptr=output,
        B=B, C_in=C_in, H=H, W=W, L=L,
        stride_ib=stride_ib, stride_ic=stride_ic, stride_ih=stride_ih, stride_iw=stride_iw,
        stride_ob=stride_ob, stride_o1=stride_o1, stride_oL=stride_oL, stride_o1w=stride_o1w,
        BLOCK_L=BLOCK_L,
        BLOCK_C=BLOCK_C,
    )
    
    return output