import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(conv2d.shape[0], 1, conv2d.shape[2] * conv2d.shape[3])
    tmp_4 = torch.nn.functional.softmax(tmp_3, 2, _stacklevel=5)
    tmp_5 = tmp_4.unsqueeze(-1)
    return (tmp_5,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_conv2d_softmax_kernel(
    input_ptr,        # in_2: [N, C_in, H, W]
    weight_ptr,       # in_1: [1, C_in, 1, 1]
    bias_ptr,         # in_0: [1]
    output_ptr,       # output: [N, 1, H*W, 1]
    N, C_in, H, W,
    # Input strides
    in_stride_n, in_stride_c, in_stride_h, in_stride_w,
    # Weight strides
    wt_stride_0, wt_stride_c, wt_stride_2, wt_stride_3,
    # Output strides
    out_stride_n, out_stride_1, out_stride_s, out_stride_last,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """
    Fused kernel: 1x1 conv2d (1 output channel) + reshape + softmax(dim=2) + unsqueeze(-1)
    Each program handles one batch element (n).
    
    The conv2d computes: conv_out[n, 0, h, w] = sum_c(weight[0,c,0,0] * input[n,c,h,w]) + bias[0]
    Then softmax over the spatial dimension (H*W) for each n.
    Output shape: [N, 1, H*W, 1]
    """
    n = tl.program_id(0)
    spatial_size = H * W
    bias_val = tl.load(bias_ptr).to(tl.float32)

    # Phase 1: Compute conv values and find max for softmax
    max_val = -float('inf')
    
    for s_start in range(0, spatial_size, BLOCK_S):
        s_offsets = s_start + tl.arange(0, BLOCK_S)
        s_mask = s_offsets < spatial_size
        
        h_off = s_offsets // W
        w_off = s_offsets % W
        
        # Compute 1x1 conv: dot product over C_in channels
        acc = tl.zeros([BLOCK_S], dtype=tl.float32)
        for c_start in range(0, C_in, BLOCK_C):
            c_offsets = c_start + tl.arange(0, BLOCK_C)
            c_mask = c_offsets < C_in
            
            # Load weight[0, c, 0, 0] = weight_ptr[c * wt_stride_c]
            w_vals = tl.load(weight_ptr + c_offsets * wt_stride_c, mask=c_mask, other=0.0).to(tl.float32)
            
            # Load input[n, c, h, w]
            # input_ptr + n*in_stride_n + c*in_stride_c + h*in_stride_h + w*in_stride_w
            in_ptrs = input_ptr + n * in_stride_n + (c_start + tl.arange(0, BLOCK_C)[:, None]) * in_stride_c \
                      + h_off[None, :] * in_stride_h + w_off[None, :] * in_stride_w
            c_s_mask = c_mask[:, None] & s_mask[None, :]
            in_vals = tl.load(in_ptrs, mask=c_s_mask, other=0.0).to(tl.float32)
            
            # acc[s] += sum_c(weight[c] * input[n, c, h(s), w(s)])
            acc += tl.sum(w_vals[:, None] * in_vals, axis=0)
        
        # Add bias
        conv_vals = acc + bias_val
        
        # Update max
        block_max = tl.max(conv_vals, axis=0)  # max over BLOCK_S elements
        max_val = tl.maximum(max_val, block_max)

    # Phase 2: Compute sum of exp(x - max)
    sum_val = 0.0
    
    for s_start in range(0, spatial_size, BLOCK_S):
        s_offsets = s_start + tl.arange(0, BLOCK_S)
        s_mask = s_offsets < spatial_size
        
        h_off = s_offsets // W
        w_off = s_offsets % W
        
        # Recompute conv values
        acc = tl.zeros([BLOCK_S], dtype=tl.float32)
        for c_start in range(0, C_in, BLOCK_C):
            c_offsets = c_start + tl.arange(0, BLOCK_C)
            c_mask = c_offsets < C_in
            
            w_vals = tl.load(weight_ptr + c_offsets * wt_stride_c, mask=c_mask, other=0.0).to(tl.float32)
            
            in_ptrs = input_ptr + n * in_stride_n + (c_start + tl.arange(0, BLOCK_C)[:, None]) * in_stride_c \
                      + h_off[None, :] * in_stride_h + w_off[None, :] * in_stride_w
            c_s_mask = c_mask[:, None] & s_mask[None, :]
            in_vals = tl.load(in_ptrs, mask=c_s_mask, other=0.0).to(tl.float32)
            
            acc += tl.sum(w_vals[:, None] * in_vals, axis=0)
        
        conv_vals = acc + bias_val
        
        # Compute exp(x - max) and accumulate sum
        exp_vals = tl.exp(conv_vals - max_val)
        sum_val += tl.sum(exp_vals)

    # Phase 3: Compute softmax and store output
    out_base = output_ptr + n * out_stride_n
    
    for s_start in range(0, spatial_size, BLOCK_S):
        s_offsets = s_start + tl.arange(0, BLOCK_S)
        s_mask = s_offsets < spatial_size
        
        h_off = s_offsets // W
        w_off = s_offsets % W
        
        # Recompute conv values
        acc = tl.zeros([BLOCK_S], dtype=tl.float32)
        for c_start in range(0, C_in, BLOCK_C):
            c_offsets = c_start + tl.arange(0, BLOCK_C)
            c_mask = c_offsets < C_in
            
            w_vals = tl.load(weight_ptr + c_offsets * wt_stride_c, mask=c_mask, other=0.0).to(tl.float32)
            
            in_ptrs = input_ptr + n * in_stride_n + (c_start + tl.arange(0, BLOCK_C)[:, None]) * in_stride_c \
                      + h_off[None, :] * in_stride_h + w_off[None, :] * in_stride_w
            c_s_mask = c_mask[:, None] & s_mask[None, :]
            in_vals = tl.load(in_ptrs, mask=c_s_mask, other=0.0).to(tl.float32)
            
            acc += tl.sum(w_vals[:, None] * in_vals, axis=0)
        
        conv_vals = acc + bias_val
        
        # Compute softmax: exp(x - max) / sum
        softmax_vals = tl.exp(conv_vals - max_val) / sum_val
        
        # Store to output[n, 0, s, 0]
        # output_ptr + n*out_stride_n + 0*out_stride_1 + s*out_stride_s + 0*out_stride_last
        out_ptrs = out_base + s_offsets * out_stride_s
        tl.store(out_ptrs, softmax_vals, mask=s_mask)


@torch.fx.wrap
def fused_conv2d_softmax_unsqueeze(in_0, in_1, in_2):
    """
    Fused implementation of conv2d + view + softmax + unsqueeze.
    
    Args:
        in_0: bias tensor [1]
        in_1: weight tensor [1, C_in, 1, 1] 
        in_2: input tensor [N, C_in, H, W]
    
    Returns:
        output tensor [N, 1, H*W, 1] (softmax result with unsqueezed dimension)
    """
    N = in_2.shape[0]
    C_in = in_2.shape[1]
    H = in_2.shape[2]
    W = in_2.shape[3]
    spatial_size = H * W
    
    # Allocate output: [N, 1, spatial_size, 1]
    output = torch.empty((N, 1, spatial_size, 1), dtype=in_2.dtype, device=in_2.device)
    
    # Determine block sizes
    BLOCK_C = min(triton.next_power_of_2(C_in), 512)
    BLOCK_S = min(triton.next_power_of_2(spatial_size), 4096)
    
    grid = (N,)
    
    fused_conv2d_softmax_kernel[grid](
        input_ptr=in_2,
        weight_ptr=in_1,
        bias_ptr=in_0,
        output_ptr=output,
        N=N, C_in=C_in, H=H, W=W,
        in_stride_n=in_2.stride(0), in_stride_c=in_2.stride(1),
        in_stride_h=in_2.stride(2), in_stride_w=in_2.stride(3),
        wt_stride_0=in_1.stride(0), wt_stride_c=in_1.stride(1),
        wt_stride_2=in_1.stride(2), wt_stride_3=in_1.stride(3),
        out_stride_n=output.stride(0), out_stride_1=output.stride(1),
        out_stride_s=output.stride(2), out_stride_last=output.stride(3),
        BLOCK_C=BLOCK_C,
        BLOCK_S=BLOCK_S,
    )
    
    return output


def replacement_func():
    return fused_conv2d_softmax_unsqueeze