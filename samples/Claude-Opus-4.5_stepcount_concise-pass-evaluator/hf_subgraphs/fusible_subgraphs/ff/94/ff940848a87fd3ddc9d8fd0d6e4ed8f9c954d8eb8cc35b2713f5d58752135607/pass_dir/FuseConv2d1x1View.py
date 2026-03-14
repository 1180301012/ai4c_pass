import torch
import triton
import triton.language as tl

# Pattern: conv2d(1x1) followed by view
# Input: [B, C_in, H, W] -> conv2d -> [B, C_out, H, W] -> view -> [B, C_out, H*W]
# This is equivalent to: input.view(B, C_in, -1).permute(0,2,1) @ weight.view(C_out, C_in).T + bias
# Then permute back to [B, C_out, H*W]

def pattern(x, weight, bias):
    # 1x1 conv2d with standard args
    conv_out = torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    # View to flatten spatial dims - use -1 for dynamic shape
    out = conv_out.view(conv_out.size(0), conv_out.size(1), -1)
    return out

def replacement_args(x, weight, bias):
    return (x, weight, bias)

@triton.jit
def conv1x1_view_kernel(
    x_ptr,        # [B, C_in, H*W]
    w_ptr,        # [C_out, C_in]
    b_ptr,        # [C_out]
    out_ptr,      # [B, C_out, H*W]
    B, C_in, C_out, HW,
    stride_xb, stride_xc, stride_xs,
    stride_ob, stride_oc, stride_os,
    BLOCK_HW: tl.constexpr,
    BLOCK_CIN: tl.constexpr,
):
    # Grid: (B, cdiv(HW, BLOCK_HW), C_out)
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_cout = tl.program_id(2)
    
    hw_offsets = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offsets < HW
    
    # Load bias for this output channel
    bias_val = tl.load(b_ptr + pid_cout)
    
    # Accumulator
    acc = tl.zeros([BLOCK_HW], dtype=tl.float32) + bias_val
    
    # Loop over input channels
    for cin_start in range(0, C_in, BLOCK_CIN):
        cin_offsets = cin_start + tl.arange(0, BLOCK_CIN)
        cin_mask = cin_offsets < C_in
        
        # Load weights: w[pid_cout, cin_offsets]
        w_offset = pid_cout * C_in + cin_offsets
        w_vals = tl.load(w_ptr + w_offset, mask=cin_mask, other=0.0)
        
        # For each cin, load x values and accumulate
        for i in range(BLOCK_CIN):
            cin = cin_start + i
            if cin < C_in:
                # x[pid_b, cin, hw_offsets]
                x_offset = pid_b * stride_xb + cin * stride_xc + hw_offsets
                x_vals = tl.load(x_ptr + x_offset, mask=hw_mask, other=0.0)
                acc += x_vals * w_vals[i]
    
    # Store output: out[pid_b, pid_cout, hw_offsets]
    out_offset = pid_b * stride_ob + pid_cout * stride_oc + hw_offsets
    tl.store(out_ptr + out_offset, acc, mask=hw_mask)

@torch.fx.wrap
def conv1x1_view_triton(x, weight, bias):
    B, C_in, H, W = x.shape
    C_out = weight.shape[0]
    HW = H * W
    
    # Reshape x to [B, C_in, H*W] - this is a view, no copy
    x_flat = x.view(B, C_in, HW)
    
    # Reshape weight to [C_out, C_in] - this is a view, no copy
    w_flat = weight.view(C_out, C_in)
    
    # Output shape [B, C_out, H*W]
    out = torch.empty((B, C_out, HW), dtype=x.dtype, device=x.device)
    
    BLOCK_HW = 128
    BLOCK_CIN = 32
    
    grid = (B, (HW + BLOCK_HW - 1) // BLOCK_HW, C_out)
    
    conv1x1_view_kernel[grid](
        x_ptr=x_flat,
        w_ptr=w_flat,
        b_ptr=bias,
        out_ptr=out,
        B=B, C_in=C_in, C_out=C_out, HW=HW,
        stride_xb=C_in * HW, stride_xc=HW, stride_xs=1,
        stride_ob=C_out * HW, stride_oc=HW, stride_os=1,
        BLOCK_HW=BLOCK_HW,
        BLOCK_CIN=BLOCK_CIN,
    )
    
    return out

def replacement_func():
    return conv1x1_view_triton