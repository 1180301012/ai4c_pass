import torch
import triton
import triton.language as tl

# Pattern matching: Matches sigmoid -> view -> expand -> mul -> add -> relu -> adaptive_avg_pool2d -> flatten
def pattern(in_0, in_1, in_2):
    tmp_0 = in_2.sigmoid()
    tmp_1 = tmp_0.view(1, -1, 1, 1)
    tmp_2 = tmp_1.expand_as(in_1)
    tmp_3 = in_1 * tmp_2
    tmp_3 = tmp_3 + in_0
    tmp_4 = tmp_3
    tmp_5 = torch.nn.functional.relu(tmp_4, inplace=True)
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    return tmp_7

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_se_attention_pool_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, out_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    stride_in: tl.constexpr, HW: tl.constexpr
):
    # pid_b and pid_c identify the channel slice we're working on
    pid_bc = tl.program_id(0)  # 1D grid: B * C programs
    
    pid_b = pid_bc // C
    pid_c = pid_bc % C
    
    # Base channel offset in the input tensors [B, C, H, W]
    ch_offset = pid_b * C * HW + pid_c * HW
    
    # Load the sigmoid weight for this channel from in_2 [1, 1, C]
    weight_offset = pid_c  # in_2 is [1, 1, 2048], we need [0, 0, pid_c]
    sigmoid_weight = tl.load(in_2_ptr + weight_offset)
    sigmoid_weight = 1.0 / (1.0 + tl.exp(-sigmoid_weight))
    
    # Accumulator for average pooling
    accumulator = 0.0
    HW_f = float(HW)
    
    # Loop over all spatial positions (H*W)
    for h in range(H):
        for w in range(W):
            offset = ch_offset + h * W + w
            
            # Load in_0 and in_1 values
            in_0_val = tl.load(in_0_ptr + offset)
            in_1_val = tl.load(in_1_ptr + offset)
            
            # SE attention: mul with sigmoid weight
            se_out = in_1_val * sigmoid_weight
            
            # Add residual and apply ReLU
            fused = se_out + in_0_val
            fused = tl.where(fused > 0, fused, 0.0)
            
            accumulator += fused
    
    # Average pooling
    pooled = accumulator / HW_f
    
    # Store output [B, C]
    out_offset = pid_b * C + pid_c
    tl.store(out_ptr + out_offset, pooled)

@torch.fx.wrap
def fused_se_attention_pool(in_0, in_1, in_2):
    B, C, H, W = in_0.shape
    out = torch.empty((B, C), dtype=in_0.dtype, device=in_0.device)
    
    # 1D grid: one program per batch per channel
    grid = (B * C,)
    
    HW = H * W
    
    fused_se_attention_pool_kernel[grid](
        in_0, in_1, in_2, out,
        B, C, H, W,
        in_0.stride(0), HW
    )
    
    return out

def replacement_func():
    return fused_se_attention_pool