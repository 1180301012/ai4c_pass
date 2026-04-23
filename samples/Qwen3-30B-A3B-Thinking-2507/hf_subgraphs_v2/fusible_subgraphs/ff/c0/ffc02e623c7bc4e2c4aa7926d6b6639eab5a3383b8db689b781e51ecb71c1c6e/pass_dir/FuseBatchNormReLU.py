import torch
import triton
import triton.language as tl

def pattern(tmp_6, in_1, in_2, in_3, in_4):
    tmp_7 = torch.nn.functional.batch_norm(tmp_6, in_1, in_2, in_4, in_3, False, 0.1, 1e-05)
    tmp_8 = torch.nn.functional.relu(tmp_7, inplace = True)
    return tmp_8

def replacement_args(tmp_6, in_1, in_2, in_3, in_4):
    return (tmp_6, in_1, in_2, in_3, in_4)

@triton.jit
def fused_bn_relu_kernel(
    input_ptr,
    mean_ptr,
    var_ptr,
    gamma_ptr,
    beta_ptr,
    output_ptr,
    batch_size,
    num_channels,
    BLOCK_SIZE_CHANS: tl.constexpr,
):
    block_idx = tl.program_id(0)
    channel_start = block_idx * BLOCK_SIZE_CHANS
    channel_end = min(channel_start + BLOCK_SIZE_CHANS, num_channels)
    
    chan = channel_start + tl.thread_id(0)
    if chan >= channel_end:
        return
    
    for batch_idx in range(batch_size):
        input_val = tl.load(input_ptr + batch_idx * num_channels + chan)
        
        mean_val = tl.load(mean_ptr + chan)
        var_val = tl.load(var_ptr + chan)
        gamma_val = tl.load(gamma_ptr + chan)
        beta_val = tl.load(beta_ptr + chan)
        
        norm = (input_val - mean_val) / tl.sqrt(var_val + 1e-5)
        bn_val = norm * gamma_val + beta_val
        output_val = tl.maximum(bn_val, 0.0)
        
        tl.store(output_ptr + batch_idx * num_channels + chan, output_val)

@torch.fx.wrap
def fused_bn_relu(input, mean, var, gamma, beta):
    batch_size = input.shape[0]
    num_channels = input.shape[1]
    assert input.shape[2] == 1 and input.shape[3] == 1, "Only works for spatial dimensions 1x1"
    
    BLOCK_SIZE_CHANS = 128
    num_blocks = (num_channels + BLOCK_SIZE_CHANS - 1) // BLOCK_SIZE_CHANS
    
    out = torch.empty_like(input)
    
    fused_bn_relu_kernel[(num_blocks,)](
        input,
        mean,
        var,
        gamma,
        beta,
        out,
        batch_size,
        num_channels,
        BLOCK_SIZE_CHANS,
    )
    
    return out

def replacement_func():
    return fused_bn_relu