import torch
import triton
import triton.language as tl

def pattern(in_5, in_4, tmp_0, tmp_1, tmp_3, tmp_2):
    """
    Pattern: fused multiplication, batch normalization, and SiLU activation
    tmp_0 = in_0
    tmp_1 = in_1  
    tmp_2 = in_2
    tmp_3 = in_3
    tmp_4 = in_5 * in_4
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    """
    # Element-wise multiplication
    tmp_4 = in_5 * in_4
    
    # Batch normalization
    tmp_5 = torch.nn.functional.batch_norm(
        tmp_4, 
        tmp_0, 
        tmp_1, 
        tmp_3, 
        tmp_2, 
        False,  # training=False
        0.1,    # momentum=0.1
        1e-05   # eps=1e-05
    )
    
    # SiLU activation
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    
    return tmp_6

def replacement_args(in_5, in_4, in_0, in_1, in_3, in_2):
    return (in_5, in_4, in_0, in_1, in_3, in_2)

@triton.jit
def fused_mul_bnorm_silu_kernel(
    x_ptr,           # Main input tensor (mul_input1)
    sigmoid_ptr,     # Sigmoid input tensor (mul_input2)
    mean_ptr,        # Batch norm running mean
    var_ptr,         # Batch norm running variance
    weight_ptr,      # Batch norm weight
    bias_ptr,        # Batch norm bias
    out_ptr,         # Output tensor
    n_elements,      # Total number of elements
    batch_size,      # Batch dimension
    channels,        # Number of channels
    height,          # Spatial height
    width,           # Spatial width
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr
):
    # Each program handles one channel in the channel dimension
    pid = tl.program_id(0)
    batch_pid = tl.program_id(1)
    
    # Calculate channel range for this program
    channel_start = pid * CHANNELS_PER_BLOCK
    channel_end = min(channel_start + CHANNELS_PER_BLOCK, channels)
    
    # Early exit if no channels to process
    if channel_start >= channels:
        return
    
    # Load batch norm parameters for this channel range
    if pid * CHANNELS_PER_BLOCK < channels:
        # Load means and variances
        mean = tl.load(mean_ptr + channel_start)
        var = tl.load(var_ptr + channel_start)
        
        # Load weights and biases for this channel range
        if weight_ptr is not None:
            weight = tl.load(weight_ptr + channel_start)
        else:
            weight = 1.0
            
        if bias_ptr is not None:
            bias = tl.load(bias_ptr + channel_start)
        else:
            bias = 0.0
        
        # Compute batch normalization parameters
        # var + eps where eps = 1e-05
        inv_std = 1.0 / tl.sqrt(var + 1e-05)
        
        # Create masks for channel processing
        mask_cond = tl.arange(0, CHANNELS_PER_BLOCK) < (channel_end - channel_start)
    else:
        return
    
    # Process each element in the batch
    for h in range(height):
        for w in range(width):
            # Load sigmoid values (broadcastable across spatial dimensions)
            sigmoid_val = tl.load(sigmoid_ptr + batch_pid * channels * height * width + 
                                channel_start * height * width + h * width + w, 
                                mask=mask_cond, other=0.0)
            
            # Process each channel in the current channel range
            for ch_off in range(0, channel_end - channel_start):
                channel_idx = channel_start + ch_off
                if mask_cond[ch_off]:
                    # Load main input value
                    src_offset = (batch_pid * channels + channel_idx) * height * width + h * width + w
                    x_val = tl.load(x_ptr + src_offset, other=0.0)
                    
                    # Fused computation:
                    # 1. Element-wise multiplication with sigmoid
                    mul_result = x_val * sigmoid_val
                    
                    # 2. Batch normalization: (x - mean) / sqrt(var + eps) * weight + bias
                    bn_result = (mul_result - mean) * inv_std * weight + bias
                    
                    # 3. SiLU activation: x * sigmoid(x)
                    silu_result = bn_result * tl.sigmoid(bn_result)
                    
                    # Store result
                    out_offset = (batch_pid * channels + channel_idx) * height * width + h * width + w
                    tl.store(out_ptr + out_offset, silu_result)

@torch.fx.wrap
def fused_mul_bnorm_silu(in_5, in_4, tmp_0, tmp_1, tmp_3, tmp_2):
    # Get tensor shapes from main input
    batch_size, channels, height, width = in_5.shape
    
    # Create output tensor
    out = torch.empty_like(in_5)
    
    # Block size for spatial optimization
    BLOCK_SIZE = 1024
    # Channels per block to balance GPU occupancy
    CHANNELS_PER_BLOCK = 32  # Process 32 channels per program
    
    # Calculate grid dimensions
    num_channel_programs = (channels + CHANNELS_PER_BLOCK - 1) // CHANNELS_PER_BLOCK
    
    # Launch kernel
    fused_mul_bnorm_silu_kernel[
        (num_channel_programs, batch_size),  # Grid dimensions: (channel_programs, batch_size)
        (1, 1, 1)  # Block dimensions
    ](
        x_ptr=in_5,
        sigmoid_ptr=in_4,
        mean_ptr=tmp_0,
        var_ptr=tmp_1,
        weight_ptr=tmp_3,
        bias_ptr=tmp_2,
        out_ptr=out,
        n_elements=in_5.numel(),
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK
    )
    
    return out

def replacement_func():
    return fused_mul_bnorm_silu