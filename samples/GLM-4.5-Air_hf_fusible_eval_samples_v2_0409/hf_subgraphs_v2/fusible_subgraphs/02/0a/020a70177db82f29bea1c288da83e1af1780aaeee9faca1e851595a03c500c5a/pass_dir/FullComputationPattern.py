import torch
import triton
import triton.language as tl

def pattern(in_0):
    """Pattern: Full computation sequence - silu -> adaptive_avg_pool2d -> flatten -> dropout"""
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
    tmp_2 = torch.flatten(tmp_1, 1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.2, False, True)
    return tmp_3

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def full_optimized_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    height,
    width,
    p_dropout,
    BLOCK_SIZE: tl.constexpr,
):
    """Fully optimized kernel combining all operations"""
    pid = tl.program_id(0)
    
    # Process each batch element
    batch_offset = pid * channels
    
    # Compute global average pooling with SiLU activation
    for channel_idx in tl.range(0, channels, BLOCK_SIZE):
        channel_mask = (channel_idx + tl.arange(0, BLOCK_SIZE)) < channels
        
        # Initialize for global average pooling with SiLU
        sum_val = tl.zeros(1, dtype=tl.float32)
        count = tl.zeros(1, dtype=tl.int32)
        
        # Process spatial dimensions
        for h_idx in range(height):
            for w_idx in range(width):
                # Compute base index
                base_idx = batch_offset + (h_idx * width + w_idx) * channels
                
                # Load input data
                x_vals = tl.load(
                    input_ptr + base_idx + channel_idx,
                    mask=channel_mask,
                    other=0.0
                )
                
                # Apply SiLU: x * sigmoid(x)
                sigmoid_x = 1.0 / (1.0 + tl.exp(-x_vals))
                silu_vals = x_vals * sigmoid_x
                
                # Sum for global average pooling
                tl.store(sum_val, sum_val + tl.sum(silu_vals))
                tl.store(count, count + tl.sum(channel_mask))
        
        # Compute average
        avg_val = sum_val / count if count > 0 else 0.0
        
        # Apply dropout (no-op when training=False, but keeping for consistency)
        dropout_vals = avg_val * (1.0 - p_dropout) if p_dropout > 0 else avg_val
        
        # Store result
        output_indices = channel_idx + tl.arange(0, BLOCK_SIZE)
        tl.store(
            output_ptr + pid * channels + output_indices,
            dropout_vals,
            mask=channel_mask
        )

@torch.fx.wrap
def fully_optimized_forward(input_tensor):
    """Fully optimized forward pass"""
    batch_size, channels, height, width = input_tensor.shape
    
    # Determine dropout rate from typical values
    p_dropout = 0.2  # Default value, can be made configurable
    
    # Block size for optimization
    BLOCK_SIZE = 128
    num_batches = (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_batches, 1, 1)
    
    # Create output tensor
    output_shape = (batch_size, channels)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch optimized kernel
    full_optimized_kernel[grid](
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        p_dropout=p_dropout,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fully_optimized_forward