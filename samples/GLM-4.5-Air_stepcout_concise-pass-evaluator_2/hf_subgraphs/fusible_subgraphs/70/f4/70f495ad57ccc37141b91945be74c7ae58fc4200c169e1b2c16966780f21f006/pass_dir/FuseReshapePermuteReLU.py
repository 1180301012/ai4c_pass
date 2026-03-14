import torch
import triton
import triton.language as tl


# Pattern matching function - matches reshape + permute + relu pattern
def pattern(in_0):
    """
    Pattern: reshape -> permute -> relu
    This fuses three operations into one kernel.
    
    in_0: input tensor [batch, 192, 1280]
    """
    # Original operations
    tmp_1 = in_0.reshape(1, 16, 12, -1)  # for batch=1
    tmp_2 = tmp_1.permute(0, 3, 1, 2)
    tmp_3 = torch.nn.functional.relu(tmp_2)
    return tmp_3


# Argument extraction function
def replacement_args(in_0):
    return (in_0,)


# Optimized Triton kernel for fused reshape + permute + relu
@triton.jit
def reshape_permute_relu_kernel(
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one element
    pid = tl.program_id(0)
    
    # Calculate which batch this program handles
    batch_idx = pid  # For now, simple mapping
    
    # The reshape transforms [batch, seq_len, hidden] to [batch, 16, 12, 192]
    # The permute transforms to [batch, 192, 16, 12]
    # The relu applies element-wise relu
    
    # Process each element in the output
    for seq_idx in range(seq_len):
        for i in range(0, 16 * 12, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < 16 * 12
            
            # Original index calculation: 
            # reshape to [batch, 16, 12, seq_len] = [batch, 16, 12, 192]
            # permute to [batch, seq_len, 16, 12] = [batch, 192, 16, 12]
            
            # So output[b, s, i, j] corresponds to input[b, s, ...]
            # where the flat index i*12+j maps to the original hidden dimension
            
            # Load input
            x = tl.load(input_ptr + batch_idx * seq_len * hidden_size + seq_idx * hidden_size + offsets,
                       mask=mask, other=0.0)
            
            # Apply relu
            output = tl.where(x > 0, x, 0.0)
            
            # Store output [batch, seq_len, 16, 12]
            tl.store(output_ptr + batch_idx * seq_len * 16 * 12 + seq_idx * 16 * 12 + offsets,
                    output, mask=mask)


@torch.fx.wrap
def fused_reshape_permute_relu_kernel(input_tensor):
    """
    Fused kernel: reshape + permute + relu
    
    Args:
        input_tensor: [batch, seq_len, hidden] = [batch, 192, 1280]
    
    Returns:
        output: [batch, seq_len, 16, 12] = [batch, 192, 16, 12]
    """
    batch, seq_len, hidden = input_tensor.shape
    
    # Target shape after reshape+permute: [batch, seq_len, 16, 12]
    # where 16 * 12 = 192 = seq_len
    output_shape = (batch, seq_len, 16, 12)
    output = torch.empty(output_shape, device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Use appropriate block size
    BLOCK_SIZE = min(1024, triton.next_power_of_2(seq_len * 16 * 12))
    
    # Launch grid
    grid = (batch,)
    
    reshape_permute_relu_kernel[grid](
        input_tensor,
        output,
        batch_size=batch,
        seq_len=seq_len,
        hidden_size=hidden,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_reshape_permute_relu_kernel