import torch
import triton
import triton.language as tl

# Pattern matching function
# Matches exactly: torch.conv2d(in_2, in_0, None, (1, 1), (32, 0), (1, 1), 4)
def pattern(in_0, in_2):
    result = torch.conv2d(in_2, in_0, None, (1, 1), (32, 0), (1, 1), 4)
    return result

# Argument extraction function
# Extracts input tensors for the kernel
# The 'in_0' and 'in_2' refer to the model's input names
# in_0: weight tensor (conv kernel)
# in_2: input tensor (value_layer)
def replacement_args(in_0, in_2):
    return (in_0, in_2)

# Triton kernel for custom depthwise convolution
@triton.jit
def depthwise_conv2d_kernel(
    input_ptr, 
    weight_ptr, 
    output_ptr,
    batch, 
    groups, 
    seq_len, 
    out_seq_len, 
    head_dim,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_HEAD: tl.constexpr
):
    # Block indices
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    output_seq_block = tl.program_id(2)
    
    # Calculate start/end of this block in output sequence
    output_seq_start = output_seq_block * BLOCK_SEQ
    output_seq_end = min(output_seq_start + BLOCK_SEQ, out_seq_len)
    
    # Load weight for current group (65 weights)
    weight = tl.load(weight_ptr + group_idx * 65, shape=(65,), 
                     mask=tl.arange(0, 65) < 65, 
                     other=0.0)

    # Process each output sequence position in this block
    for output_seq in range(output_seq_start, output_seq_end):
        # Input sequence position range: [32 + output_seq, 32 + output_seq + 64]
        # Check for valid positions (before padding and beyond input sequence)
        acc = tl.zeros((BLOCK_HEAD,), dtype=tl.float32)
        for k in range(65):
            input_idx = output_seq + k  # 32 + output_seq + k - 32
            # Handle padding (treat out-of-bound as 0)
            if input_idx >= 0 and input_idx < seq_len:
                input_data = tl.load(
                    input_ptr + batch_idx * groups * seq_len * head_dim + 
                    group_idx * seq_len * head_dim + 
                    input_idx * head_dim,
                    shape=(BLOCK_HEAD,),
                    mask=tl.arange(0, BLOCK_HEAD) < head_dim
                )
                acc += input_data * weight[k]
        # Store result
        tl.store(
            output_ptr + batch_idx * groups * out_seq_len * head_dim + 
            group_idx * out_seq_len * head_dim + 
            output_seq * head_dim,
            acc,
            mask=tl.arange(0, BLOCK_HEAD) < head_dim
        )

# Kernel wrapper for Triton
@torch.fx.wrap
def depthwise_conv2d(in_0, in_2):
    # Get tensor shapes
    batch, groups, seq_len, head_dim = in_2.shape
    # Calculate output sequence length: 64 + 32 + 0 - 65 + 1 = 32
    out_seq_len = seq_len + 32 - 65 + 1  # seq_len - 32
    
    # Reshape weight to [groups, 65] (squeeze out 1x1 dimensions)
    weight_reshaped = in_0.squeeze(1).squeeze(3)
    
    # Create output tensor
    output = torch.empty(
        (batch, groups, out_seq_len, head_dim),
        dtype=in_2.dtype,
        device=in_2.device
    )
    
    # Configurable block sizes
    BLOCK_SEQ = 32  # Sequence elements per thread block
    BLOCK_HEAD = 8  # Head dimensions per thread
    
    # Calculate grid dimensions
    grid = (
        batch,  # Batch dimension
        groups,  # Groups dimension
        (out_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ  # Sequence blocks
    )
    
    # Launch kernel
    depthwise_conv2d_kernel[grid](
        in_2, weight_reshaped, output,
        batch, groups, seq_len, out_seq_len, head_dim,
        BLOCK_SEQ, BLOCK_HEAD
    )
    
    return output

# Replacement function (return the kernel wrapper)
def replacement_func():
    return depthwise_conv2d