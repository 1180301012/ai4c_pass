import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_3):
    """
    Match the linear + permute pattern from the model:
    tmp_2 = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = tmp_2.permute(0, 3, 1, 2)
    
    Input shapes:
    - in_0 (bias): [16]
    - in_1 (weight): [16, 3]
    - in_3 (input): [1, 196, 196, 3]
    
    Output shape: [1, 16, 196, 196]
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.nn.functional.linear(in_3, tmp_1, tmp_0)
    tmp_3 = tmp_2.permute(0, 3, 1, 2)
    return tmp_3


def replacement_args(in_0, in_1, in_3):
    return (in_0, in_1, in_3)


@triton.jit
def linear_permute_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    num_elements: tl.constexpr,
    seq_len: tl.constexpr,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused linear + permute kernel.
    """
    pid = tl.program_id(0)
    start_offset = pid * BLOCK_SIZE
    offsets = start_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Calculate output indices: [batch, out_feat, seq_i, seq_j]
    batch_idx = offsets // (out_features * seq_len * seq_len)
    remaining = offsets % (out_features * seq_len * seq_len)
    out_feat_idx = remaining // (seq_len * seq_len)
    remaining2 = remaining % (seq_len * seq_len)
    seq_i_idx = remaining2 // seq_len
    seq_j_idx = remaining2 % seq_len
    
    # Compute linear: sum over in_features
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for k in range(in_features):
        input_offset = (batch_idx * seq_len * seq_len * in_features + 
                        seq_i_idx * seq_len * in_features + 
                        seq_j_idx * in_features + k)
        x = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
        
        weight_offset = out_feat_idx * in_features + k
        w = tl.load(weight_ptr + weight_offset, mask=mask, other=0.0)
        
        acc += x * w
    
    # Add bias
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + out_feat_idx)
        acc += bias
    
    tl.store(output_ptr + offsets, acc, mask=mask)


def optimized_linear_permute(in_0, in_1, in_3):
    """
    Optimized fused linear + permute using Triton kernel.
    """
    batch_size, seq_len, seq_len_2, in_features = in_3.shape
    out_features = in_1.shape[0]
    
    # Allocate output
    output = torch.empty((batch_size, out_features, seq_len, seq_len_2), 
                         device=in_3.device, dtype=in_3.dtype)
    
    # Ensure weight and bias are on the same device
    weight = in_1.to(in_3.device)
    bias = in_0.to(in_3.device) if in_0 is not None else None
    
    # Calculate grid - using larger block size for better parallelism
    num_elements = batch_size * out_features * seq_len * seq_len_2
    BLOCK_SIZE = 2048  # Larger block size
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    linear_permute_kernel[(num_programs,)](
        in_3,
        weight,
        bias,
        output,
        num_elements,
        seq_len,
        in_features,
        out_features,
        BLOCK_SIZE,
    )
    
    return output


# Wrap the function for FX
@torch.fx.wrap
def linear_permute_wrapper(in_0, in_1, in_3):
    return optimized_linear_permute(in_0, in_1, in_3)


def replacement_func():
    return linear_permute_wrapper