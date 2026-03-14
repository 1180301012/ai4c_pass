import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1):
    """
    Match the computation pattern:
    1. in_1 += in_0 (element-wise addition)
    2. tmp_1 = tmp_0.float() (redundant type conversion)
    3. tmp_2 = softmax(tmp_1, dim=-1) 
    4. tmp_3 = tmp_2.type_as(tmp_0) (redundant type conversion back)
    5. tmp_4 = dropout(tmp_3, p=0.1, training=False)
    """
    # Note: We match the operations but will optimize in the kernel
    # Since the original has in-place addition, we need to track this
    result_add = in_1 + in_0
    tmp_1 = result_add.float()
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = tmp_2.type_as(result_add)
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.1, training=False)
    return tmp_4

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized kernel with fused operations
@triton.jit
def fused_add_softmax_dropout_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_elements,
    dropout_p: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel that fuses:
    1. Element-wise addition: out = in_1 + in_0
    2. Softmax over last dimension
    3. Dropout
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    in_0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_1 = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    
    # Step 1: Element-wise addition
    add_result = in_1 + in_0
    
    # Step 2: Softmax with better numerical stability
    # Subtract max for numerical stability
    max_val = tl.max(add_result, axis=0)
    stabilized = add_result - max_val
    exp_stabilized = tl.exp(stabilized)
    sum_exp = tl.sum(exp_stabilized, axis=0)
    softmax_out = exp_stabilized / sum_exp
    
    # Step 3: Dropout during inference (just scaling, no masking)
    # For training=False, dropout just scales by (1-p)
    dropout_scale = 1.0 - dropout_p
    dropout_out = softmax_out * dropout_scale
    
    # Store result
    tl.store(out_ptr + offsets, dropout_out, mask=mask)

@torch.fx.wrap
def fused_add_softmax_dropout(in_0, in_1, dropout_p=0.1):
    """
    Optimized fused function that combines addition, softmax, and dropout.
    Handles different tensor shapes dynamically by flattening and reshaping.
    """
    # Ensure both tensors are on the same device
    if in_0.device != in_1.device:
        in_1 = in_1.to(in_0.device)
    
    # Get output shape and flatten for processing
    output_shape = in_0.shape
    original_ndim = len(output_shape)
    
    # Handle both 4D tensors: [batch, heads, seq_len, seq_len]
    # For softmax over last dimension, we need to handle the last dim specially
    if original_ndim == 4:
        # Reshape to [batch * heads * seq_len, seq_len] for softmax over last dim
        batch_size, n_heads, seq_len, _ = output_shape
        n_elements = batch_size * n_heads * seq_len * seq_len
        
        # Create output tensor
        out = torch.empty_like(in_0)
        
        # Choose appropriate block size based on tensor size
        if n_elements < 1024:
            BLOCK_SIZE = 128
        elif n_elements < 65536:
            BLOCK_SIZE = 256
        else:
            BLOCK_SIZE = 512
            
        num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Launch fused kernel
        fused_add_softmax_dropout_kernel[(num_programs,)](
            in_0_ptr=in_0,
            in_1_ptr=in_1,
            out_ptr=out,
            n_elements=n_elements,
            dropout_p=dropout_p,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return out
    else:
        # For other tensor shapes, use PyTorch operations (fallback)
        result = in_0 + in_1
        result = torch.nn.functional.softmax(result, dim=-1)
        result = torch.nn.functional.dropout(result, p=dropout_p, training=False)
        return result

# Replacement function (returns the function reference)
def replacement_func():
    return fused_add_softmax_dropout