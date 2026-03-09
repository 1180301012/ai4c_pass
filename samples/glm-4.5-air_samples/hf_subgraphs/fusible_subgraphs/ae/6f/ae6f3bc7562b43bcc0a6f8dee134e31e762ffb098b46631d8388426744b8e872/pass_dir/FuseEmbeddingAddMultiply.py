import torch
import triton
import triton.language as tl

def pattern(input_ids, weight, add_tensor, multiply_tensor):
    # Pattern: embedding lookup + addition + multiplication
    tmp_6 = torch.nn.functional.embedding(input_ids, weight, 1, None, 2.0, False, False)
    tmp_7 = add_tensor + tmp_6
    tmp_8 = multiply_tensor.unsqueeze(-1)
    tmp_9 = tmp_7 * tmp_8
    return tmp_9

def replacement_args(input_ids, weight, add_tensor, multiply_tensor):
    return (input_ids, weight, add_tensor, multiply_tensor)

@triton.jit
def fused_embedding_add_multiply_kernel(
    input_ids_ptr,
    weight_ptr,
    add_tensor_ptr,
    multiply_tensor_ptr,
    out_ptr,
    input_ids_dim0,
    input_ids_dim1,
    feat_dim,
    weight_vocab_size,
    weight_feat_dim,
    BLOCK_SIZE_M: tl.constexpr,
):
    # Matrix dimensions
    M = input_ids_dim0 * input_ids_dim1
    N = feat_dim
    
    # Program ID
    pid_m = tl.program_id(0)
    
    # Process multiple elements per program
    start_m = pid_m * BLOCK_SIZE_M
    offs_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < M
    
    # Process entire feature dimension at once
    for offs_n in range(0, N, 128):  # Process in chunks of 128 features
        end_n = min(offs_n + 128, N)
        chunk_size = end_n - offs_n
        
        # Create offset for this chunk
        offs_n_chunk = tl.arange(0, chunk_size)
        
        # Load input_ids
        input_vals = tl.load(input_ids_ptr + offs_m, mask=mask_m, other=0).to(tl.int32)
        
        # Load weight for this chunk
        weight_offs = input_vals * weight_feat_dim + (offs_n + offs_n_chunk)
        weight_vals = tl.load(weight_ptr + weight_offs, 
                             mask=mask_m[:, None] & (offs_n_chunk[None, :] < chunk_size), 
                             other=0.0)
        
        # Load add tensor (broadcast over features)
        add_offs = offs_m * feat_dim + offs_n
        add_vals = tl.load(add_tensor_ptr + add_offs, mask=mask_m, other=0.0)
        add_vals = add_vals[:, None] + tl.zeros([BLOCK_SIZE_M, chunk_size], dtype=tl.float32)
        
        # Load multiply tensor (unsqueeze over features)
        multiply_vals = tl.load(multiply_tensor_ptr + offs_m, mask=mask_m, other=1.0)
        multiply_expanded = multiply_vals[:, None]
        
        # Computation: embedding + add + multiply
        result = (weight_vals + add_vals) * multiply_expanded
        
        # Store result for this chunk
        out_offs = offs_m[:, None] * feat_dim + offs_n + offs_n_chunk
        tl.store(out_ptr + out_offs, result, 
                mask=mask_m[:, None] & (offs_n_chunk[None, :] < chunk_size))

@torch.fx.wrap
def fused_embedding_add_multiply(input_ids, weight, add_tensor, multiply_tensor):
    # Get tensor shapes
    input_ids_dim0, input_ids_dim1 = input_ids.shape
    feat_dim = weight.shape[1]
    weight_vocab_size, weight_feat_dim = weight.shape
    
    # Reshape input_ids to 2D for processing
    input_ids_2d = input_ids.reshape(-1)
    
    # Reshape add_tensor to be compatible
    if len(add_tensor.shape) == 2 and add_tensor.shape == (input_ids_dim0 * input_ids_dim1, feat_dim):
        # Already correct shape
        add_tensor_2d = add_tensor
    else:
        # Reshape to 2D (batch x features)
        if len(add_tensor.shape) == 3:
            add_tensor_2d = add_tensor.reshape(-1, feat_dim)
        else:
            # Handle broadcasting - make it compatible
            add_tensor_2d = add_tensor.reshape(-1, 1)  # Will be broadcasted
    
    # Output shape - should be 3D
    out_shape = (input_ids_dim0, input_ids_dim1, feat_dim)
    out = torch.empty(out_shape, dtype=torch.float32, device=weight.device)
    
    # Block size selection
    BLOCK_SIZE_M = 64  # Number of input IDs to process per program
    
    # Number of programs (one per block of input IDs)
    total_input_ids = input_ids_dim0 * input_ids_dim1
    num_programs = (total_input_ids + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    # Launch kernel
    fused_embedding_add_multiply_kernel[(num_programs,)](
        input_ids_ptr=input_ids_2d,
        weight_ptr=weight,
        add_tensor_ptr=add_tensor_2d,
        multiply_tensor_ptr=multiply_tensor,
        out_ptr=out,
        input_ids_dim0=input_ids_dim0,
        input_ids_dim1=input_ids_dim1,
        feat_dim=feat_dim,
        weight_vocab_size=weight_vocab_size,
        weight_feat_dim=weight_feat_dim,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )
    
    return out

def replacement_func():
    return fused_embedding_add_multiply