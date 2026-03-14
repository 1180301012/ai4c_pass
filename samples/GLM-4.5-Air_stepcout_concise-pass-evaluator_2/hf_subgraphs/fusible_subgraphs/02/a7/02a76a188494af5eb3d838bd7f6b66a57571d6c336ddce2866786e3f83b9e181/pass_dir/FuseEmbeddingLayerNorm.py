import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline
import math

def pattern(input_ids, norm_weight, embedding_weight):
    # Mirror the exact operations from model.py
    tmp_3 = torch.nn.functional.embedding(input_ids, embedding_weight, 50283, None, 2.0, False, False)
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (768,), norm_weight, None, 1e-05)
    return tmp_4

def replacement_args(input_ids, norm_weight, embedding_weight):
    return (input_ids, norm_weight, embedding_weight)

@triton.jit
def fused_embedding_layer_norm_kernel(
    input_ids_ptr,
    embedding_weight_ptr,
    norm_weight_ptr,
    bias_ptr,
    output_ptr,
    input_ids_size,
    embedding_dim,
    vocab_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate the range of input IDs this program handles
    start_idx = pid * BLOCK_SIZE
    end_idx = min(start_idx + BLOCK_SIZE, input_ids_size)
    
    # Process each input ID in the block
    for i in range(start_idx, end_idx):
        input_id = tl.load(input_ids_ptr + i)
        
        # Clamp input ID to valid range
        input_id = tl.max(input_id, 0)
        input_id = tl.min(input_id, vocab_size - 1)
        
        # Load embedding vector
        embedding_offset = input_id * embedding_dim
        embedding_vec = tl.load(embedding_weight_ptr + embedding_offset, mask=(input_id >= 0) & (input_id < vocab_size))
        
        # Apply layer normalization
        # Calculate mean
        mean = tl.sum(embedding_vec) / embedding_dim
        
        # Calculate variance
        centered_vec = embedding_vec - mean
        variance = tl.sum(centered_vec * centered_vec) / embedding_dim
        
        # Normalize
        normalized_vec = centered_vec / tl.sqrt(variance + eps)
        
        # Apply affine transformation: weight * normalized_vec + bias
        if bias_ptr != 0:
            bias = tl.load(bias_ptr + tl.arange(0, embedding_dim))
            result = normalized_vec * norm_weight + bias
        else:
            result = normalized_vec * norm_weight
        
        # Store result
        output_offset = i * embedding_dim
        tl.store(output_ptr + output_offset, result)

@torch.fx.wrap
def fused_embedding_layer_norm(input_ids, norm_weight, embedding_weight, eps=1e-05):
    # Get input dimensions
    batch_size, seq_len = input_ids.shape
    
    # If bias is None, create a zero tensor
    if hasattr(torch, 'zeros_like'):
        bias = torch.zeros_like(norm_weight)
    else:
        bias = torch.zeros(norm_weight.shape, dtype=norm_weight.dtype, device=norm_weight.device)
    
    # Set up output tensor
    output = torch.empty((batch_size, seq_len, embedding_weight.shape[1]), 
                        dtype=torch.float32, device=input_ids.device)
    
    # Flatten input_ids for easier processing
    input_ids_flat = input_ids.flatten()
    input_size = input_ids_flat.numel()
    
    # Set Triton kernel configurations
    embedding_dim = embedding_weight.shape[1]
    vocab_size = embedding_weight.shape[0]
    
    # Choose block size based on input size
    if input_size < 1024:
        BLOCK_SIZE = 64
    elif input_size < 8192:
        BLOCK_SIZE = 256
    else:
        BLOCK_SIZE = 1024
    
    num_programs = (input_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch Triton kernel
    fused_embedding_layer_norm_kernel[(num_programs,)](
        input_ids_ptr=input_ids_flat,
        embedding_weight_ptr=embedding_weight,
        norm_weight_ptr=norm_weight,
        bias_ptr=bias,
        output_ptr=output,
        input_ids_size=input_size,
        embedding_dim=embedding_dim,
        vocab_size=vocab_size,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape output to match expected dimensions
    return output.view(batch_size, seq_len, embedding_dim)

def replacement_func():
    return fused_embedding_layer_norm