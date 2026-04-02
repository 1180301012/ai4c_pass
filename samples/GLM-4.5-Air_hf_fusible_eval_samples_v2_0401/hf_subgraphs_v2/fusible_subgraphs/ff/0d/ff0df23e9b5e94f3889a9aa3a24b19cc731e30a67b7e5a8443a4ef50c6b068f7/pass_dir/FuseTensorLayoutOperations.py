import torch
import triton
import triton.language as tl

from torch import device

# Pattern matching function - exact sequence from model.py
def pattern(input_tensor, embed_dim, batch_size, seq_len, H, W):
    """Match the sequence: permute -> unsqueeze -> expand -> contiguous"""
    tmp_1 = input_tensor
    tmp_2 = None  # This would be the embedding output
    tmp_3 = tmp_1.permute([2, 0, 1])  # [embed_dim, batch_size, seq_len]
    tmp_1 = None
    tmp_4 = tmp_3.unsqueeze(0)  # [1, embed_dim, batch_size, seq_len]
    tmp_3 = None
    tmp_5 = tmp_4.expand((1, -1, H, W))  # [1, embed_dim, H, W]
    tmp_4 = None
    tmp_6 = tmp_5.contiguous()  # [1, embed_dim, H, W]
    tmp_5 = None
    return tmp_6

# Argument extraction function
def replacement_args(input_tensor, embed_dim, batch_size, seq_len, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    return (input_tensor, embed_dim, batch_size, seq_len)

# Optimized Triton kernel for tensor layout fusion
@triton.jit
def tensor_layout_fusion_kernel(
    input_ptr,
    output_ptr,
    embed_dim,
    batch_size,
    seq_len,
    H,
    W,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """High-performance kernel that fuses permute, unsqueeze, expand, and contiguous"""
    # Grid setup: each program handles one output position in the final tensor
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Output coordinates: [1, embed_dim, H, W]
    # pid_m corresponds to embed_dim, pid_n corresponds to H * W
    embed_idx = pid_m
    output_linear_idx = pid_n
    
    # Convert linear output index to H, W coordinates
    W_idx = output_linear_idx % W
    H_idx = output_linear_idx // W
    
    # Check bounds
    if embed_idx >= embed_dim or H_idx >= H or W_idx >= W:
        return
    
    # Calculate input tensor dimensions after operations:
    # Before: [seq_len, batch_size, embed_dim]
    # After permute [2,0,1]: [embed_dim, batch_size, seq_len]
    # After unsqueeze(0): [1, embed_dim, batch_size, seq_len]
    # After expand: [1, embed_dim, H, W]
    
    # Mapping output to input coordinates:
    # Output [1, embed_idx, H_idx, W_idx] comes from input [embed_idx, 0, src_seq_idx]
    # where src_seq_idx maps to W_idx through interpolation/expansion
    
    # For expand operation: src_seq_idx = min(W_idx * seq_len // W, seq_len - 1)
    src_seq_idx = W_idx * seq_len // W if W > 1 else 0
    src_seq_idx = min(src_seq_idx, seq_len - 1)
    
    # Load input element: input[embed_idx, 0, src_seq_idx]
    input_offset = (src_seq_idx * batch_size + 0) * embed_dim + embed_idx
    output_offset = ((H_idx * W + W_idx) * embed_dim) + embed_idx
    
    # Load input value
    input_val = tl.load(input_ptr + input_offset)
    
    # Store output value
    tl.store(output_ptr + output_offset, input_val)

# Kernel wrapper
@torch.fx.wrap
def fused_tensor_layout_operations(embedding_output, H, W):
    """Optimized fused tensor layout operations"""
    # Get input tensor shape
    seq_len, batch_size, embed_dim = embedding_output.shape
    
    # Initialize output tensor
    output = torch.empty(1, embed_dim, H, W, 
                        dtype=embedding_output.dtype, 
                        device=embedding_output.device)
    
    # Grid dimensions: embed_dim x (H * W)
    grid_m = embed_dim
    grid_n = H * W
    
    # Choose block sizes for better memory coalescing
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    
    # Launch kernel
    tensor_layout_fusion_kernel[(grid_m, grid_n)](
        input_ptr=embedding_output,
        output_ptr=output,
        embed_dim=embed_dim,
        batch_size=batch_size,
        seq_len=seq_len,
        H=H,
        W=W,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output

# Replacement function (must return function reference)
def replacement_func():
    return fused_tensor_layout_operations