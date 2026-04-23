import torch
import triton
import triton.language as tl

# Pattern matching function
# Matches the entire computation sequence from embedding through contiguous
# (including the redundant expand, which we'll handle in optimization)
def pattern(in_0, in_1):
    tmp_1 = in_1.to(device='cuda', index=0)
    tmp_2 = torch.nn.functional.embedding(tmp_1, in_0, None, None, 2.0, False, False)
    tmp_3 = tmp_2.permute([2, 0, 1])
    tmp_4 = tmp_3.unsqueeze(0)
    tmp_5 = tmp_4.expand((1, -1, 45, 45))
    tmp_6 = tmp_5.contiguous()
    return tmp_6

# Argument extraction
# Returns the inputs needed for the optimized kernel

def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Triton embedding kernel optimized for permuted output
@triton.jit
def embedding_kernel(
    weight_ptr,  
    indices_ptr,  
    output_ptr,  
    num_embeddings,  
    embedding_dim,  
    H,  
    W,  
    BLOCK_SIZE_H: tl.constexpr,  
    BLOCK_SIZE_W: tl.constexpr,
):
    # Calculate grid indices
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    
    # Calculate block offsets
    block_start_h = pid_h * BLOCK_SIZE_H
    block_start_w = pid_w * BLOCK_SIZE_W
    
    # Create thread offsets within block
    offs_h = tl.arange(0, BLOCK_SIZE_H)
    offs_w = tl.arange(0, BLOCK_SIZE_W)
    
    # Mask for valid indices in H and W dimensions
    valid_h = (block_start_h + offs_h) < H
    valid_w = (block_start_w + offs_w) < W
    
    # Load indices for current block
    indices = tl.load(
        indices_ptr + (block_start_h + offs_h)[:, None] * W + (block_start_w + offs_w)[None, :],
        mask=valid_h[:, None] & valid_w[None, :],
        other=0
    )
    
    # Load weight rows corresponding to indices
    weights = tl.load(
        weight_ptr + indices[:, :, None] * embedding_dim + tl.arange(0, embedding_dim),
        mask=indices[:, :, None] < num_embeddings,
        other=0.0
    )
    
    # Calculate output pointer for current block
    out_ptr = output_ptr + (block_start_h + offs_h)[:, None] * W * embedding_dim + (block_start_w + offs_w)[None, :] * embedding_dim
    
    # Store results
    tl.store(
        out_ptr,
        weights,
        mask=valid_h[:, None] & valid_w[None, :] & (indices[:, :, None] < num_embeddings)
    )

# Kernel wrapper with correct tensor shapes
@torch.fx.wrap
def optimized_embedding(in_0, in_1):
    # Ensure inputs are on correct device
    if in_1.device.type != 'cuda':
        in_1 = in_1.to('cuda')
    
    # Get tensor dimensions
    H, W = in_1.shape
    num_embeddings, embedding_dim = in_0.shape
    
    # Allocate output: [1, embedding_dim, H, W]
    output = torch.empty((1, embedding_dim, H, W), dtype=in_0.dtype, device='cuda')
    
    # Calculate grid dimensions (32x32 blocks)
    grid_h = (H + 31) // 32
    grid_w = (W + 31) // 32
    
    # Launch kernel
    embedding_kernel[(grid_h, grid_w), 32, 32](
        weight_ptr=in_0,
        indices_ptr=in_1,
        output_ptr=output,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        H=H,
        W=W,
        BLOCK_SIZE_H=32,
        BLOCK_SIZE_W=32,
    )
    
    return output

# Replacement function (returns kernel wrapper)
def replacement_func():
    return optimized_embedding