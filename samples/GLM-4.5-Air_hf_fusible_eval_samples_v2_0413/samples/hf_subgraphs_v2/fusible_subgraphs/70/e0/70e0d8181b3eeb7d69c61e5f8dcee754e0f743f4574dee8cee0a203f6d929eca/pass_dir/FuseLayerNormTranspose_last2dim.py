import torch
import triton
import triton.language as tl

@triton.jit
def simple_layer_norm_kernel(
    x_ptr,  # Input tensor [1, 3999, 512]
    weight_ptr,  # LayerNorm weight [512]
    bias_ptr,    # LayerNorm bias [512] 
    out_ptr,     # Output tensor [1, 3999, 512] (no transpose, just LayerNorm)
    M,           # 3999 (sequence length)
    N,           # 512 (feature dimension) 
    eps,         # 1e-05
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Simple LayerNorm kernel - temporary implementation"""
    # Each program handles a block of sequence dimension
    m_offset = tl.program_id(0) * BLOCK_SIZE_M
    m_index = m_offset + tl.arange(0, BLOCK_SIZE_M)
    
    # Process block of feature dimensions
    n_offset = tl.arange(0, BLOCK_SIZE_N)
    n_index = 0  # Start at feature 0
    
    # Create mask
    mask = (m_index[:, None] < M) & (n_offset[None, :] < N)
    
    # Load input data [BLOCK_SIZE_M, BLOCK_SIZE_N]
    x_ptrs = x_ptr + m_index[:, None] * N + n_offset[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    
    # Calculate mean and variance for LayerNorm (must be done BEFORE using mean)
    mean = tl.sum(x, axis=1) / M
    x_centered = x - mean[:, None] 
    var = tl.sum(x_centered * x_centered, axis=1) / M
    
    # Load current block of weights and bias (moved after computation)
    weight = tl.load(weight_ptr + n_index + n_offset)
    bias = tl.load(bias_ptr + n_index + n_offset)
    
    # Normalize and apply weight/bias
    sqrt_var = tl.math.rsqrt(var + eps)
    x_norm = x_centered * sqrt_var[:, None]
    out = x_norm * weight[None, :] + bias[None, :]
    
    # Store result (same shape as input)
    out_ptrs = out_ptr + m_index[:, None] * N + n_offset[None, :]
    tl.store(out_ptrs, out, mask=mask)

@torch.fx.wrap
def fused_layer_norm_transpose(x, weight, bias):
    """Fused LayerNorm + Transpose operation 
    (temporary: separate LayerNorm + transpose for testing)"""
    M = x.size(1)  # 3999
    N = x.size(2)  # 512
    
    # Get device for memory allocation
    device = x.device
    
    # Step 1: Apply LayerNorm using Triton kernel
    out_layer_norm = torch.empty_like(x)
    
    # Configure block sizes for optimal GPU occupancy
    BLOCK_SIZE_M = 256  # Process 256 sequence positions at once  
    BLOCK_SIZE_N = 512  # Process full feature dimension
    
    # Calculate grid dimensions
    grid_size_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    # Launch LayerNorm kernel
    simple_layer_norm_kernel[(grid_size_m,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out_layer_norm,
        M=M,
        N=N,
        eps=1e-05,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    # Step 2: Apply transpose using Triton kernel (move transpose into kernel)
    # Output should be [1, 512, 3999] (transposed input shape)
    out = torch.empty((x.size(0), N, M), dtype=x.dtype, device=x.device)
    
    # Launch final transpose kernel - one program per sequence position
    transpose_kernel[(M,)](
        x_ptr=out_layer_norm,
        out_ptr=out,
        M=M,
        BLOCK_SIZE_M=1,  # Not used in this kernel, but required
        N_FEATURES=512,  # Compile-time constant known for this problem
    )
    
    return out

@triton.jit
def transpose_kernel(
    x_ptr,      # Input [1, M, N]
    out_ptr,    # Output [1, N, M] 
    M,          # Sequence length
    BLOCK_SIZE_M: tl.constexpr,
    N_FEATURES: tl.constexpr,  # 512 (compile-time constant)
):
    """Simple transpose kernel - process one sequence position at a time"""
    # Each program handles one sequence position
    m_pos = tl.program_id(0)  # Single sequence position
    
    # Process all features
    n_offset = tl.arange(0, N_FEATURES)
    
    mask = m_pos < M  # Only process within sequence bounds
    
    # Load input sequence: [N_FEATURES] for this position
    x_ptrs = x_ptr + m_pos * N_FEATURES + n_offset
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    
    # Store to transposed position: put at column m_pos, row n_offset
    out_ptrs = out_ptr + n_offset * M + m_pos
    tl.store(out_ptrs, x, mask=mask)

def pattern(x, weight, bias):
    """Match LayerNorm → Transpose sequence"""
    # LayerNorm (normalized over last dimension, size hardcoded as 512)
    tmp_2 = torch.nn.functional.layer_norm(x, (512,), weight, bias, 1e-05)
    # Transpose last two dimensions
    tmp_3 = tmp_2.transpose(-2, -1)
    return tmp_3

def replacement_args(x, weight, bias):
    """Extract arguments for fused operation"""
    return (x, weight, bias)

def replacement_func():
    """Return fused kernel function"""
    return fused_layer_norm_transpose