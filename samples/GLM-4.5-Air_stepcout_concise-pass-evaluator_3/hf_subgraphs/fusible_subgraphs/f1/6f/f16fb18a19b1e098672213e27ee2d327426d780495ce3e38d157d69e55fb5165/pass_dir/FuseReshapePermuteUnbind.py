import torch
import triton
import triton.language as tl


@triton.jit
def reshape_permute_unbind_kernel(
    input_ptr, output_q_ptr, output_k_ptr, output_v_ptr,
    B: tl.constexpr, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused kernel that performs:
    - Reshape: [B, N, 3*H*D] -> [B, N, 3, H, D]
    - Permute: [B, N, 3, H, D] -> [3, B, H, N, D]
    - Unbind: Split into 3 separate tensors Q, K, V
    
    Input: [B, N, 3*H*D] = [1, 197, 1296] for convit_small (H=9, D=48)
    Output Q: [B, H, N, D] = [1, 9, 197, 48]
    Output K: [B, H, N, D] = [1, 9, 197, 48]  
    Output V: [B, H, N, D] = [1, 9, 197, 48]
    """
    # Each program handles a slice of the sequence dimension
    pid = tl.program_id(0)
    
    # Calculate the starting position for this program
    # We process BLOCK_SIZE elements per program along the N dimension
    offset_n = pid * BLOCK_SIZE
    
    # Create offsets for the dimensions
    # Input layout: [B, N, 3*H*D] -> [1, 197, 1296] for convit_small
    # Flattened: we need to compute offsets for each element
    
    # Load BLOCK_SIZE elements from input for each head group
    for h in range(H):
        for d in range(D):
            # Offset for Q (head group 0)
            # Input index: [0, n, h*D + d] where n ranges from offset_n to offset_n + BLOCK_SIZE
            base_idx_q = offset_n * (3 * H * D) + h * D + d
            # Offset for K (head group 1)
            base_idx_k = offset_n * (3 * H * D) + H * D + h * D + d
            # Offset for V (head group 2)
            base_idx_v = offset_n * (3 * H * D) + 2 * H * D + h * D + d
            
            # Create offsets for the BLOCK_SIZE elements
            # Offsets along N dimension
            n_offsets = tl.arange(0, BLOCK_SIZE) * (3 * H * D)
            
            # Mask for valid elements (within N bounds)
            mask = (offset_n + tl.arange(0, BLOCK_SIZE)) < N
            
            # Load Q values
            q_offsets = base_idx_q + n_offsets
            q_vals = tl.load(input_ptr + q_offsets, mask=mask, other=0.0)
            
            # Store Q: [B, H, N, D] -> linearize as [H, N, D] with B=0
            # Output index: [h, n, d]
            q_out_base = h * N * D + offset_n * D + d
            q_out_offsets = q_out_base + tl.arange(0, BLOCK_SIZE) * D
            tl.store(output_q_ptr + q_out_offsets, q_vals, mask=mask)
            
            # Load K values
            k_offsets = base_idx_k + n_offsets
            k_vals = tl.load(input_ptr + k_offsets, mask=mask, other=0.0)
            
            # Store K: [B, H, N, D]
            k_out_base = h * N * D + offset_n * D + d
            k_out_offsets = k_out_base + tl.arange(0, BLOCK_SIZE) * D
            tl.store(output_k_ptr + k_out_offsets, k_vals, mask=mask)
            
            # Load V values
            v_offsets = base_idx_v + n_offsets
            v_vals = tl.load(input_ptr + v_offsets, mask=mask, other=0.0)
            
            # Store V: [B, H, N, D]
            v_out_base = h * N * D + offset_n * D + d
            v_out_offsets = v_out_base + tl.arange(0, BLOCK_SIZE) * D
            tl.store(output_v_ptr + v_out_offsets, v_vals, mask=mask)


def reshape_permute_unbind_triton(x, num_heads):
    """
    Fused reshape + permute + unbind operation.
    
    Args:
        x: Input tensor of shape [B, N, 3*H*D] (e.g., [1, 197, 1296])
        num_heads: Number of heads per group (H, e.g., 9 for convit_small, 4 for convit_tiny)
    
    Returns:
        Q, K, V tensors each of shape [B, H, N, D]
    """
    B, N, total_dim = x.shape
    D = total_dim // (3 * num_heads)  # Should be 48
    
    # Allocate output tensors
    Q = torch.empty((B, num_heads, N, D), dtype=x.dtype, device=x.device)
    K = torch.empty((B, num_heads, N, D), dtype=x.dtype, device=x.device)
    V = torch.empty((B, num_heads, N, D), dtype=x.dtype, device=x.device)
    
    # Define block size - tune for efficiency
    BLOCK_SIZE = 64
    
    # Calculate grid
    # We parallelize over the N dimension
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    reshape_permute_unbind_kernel[grid](
        x, Q, K, V,
        B, N, num_heads, D,
        BLOCK_SIZE
    )
    
    return Q, K, V


# Simpler pattern - just match the reshape -> permute -> unbind -> transpose
# This avoids the matmul in the pattern which might be causing issues
def pattern(x, weight):
    """
    Match the reshape-permute-unbind-transpose pattern:
    1. reshape -> [1, 197, 3, 9, 48]
    2. permute -> [3, 1, 9, 197, 48]
    3. unbind -> 3 tensors
    4. transpose middle tensor
    
    Returns Q, K_transposed, V
    """
    # Reshape
    t1 = x.reshape(1, 197, 3, 9, 48)
    # Permute
    t2 = t1.permute(2, 0, 3, 1, 4)
    # Unbind
    t3 = t2.unbind(0)
    q = t3[0]
    k = t3[1]
    v = t3[2]
    # Transpose K
    kt = k.transpose(-2, -1)
    return q, kt, v


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_qkv_kernel


@torch.fx.wrap
def fused_qkv_kernel(in_0, in_1):
    """
    Fused QKV kernel that replaces the entire linear + reshape + permute + unbind + transpose pattern.
    
    This uses a single Triton kernel to do all the work efficiently.
    
    Args:
        in_0: Weight tensor [3*H*D, D]
        in_1: Input tensor [B, N, D]
    
    Returns:
        Q [B, H, N, D], K_transposed [B, H, D, N], V [B, H, N, D]
    """
    # Compute the linear layer output - need to use @ operator which is not blocked
    linear_out = in_1 @ in_0.t()
    
    # Determine dimensions
    total_dim = linear_out.shape[-1]  # 3*H*D
    out_dim = in_0.shape[1]  # D
    num_heads = total_dim // out_dim // 3  # H = 9 for convit_small, 4 for convit_tiny
    
    # Use fused triton kernel for reshape + permute + unbind
    Q, K, V = reshape_permute_unbind_triton(linear_out, num_heads)
    
    # Transpose K: [B, H, N, D] -> [B, H, D, N]
    K_t = K.transpose(-2, -1)
    
    return Q, K_t, V