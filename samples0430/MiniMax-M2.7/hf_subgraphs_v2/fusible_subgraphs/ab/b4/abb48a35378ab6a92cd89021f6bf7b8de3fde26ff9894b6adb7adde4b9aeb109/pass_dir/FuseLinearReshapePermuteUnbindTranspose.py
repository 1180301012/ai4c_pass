import torch
import triton
import triton.language as tl


@triton.jit
def fused_qkv_kernel(
    # Input pointers
    x_ptr,         # (B, N, Cin) - [1, 197, 192/432/768]
    weight_ptr,    # (Cout, Cin) - [576/1296/2304, 192/432/768]
    # Output pointers
    q_out_ptr,     # (B, H, N, D) - [1, 4/9/16, 197, 48]
    k_out_ptr,     # (B, H, D, N) - [1, 4/9/16, 48, 197]
    v_out_ptr,     # (B, H, N, D) - [1, 4/9/16, 197, 48]
    # Shape info
    B: tl.constexpr,
    N: tl.constexpr,
    Cin: tl.constexpr,
    Cout: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    # Tile sizes
    BLOCK_SIZE: tl.constexpr,
    TILE_N: tl.constexpr,
):
    """
    Fused kernel for: linear + reshape + permute + unbind + transpose
    
    Computes: Q, K^T, V from x and weights
    - Linear: out = x @ weight.T  -> (B, N, Cout)
    - Reshape: -> (B, N, 3, H, D)
    - Permute: -> (3, B, H, N, D)
    - Unbind: Q=(B,H,N,D), K=(B,H,N,D), V=(B,H,N,D)
    - Transpose: K^T = (B,H,D,N)
    """
    # Get program IDs for 2D grid
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    # Offset calculations for the head
    # weight offset for this head (Cout = 3 * H * D)
    head_offset = pid_h * D
    # weight base pointer for this head (each head has 3*D rows, but we need all 3 for Q,K,V)
    
    # Create offset arrays
    n_offsets = tl.arange(0, TILE_N)  # N dimension
    d_offsets = tl.arange(0, BLOCK_SIZE)  # D dimension (tile size for inner dim)
    
    # For linear: we need to compute x @ weight.T
    # x[b, n, cin] * weight[cout, cin] -> sum over cin
    # Output: out[b, n, cout] where cout = 3*H*D
    
    # We tile over N and D dimensions
    # For each (n_tile, d_tile) in output
    for n_start in range(0, N, TILE_N):
        # Compute output for this tile
        n_offsets = n_start + tl.arange(0, TILE_N)
        n_mask = n_offsets < N
        
        # Initialize accumulators for Q, K, V (3 heads each)
        acc_q = tl.zeros((TILE_N, BLOCK_SIZE), dtype=tl.float32)
        acc_k = tl.zeros((TILE_N, BLOCK_SIZE), dtype=tl.float32)
        acc_v = tl.zeros((TILE_N, BLOCK_SIZE), dtype=tl.float32)
        
        # Accumulate over Cin
        for cin_start in range(0, Cin, BLOCK_SIZE):
            cin_offsets = cin_start + tl.arange(0, BLOCK_SIZE)
            cin_mask = cin_offsets < Cin
            
            # Load x: (B, N, Cin)
            x_ptrs = x_ptr + pid_b * N * Cin + n_offsets[:, None] * Cin + cin_offsets[None, :]
            x = tl.load(x_ptrs, mask=cin_mask[None, :] & n_mask[:, None], other=0.0)
            
            # Load weights for Q, K, V: each is (H*D, Cin) per head
            # Weight layout: [Q_head0, K_head0, V_head0, Q_head1, K_head1, V_head1, ...]
            for sub_head in range(3):
                w_row_offset = (sub_head * H + pid_h) * D
                
                # Weight pointers: weight[cout_offset + d, cin]
                w_ptrs = weight_ptr + (w_row_offset + d_offsets[None, :]) * Cin + cin_offsets[:, None]
                w = tl.load(w_ptrs, mask=cin_mask[:, None] & (d_offsets[None, :] < D), other=0.0)
                
                # Multiply and accumulate
                if sub_head == 0:
                    acc_q += tl.dot(x, w)
                elif sub_head == 1:
                    acc_k += tl.dot(x, w)
                else:
                    acc_v += tl.dot(x, w)
        
        # Normalize by 1/sqrt(D) for numerical stability
        scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))
        acc_q = acc_q * scale
        acc_k = acc_k * scale
        acc_v = acc_v * scale
        
        # Store outputs
        # Q: (B, H, N, D)
        q_offset = ((pid_b * H + pid_h) * N + n_offsets[:, None]) * D + d_offsets[None, :]
        tl.store(q_out_ptr + q_offset, acc_q, mask=n_mask[:, None] & (d_offsets[None, :] < D))
        
        # K: (B, H, D, N) - transposed
        k_offset = ((pid_b * H + pid_h) * D + d_offsets[:, None]) * N + n_offsets[None, :]
        tl.store(k_out_ptr + k_offset, acc_k, mask=(d_offsets[:, None] < D) & n_mask[None, :])
        
        # V: (B, H, N, D)
        v_offset = ((pid_b * H + pid_h) * N + n_offsets[:, None]) * D + d_offsets[None, :]
        tl.store(v_out_ptr + v_offset, acc_v, mask=n_mask[:, None] & (d_offsets[None, :] < D))


@torch.fx.wrap
def fused_qkv_wrapper(x, weight, B, N, Cin, Cout, H, D):
    """
    Wrapper for the fused QKV computation.
    
    Args:
        x: Input tensor (B, N, Cin)
        weight: Weight tensor (Cout, Cin) where Cout = 3 * H * D
        B: Batch size
        N: Sequence length
        Cin: Input channels
        Cout: Output channels (3 * H * D)
        H: Number of heads
        D: Head dimension
    
    Returns:
        Q: (B, H, N, D)
        K_T: (B, H, D, N) - transposed
        V: (B, H, N, D)
    """
    # Allocate output tensors
    Q = torch.empty((B, H, N, D), device=x.device, dtype=x.dtype)
    K_T = torch.empty((B, H, D, N), device=x.device, dtype=x.dtype)
    V = torch.empty((B, H, N, D), device=x.device, dtype=x.dtype)
    
    # Define tile sizes based on problem size
    TILE_N = min(64, N)
    BLOCK_SIZE = min(64, D)
    
    # Calculate grid dimensions
    grid = (B, H)
    
    fused_qkv_kernel[grid](
        x, weight, Q, K_T, V,
        B, N, Cin, Cout, H, D,
        BLOCK_SIZE=BLOCK_SIZE,
        TILE_N=TILE_N,
    )
    
    return Q, K_T, V


def pattern(in_0, in_1):
    """
    Match the pattern: linear + reshape + permute + unbind + transpose
    This version matches convit_small: reshape(1, 197, 3, 9, 48), H=9, D=48
    
    Returns Q, K^T, V tensors.
    """
    tmp_1 = torch.nn.functional.linear(in_1, in_0, None)
    tmp_2 = tmp_1.reshape(1, 197, 3, 9, 48)
    tmp_3 = tmp_2.permute(2, 0, 3, 1, 4)
    tmp_4 = tmp_3.unbind(0)
    tmp_5 = tmp_4[0]
    tmp_6 = tmp_4[1]
    tmp_7 = tmp_4[2]
    tmp_8 = tmp_6.transpose(-2, -1)
    
    return tmp_5, tmp_8, tmp_7


def replacement_args(in_0, in_1):
    """
    Extract arguments needed for the replacement function.
    in_0: weight tensor [Cout, Cin]
    in_1: input tensor [B, N, Cin]
    """
    return (in_0, in_1)


def replacement_func():
    return fused_qkv_dispatcher


# Dispatcher that handles different shapes
@torch.fx.wrap  
def fused_qkv_dispatcher(in_0, in_1):
    """
    Dispatcher that determines shapes and calls the appropriate kernel.
    
    Original pattern produces:
    - tmp_5: (1, 197, 9, 48) - this is (B, N, H, D) permuted -> (B, H, N, D)
    - tmp_8: (1, 9, 48, 197) - this is (B, H, D, N) - K transposed
    - tmp_7: (1, 197, 9, 48) - this is (B, N, H, D) permuted -> (B, H, N, D)
    
    So outputs should be: Q=(B, H, N, D), K^T=(B, H, D, N), V=(B, H, N, D)
    """
    weight = in_0  # [Cout, Cin]
    x = in_1       # [B, N, Cin]
    
    B, N, Cin = x.shape
    Cout = weight.shape[0]
    
    # Cout = 3 * H * D
    # Infer H and D from shapes
    third = Cout // 3
    
    # Try to infer D from known patterns
    if third == 192:  # convit_tiny: 4 * 48
        H = 4
        D = 48
    elif third == 432:  # convit_small: 9 * 48
        H = 9
        D = 48
    elif third == 768:  # convit_base: 16 * 48
        H = 16
        D = 48
    else:
        # Generic: assume H = sqrt(third / 3), D = third / H
        # Or use D=48 as default
        H = max(1, third // 48)
        D = 48
        if H * D != third:
            H = max(1, int(third ** 0.5))
            D = third // H
    
    return fused_qkv_wrapper(x, weight, B, N, Cin, Cout, H, D)