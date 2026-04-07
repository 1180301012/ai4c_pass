import torch
import triton
import triton.language as tl

def pattern(x):
    tmp_2 = x[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))]
    tmp_3 = tmp_2.transpose(-1, -2)
    tmp_4 = tmp_3.reshape(1, 152, 7, 7)
    return tmp_4

def replacement_args(x):
    return (x,)

@triton.jit
def slice_transpose_reshape_kernel(
    in_ptr,
    out_ptr,
    batch_size,
    heads, 
    original_K,
    original_N,
    target_C,
    target_H,
    target_W,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate thread indices
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_out = tl.program_id(2)
    
    # Skip slicing by directly calculating output offset
    # The slice tmp_2 = x[:, :, 1:, :] starts at index 1 in the K dimension
    # So we start from K=1 instead of K=0
    K_start = 1
    
    # Calculate which position in the output this thread handles
    output_idx = pid_out * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Calculate total output size for masking
    total_elements = batch_size * heads * target_C * target_H * target_W
    mask = output_idx < total_elements
    
    # Map linear output index back to 4D coordinates (1, C, H, W)
    output_4d = output_idx.reshape(-1, 1, 1, 1).expand(-1, target_C, target_H, target_W)
    c_idx = (output_4d // (target_H * target_W)) % target_C
    h_idx = (output_4d // target_W) % target_H
    w_idx = output_4d % target_W
    
    # Map to original input coordinates (B, H, K, N)
    # The reshape logic: [B, H, K-1, N] -> [B, C, H_out, W_out]
    # where C = heads * N_eff, H_out = K_eff, W_out = H_spatial
    # We need to decode the dimensions
    
    # For the specific case where input is [1, 8, 50, 19] and output is [1, 152, 7, 7]:
    # - The slice removes element 0 from K dimension: 50 -> 49
    # - After transpose: [1, 8, 19, 49] 
    # - Reshape: [1, 152, 7, 7] where 152 = 8*19, 7 = 49/7, 7 = 7
    
    # General mapping approach:
    # C = heads * N (19 in this case)
    # H_eff = K_eff (49 elements, reshaped to 7*7)
    # W_spatial = original spatial dimension (7 in this case)
    
    # Calculate indices in the intermediate [B, H, N, K_eff] space (after slice+transpose)
    local_c = c_idx // N if N > 0 else c_idx
    head_idx = local_c // heads if heads > 0 else 0
    
    # Map back to original input indices
    if N > 0:
        n_idx = local_c % N
        # K_eff = original_K - 1
        # H_eff = K_eff
        # W_spatial = target_W
        k_idx = h_idx
        h_spatial_idx = w_idx
    else:
        # Fallback for edge cases
        n_idx = 0
        k_idx = h_idx
        h_spatial_idx = w_idx
    
    # Calculate input pointer offset
    in_offset = (pid_batch * heads + head_idx) * (original_K * original_N) + \
                k_idx * original_N + n_idx
    
    # Load from input
    in_val = tl.load(in_ptr + in_offset, mask=mask, other=0.0)
    tl.store(out_ptr + output_idx, in_val, mask=mask)

@torch.fx.wrap  
def optimized_slice_transpose_reshape(x, target_shape=(1, 152, 7, 7)):
    B, H, K_orig, N = x.shape
    C, H_out, W_out = target_shape[1], target_shape[2], target_shape[3]
    
    # The slice removes K_orig=1 element, so K_eff = K_orig - 1
    K_eff = K_orig - 1
    
    # Validate reshape dimensions
    assert C * H_out * W_out == H * N * K_eff, f"Shape mismatch: {C*H_out*W_out} != {H*N*K_eff}"
    
    out = torch.empty(target_shape, dtype=x.dtype, device=x.device)
    
    # Calculate grid dimensions
    total_elements = B * H * C * H_out * W_out
    BLOCK_SIZE = 1024
    num_elements = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Use 3D grid: batch, heads, output_elements
    slice_transpose_reshape_kernel[(B, H, num_elements,)](
        in_ptr=x,
        out_ptr=out,
        batch_size=B,
        heads=H,
        original_K=K_orig,
        original_N=N,
        target_C=C,
        target_H=H_out,
        target_W=W_out,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_slice_transpose_reshape