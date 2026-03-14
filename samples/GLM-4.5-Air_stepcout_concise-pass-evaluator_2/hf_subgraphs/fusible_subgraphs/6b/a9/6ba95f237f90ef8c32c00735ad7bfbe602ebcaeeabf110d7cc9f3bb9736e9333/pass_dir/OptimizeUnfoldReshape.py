import torch
import triton
import triton.language as tl

@triton.jit
def triton_unfold_reshape_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    in_channels,
    in_height,
    in_width,
    out_channels,
    patch_height,
    patch_width,
    stride_h,
    stride_w,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_PATCH: tl.constexpr
):
    # Input: [batch_size, in_channels, in_height, in_width]
    # Output: [batch_size, out_channels, patch_height, patch_width, out_h, out_w]
    # But we want: [batch_size, out_channels, patch_height*patch_width, out_h*out_w]
    # Then reshape to: [batch_size, out_channels, patch_height*patch_width, out_h*out_w]
    
    # Total patches = out_h * out_w
    out_h = (in_height - patch_height) // stride_h + 1
    out_w = (in_width - patch_width) // stride_w + 1
    total_patches = out_h * out_w
    
    grid_m = tl.cdiv(batch_size * total_patches, BLOCK_SIZE_M)
    grid_n = tl.cdiv(out_channels, BLOCK_SIZE_N)
    grid_k = tl.cdiv(in_channels * patch_height * patch_width, BLOCK_SIZE_K)
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    # Block ranges
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    k_start = pid_k * BLOCK_SIZE_K
    
    # Accumulator for dot product
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # K loop over patches and input channels
    k_end = min(k_start + BLOCK_SIZE_K, in_channels * patch_height * patch_width)
    for k in range(k_start, k_end):
        # Determine which channel and which patch element we're processing
        channel = k // (patch_height * patch_width)
        patch_idx = k % (patch_height * patch_width)
        patch_h = patch_idx // patch_width
        patch_w = patch_idx % patch_width
        
        for m_idx in range(BLOCK_SIZE_M):
            m_offset = m_start + m_idx
            if m_offset >= batch_size * total_patches:
                continue
            
            # Compute spatial location of this patch
            patch_num = m_offset // total_patches
            local_patch_idx = m_offset % total_patches
            local_h = local_patch_idx // out_w
            local_w = local_patch_idx % out_w
            
            # Compute input coordinates
            in_h = local_h * stride_h + patch_h
            in_w = local_w * stride_w + patch_w
            if in_h >= in_height or in_w >= in_width:
                continue
            
            # Load input element
            x_offset = (patch_num * in_channels * in_height * in_width +
                       channel * in_height * in_width +
                       in_h * in_width + in_w)
            x_val = tl.load(x_ptr + x_offset, other=0.0)
            
            for n_idx in range(BLOCK_SIZE_N):
                n_offset = n_start + n_idx
                if n_offset >= out_channels:
                    continue
                
                # Weight for this channel output combination
                # Since we're doing identity transform (unfold), weight is 1.0
                weight = 1.0
                acc[m_idx, n_idx] += x_val * weight
    
    # Store results
    for m_idx in range(BLOCK_SIZE_M):
        m_offset = m_start + m_idx
        if m_offset >= batch_size * total_patches:
            continue
        
        for n_idx in range(BLOCK_SIZE_N):
            n_offset = n_start + n_idx
            if n_offset >= out_channels:
                continue
            
            # Store in output matrix [batch_size * total_patches, out_channels]
            out_offset = m_offset * out_channels + n_offset
            tl.store(out_ptr + out_offset, acc[m_idx, n_idx])

@torch.fx.wrap
def triton_unfold_reshape(x, kernel_size=(2, 2), stride=(2, 2), reshape_target=(1, 128, 4, -1)):
    batch_size, in_channels, in_height, in_width = x.shape
    patch_height, patch_width = kernel_size
    stride_h, stride_w = stride
    
    # Calculate output dimensions
    out_h = (in_height - patch_height) // stride_h + 1
    out_w = (in_width - patch_width) // stride_w + 1
    total_patches = out_h * out_w
    
    # For the unfold operation, we effectively change in_channels to out_channels = in_channels * patch_height * patch_width
    # But in this case, the model expects to maintain the channel transformation, so we need to be careful
    
    # Based on the model, after conv2d we have [1, 128, 32, 32]
    # Then unfold produces [1, 128*4, 16*16] = [1, 512, 256]
    # Then reshape to [1, 128, 4, 64]
    
    # Direct compute from conv2d output to final reshape
    out_channels_intermediate = in_channels * patch_height * patch_width  # 128 * 4 = 512
    
    # Create output matrix: [batch_size * total_patches, out_channels_intermediate]
    out_matrix = torch.empty(batch_size * total_patches, out_channels_intermediate, device=x.device, dtype=x.dtype)
    
    # Set up Triton kernel
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_PATCH = 4
    
    grid_m = (batch_size * total_patches + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (out_channels_intermediate + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_k = (in_channels * patch_height * patch_width + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    triton_unfold_reshape_kernel[(grid_m, grid_n, grid_k)](
        x,
        out_matrix,
        batch_size, in_channels, in_height, in_width,
        out_channels_intermediate, patch_height, patch_width,
        stride_h, stride_w,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, BLOCK_SIZE_PATCH
    )
    
    # Reshape to final target: [batch_size, out_channels_intermediate, total_patches]
    out_reshaped = out_matrix.reshape(batch_size, out_channels_intermediate, total_patches)
    
    # Final reshape as specified in the model: [1, 128, 4, -1]
    # Since out_channels_intermediate = 512 and we want 128, this suggests grouping
    final_out = out_reshaped.reshape(reshape_target)
    
    return final_out

def pattern(x):
    tmp_2 = torch.nn.functional.unfold(x, kernel_size=(2, 2), stride=(2, 2))
    tmp_3 = tmp_2.reshape(1, 128, 4, -1)
    return tmp_3

def replacement_args(conv_out):
    return (conv_out,)

def replacement_func():
    return triton_unfold_reshape