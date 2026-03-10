import torch
import triton
import triton.language as tl

def pattern(x, y):
    # Match: addition + split + permute + view pattern for 576 feature size
    tmp_0 = x + y
    tmp_1 = torch.functional.split(tmp_0, [1, 576], 1)
    tmp_2 = tmp_1[0]
    tmp_3 = tmp_1[1]
    tmp_4 = tmp_3.permute(0, 2, 1)
    tmp_5 = tmp_4.view(1, 384, 24, 24)
    return tmp_2, tmp_5

def replacement_args(x, y):
    # Return just the input tensors - calculations done in wrapper
    return x, y

def replacement_func():
    return fused_split_permute_view_wrapper

@triton.jit
def fused_split_permute_view_kernel(x_ptr, y_ptr, out_head_ptr, out_feature_ptr, 
                                   n_batch, n_head_plus_feature, n_channels,
                                   head_size, feature_size, output_height, output_width,
                                   BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, 
                                   BLOCK_SIZE_K: tl.constexpr):
    """
    Fused kernel that performs:
    1. Element-wise addition
    2. Split and process the feature portion through permute and reshape
    """
    # Each program processes a tile of the output
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Handle addition and processing
    for k in range(0, n_channels, BLOCK_SIZE_K):
        # Process the feature portion permutation and reshape
        if pid_n < feature_size // BLOCK_SIZE_N:
            # Column offset for feature portion (dim 1, starting from head_size)
            col_offset = head_size + pid_n * BLOCK_SIZE_N
            # Load X and Y from feature portion
            x = tl.load(x_ptr + pid_m * n_head_plus_feature * n_channels + col_offset * n_channels + k,
                       mask=k < n_channels - k, other=0.0)
            y = tl.load(y_ptr + pid_m * n_head_plus_feature * n_channels + col_offset * n_channels + k,
                       mask=k < n_channels - k, other=0.0)
            
            # Element-wise addition
            add_result = x + y
            
            # Store the head portion (first column directly to head output)
            if pid_n == 0:
                head_output_addr = out_head_ptr + pid_m * n_channels + k
                if k < n_channels:
                    tl.store(head_output_addr, add_result)
            
            # For feature portion: need permute (0,2,1) + reshape to (batch, channels, height, width)
            # Calculate 2D coordinates for the feature map
            # We're processing (features, channels) -> (channels, height, width)
            feature_idx = col_offset - head_size  # 0 to feature_size-1
            height = feature_idx // output_width
            width = feature_idx % output_width
            
            # Store with permute and reshape
            output_addr = out_feature_ptr + \
                         pid_m * n_channels * output_height * output_width + \
                         k * output_height * output_width + \
                         height * output_width + width
            
            if k < n_channels:
                tl.store(output_addr, add_result)

@torch.fx.wrap
def fused_split_permute_view_wrapper(x, y):
    batch_size, seq_len, channels = x.shape
    head_size = 1
    feature_size = 576  # Specific to this case
    output_height = 24    # Specific to this case
    output_width = 24     # Specific to this case
    
    output_head = torch.empty(batch_size, head_size, channels, dtype=x.dtype, device=x.device)
    output_feature = torch.empty(batch_size, channels, output_height, output_width, dtype=x.dtype, device=x.device)
    
    # Configure block sizes for optimal performance
    BLOCK_SIZE_M = 1  # batch size is 1
    BLOCK_SIZE_N = min(128, feature_size)
    BLOCK_SIZE_K = min(128, channels)
    
    # Calculate grid dimensions
    grid_m = batch_size
    grid_n = (feature_size + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    fused_split_permute_view_kernel[(
        grid_m,
        grid_n,
        (channels + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    )](
        x, y, output_head, output_feature,
        batch_size, seq_len, channels,
        head_size, feature_size, output_height, output_width,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return output_head, output_feature