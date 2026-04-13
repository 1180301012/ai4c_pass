import torch
import triton
import triton.language as tl

# Pattern matching function - matches the exact computation sequence
def pattern(in_0: torch.Tensor, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.sigmoid()
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.gelu(tmp_4, approximate='none')
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    tmp_8 = torch.nn.functional.dropout(tmp_7, 0.0, False, False)
    return (tmp_8,)

# Argument extraction function
def replacement_args(in_0: torch.Tensor, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Custom Triton kernels with compile-time constants for specific shapes
@triton.jit
def se_block_conv_kernel(
    x_ptr,  # input: [B, 64, 1, 1] 
    weight_ptr,  # weights: [1024, 64, 1, 1]
    bias_ptr,  # bias: [1024]
    out_ptr,  # output: [B, 1024, 1, 1]
    B: tl.constexpr,
    C_IN: tl.constexpr,  # 64
    C_OUT: tl.constexpr, # 1024
    C_TILE: tl.constexpr,
):
    """1x1 convolution + sigmoid activation for specific shapes"""
    pid = tl.program_id(0)
    
    # Calculate ranges
    batch_id = pid // ((C_OUT + C_TILE - 1) // C_TILE)
    c_start = (pid % ((C_OUT + C_TILE - 1) // C_TILE)) * C_TILE
    c_offset = tl.arange(0, C_TILE)
    c_mask = c_offset < (C_OUT - c_start)
    
    # Load input for this batch (spatial size = 1x1, so just load the values)
    x_vals = tl.load(x_ptr + batch_id * C_IN + tl.arange(0, C_TILE), mask=c_mask, other=0.0)
    
    # Compute convolution for each output channel
    conv_vals = tl.zeros((C_TILE,), dtype=tl.float32)
    for k in range(C_IN):
        # Load weight vector for this input channel
        weight = tl.load(weight_ptr + (c_start + c_offset) * C_IN + k, mask=c_mask)
        conv_vals += weight * x_vals
    
    # Load bias and apply sigmoid
    bias = tl.load(bias_ptr + c_start + c_offset, mask=c_mask)
    result = 1.0 / (1.0 + tl.exp(-(conv_vals + bias)))
    
    # Store result
    out_offset = batch_id * C_OUT + c_start + c_offset
    tl.store(out_ptr + out_offset, result, mask=c_mask)

@triton.jit
def se_block_mult_gelu_kernel(
    sigmoid_ptr,  # sigmoid output: [B, 1024, 1, 1]
    features_ptr,  # features: [B, 1024, H, W]
    out_ptr,  # output: [B, 1024, H, W]
    B: tl.constexpr,
    C: tl.constexpr,  # 1024
    H: tl.constexpr,
    W: tl.constexpr,
    C_TILE: tl.constexpr,
    HW_TILE: tl.constexpr,
):
    """Element-wise multiplication with features + GELU activation for specific shapes"""
    pid = tl.program_id(0)
    
    # Calculate ranges: pid = batch_id * (num_channel_spatial_programs) + channel_program_id
    total_programs = ((C + C_TILE - 1) // C_TILE) * ((H * W + HW_TILE - 1) // HW_TILE)
    batch_id = pid // total_programs
    spatial_program_id = (pid % total_programs) // ((C + C_TILE - 1) // C_TILE)
    channel_program_id = (pid % total_programs) % ((C + C_TILE - 1) // C_TILE)
    
    # Calculate ranges within this program
    c_start = channel_program_id * C_TILE
    s_start = spatial_program_id * HW_TILE
    c_offset = tl.arange(0, C_TILE)
    s_offset = tl.arange(0, HW_TILE)
    
    # masks
    c_mask = c_offset < (C - c_start)
    s_mask = s_offset < min(HW_TILE, H * W - s_start)
    
    # Batch stride
    sigmoid_offset = batch_id * C * H * W + c_start * H * W + s_start
    features_offset = sigmoid_offset
    out_offset = sigmoid_offset
    
    # Load sigmoid values (broadcast across spatial dimensions)
    sigmoid_vals = tl.load(sigmoid_ptr + batch_id * C + c_start + c_offset, mask=c_mask, other=0.0)
    
    # Process each spatial position in the tile
    for s_idx in tl.range(0, min(HW_TILE, H * W - s_start)):
        spatial_offset = features_offset + (s_idx * C)
        
        # Load features for this spatial position
        feats = tl.load(features_ptr + spatial_offset + c_offset, mask=c_mask, other=0.0)
        
        # Element-wise operations
        mult_out = sigmoid_vals * feats
        
        # GELU computation (tensors are already fp32)
        x = mult_out
        sigmoid_input = -1.702 * x + 0.5 * x * x
        sigmoid_output = tl.exp(sigmoid_input)
        gelu_out = x * (1.0 / (1.0 + sigmoid_output))
        
        # Store result
        tl.store(out_ptr + spatial_offset + c_offset, gelu_out, mask=c_mask)

@triton.jit
def se_block_pool_kernel(
    features_ptr,  # features: [B, 1024, H, W]
    out_ptr,  # output: [B, 1024]
    B: tl.constexpr,
    C: tl.constexpr,  # 1024
    H: tl.constexpr,
    W: tl.constexpr,
    C_TILE: tl.constexpr,
):
    """Global mean pooling + flatten (dropout eliminated) for specific shapes"""
    pid = tl.program_id(0)
    
    # Calculate ranges: pid = batch_id * (num_channel_programs) + channel_program_id
    num_channel_programs = (C + C_TILE - 1) // C_TILE
    batch_id = pid // num_channel_programs
    channel_program_id = pid % num_channel_programs
    
    # Calculate range within this program
    c_start = channel_program_id * C_TILE
    c_offset = tl.arange(0, C_TILE)
    c_mask = c_offset < (C - c_start)
    
    # Global mean pooling
    pool_sum = tl.zeros((C_TILE,), dtype=tl.float32)
    spatial_dim = H * W
    
    # Accumulate across spatial positions
    for hw in range(spatial_dim):
        spatial_offset = (batch_id * spatial_dim + hw) * C + c_start
        feats = tl.load(features_ptr + spatial_offset + c_offset, mask=c_mask, other=0.0)
        pool_sum += feats
    
    # Compute mean
    pool_mean = pool_sum / spatial_dim
    
    # Store result
    out_offset = batch_id * C + c_start + c_offset
    tl.store(out_ptr + out_offset, pool_mean, mask=c_mask)

# Factory function to create optimized SE block with specific shape
class FastVITSEBlockOptimizedFactory:
    @staticmethod
    def create_optimized_se_block(B, H, W):
        """Create optimized SE block for specific batch size and spatial dimensions"""
        dtype = torch.bfloat16  # Default, will be overridden in the actual calls
        
        def optimized_se_block(in_0, in_1, in_2, in_3):
            # Convert to fp32 for computation (better math support) then back to original dtype
            original_dtype = dtype
            in_0 = in_0.to(torch.float32)
            in_1 = in_1.to(torch.float32)
            in_2 = in_2.to(torch.float32)
            in_3 = in_3.to(torch.float32)
            
            # Step 1: Conv + Sigmoid
            conv_sigmoid_out = torch.empty((B, 1024, 1, 1), dtype=torch.float32, device=in_3.device)
            C_TILE_conv = 128  # Tile size for channels
            
            # Launch kernel: 1D grid that combines batch and channel processing
            num_channel_programs = (1024 + C_TILE_conv - 1) // C_TILE_conv
            num_programs_conv = B * num_channel_programs
            se_block_conv_kernel[(num_programs_conv,)](
                in_3, in_1, in_0, conv_sigmoid_out,
                B, C_IN=64, C_OUT=1024, C_TILE=C_TILE_conv
            )
            
            # Step 2: Element-wise mult + GELU
            mult_gelu_out = torch.empty((B, 1024, H, W), dtype=torch.float32, device=in_3.device)
            C_TILE_mult = 64   # Tile size for channels
            HW_TILE_mult = 1024  # Tile size for spatial dimensions
            
            # Launch kernel: 1D grid that combines batch, channel, and spatial processing
            num_channel_programs_mult = (1024 + C_TILE_mult - 1) // C_TILE_mult
            num_spatial_programs_mult = (H * W + HW_TILE_mult - 1) // HW_TILE_mult
            num_programs_mult = B * num_channel_programs_mult * num_spatial_programs_mult
            se_block_mult_gelu_kernel[(num_programs_mult,)](
                conv_sigmoid_out, in_2, mult_gelu_out,
                B, C=1024, H=H, W=W,
                C_TILE=C_TILE_mult,
                HW_TILE=HW_TILE_mult
            )
            
            # Step 3: Global mean pooling + flatten (dropout eliminated)
            final_out = torch.empty((B, 1024), dtype=torch.float32, device=in_3.device)
            C_TILE_pool = 128  # Tile size for channels
            
            # Launch kernel: 1D grid that combines batch and channel processing
            num_channel_programs_pool = (1024 + C_TILE_pool - 1) // C_TILE_pool
            num_programs_pool = B * num_channel_programs_pool
            se_block_pool_kernel[(num_programs_pool,)](
                mult_gelu_out, final_out,
                B, C=1024, H=H, W=W,
                C_TILE=C_TILE_pool
            )
            
            # Cast back to original dtype
            final_out = final_out.to(original_dtype)
            
            return (final_out,)
        
        return optimized_se_block

# Dynamic dispatch based on input shapes
@torch.fx.wrap
def fastvit_se_block_optimized(in_0, in_1, in_2, in_3):
    """Comprehensive SE block optimization with dynamic shape handling"""
    
    # Get tensor shapes and device
    B, C, H, W = in_2.shape
    
    # Create optimized function for this specific shape
    optimized_func = FastVITSEBlockOptimizedFactory.create_optimized_se_block(B, H, W)
    
    # Call the optimized function
    return optimized_func(in_0, in_1, in_2, in_3)

# Replacement function
def replacement_func():
    return fastvit_se_block_optimized