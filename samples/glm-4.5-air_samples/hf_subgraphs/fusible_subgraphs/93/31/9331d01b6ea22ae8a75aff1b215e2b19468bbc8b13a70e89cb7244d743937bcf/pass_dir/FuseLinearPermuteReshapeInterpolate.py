import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    # Linear operation: [batch, seq_len, in_features] @ [out_features, in_features].T + [out_features]
    linear_out = torch.nn.functional.linear(x, weight, bias)
    # Permute: [batch, seq_len, out_features] -> [batch, out_features, seq_len]
    permute_out = linear_out.permute(0, 2, 1)
    # Reshape: [batch, out_features, seq_len] -> [batch, channels, height, width]
    # where channels * height * width = out_features * seq_len
    reshape_out = permute_out.reshape(x.size(0), -1, 64, 64)
    # Interpolate: upscale spatial dimensions
    interpolate_out = torch.nn.functional.interpolate(reshape_out, size=(128, 128), mode='bilinear', align_corners=False)
    return interpolate_out

def replacement_args(x, weight, bias):
    return (x, weight, bias, x.size(0))

@triton.jit
def linear_permute_reshape_interpolate_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_size, seq_len, in_features, out_features,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one block of the output
    batch_idx = tl.program_id(0)
    feature_idx = tl.program_id(1)
    spatial_idx = tl.program_id(2)
    
    # Calculate offset in the output tensor
    out_spatial_size = 128 * 128
    out_offset = batch_idx * out_features * out_spatial_size + feature_idx * out_spatial_size + spatial_idx
    
    # Load bias for this feature
    bias_val = tl.load(bias_ptr + feature_idx)
    
    # Calculate spatial position
    h_out = spatial_idx // 128
    w_out = spatial_idx % 128
    
    # Scale factor from input 64x64 to output 128x128
    scale = 2.0
    
    # Calculate source positions (bilinear interpolation)
    h_in = h_out / scale
    w_in = w_out / scale
    
    h0 = tl.math.floor(h_in)
    h1 = h0 + 1
    w0 = tl.math.floor(w_in)
    w1 = w0 + 1
    
    # Bilinear weights
    alpha_h = h_in - h0
    alpha_w = w_in - w0
    
    beta_h = 1.0 - alpha_h
    beta_w = 1.0 - alpha_w
    
    # Ensure bounds
    h0 = tl.math.max(0, tl.math.min(63, h0))
    h1 = tl.math.max(0, tl.math.min(63, h1))
    w0 = tl.math.max(0, tl.math.min(63, w0))
    w1 = tl.math.max(0, tl.math.min(63, w1))
    
    # Interpolation result
    result = bias_val
    
    # Compute interpolated value for each input position that contributes to this output
    # We need to map the spatial position back to sequence position
    src_spatial_pos = (h_out * 128 + w_out) // 4  # 64*64 maps to 1/4 of 128*128 spatial locations in source
    seq_idx = src_spatial_pos
    
    if seq_idx < seq_len:
        # Load input feature for this position
        src_offset = batch_idx * seq_len * in_features + seq_idx * in_features + feature_idx % in_features
        x_val = tl.load(x_ptr + src_offset, mask=seq_idx < seq_len, other=0.0)
        
        # Weight for this feature combination
        weight_offset = feature_idx % out_features * in_features + feature_idx % in_features
        weight_val = tl.load(weight_ptr + weight_offset)
        
        result += x_val * weight_val
    
    # Store result
    tl.store(out_ptr + out_offset, result)

@torch.fx.wrap
def optimized_linear_permute_reshape_interpolate(x, weight, bias):
    batch_size, seq_len, in_features = x.shape
    out_features = weight.shape[0]
    
    # Output shape: [batch_size, out_features, 128, 128]
    out_spatial_size = 128 * 128
    output_size = batch_size * out_features * out_spatial_size
    
    out = torch.empty((batch_size, out_features, 128, 128), dtype=x.dtype, device=x.device)
    
    # Adjust grid dimensions for optimization
    BLOCK_SIZE = 256
    batch_grid = batch_size
    feature_grid = (out_features + 31) // 32  # 32 features per program
    spatial_grid = (out_spatial_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    linear_permute_reshape_interpolate_kernel[(batch_grid, feature_grid, spatial_grid)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        in_features=in_features,
        out_features=out_features,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return optimized_linear_permute_reshape_interpolate