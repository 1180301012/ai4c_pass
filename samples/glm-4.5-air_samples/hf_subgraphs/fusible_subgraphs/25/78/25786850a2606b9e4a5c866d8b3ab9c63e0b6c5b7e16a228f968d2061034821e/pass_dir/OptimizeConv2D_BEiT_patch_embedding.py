import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    # Conv2D: in_0(input), in_2(weight), in_1(bias)
    # stride=(16, 16), padding=(0, 0), dilation=(1, 1), groups=1
    tmp_5 = torch.conv2d(x, weight, bias, (16, 16), (0, 0), (1, 1), 1)
    return tmp_5

def replacement_args(x, weight, bias):
    return (x, weight, bias)

@triton.jit
def conv2d_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N, C_in, H_in, W_in,
    C_out,
    stride_H, stride_W,
    H_out, W_out,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
):
    # Get program IDs
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Compute offsets for output
    n_offset = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_offset = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    n_mask = n_offset < N
    c_mask = c_offset < C_out
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_C), dtype=tl.float32)
    
    # Loop over input channels
    for k in range(C_in):
        # Add bias
        bias = tl.load(bias_ptr + c_offset, mask=c_mask, other=0.0).to(tl.float32)
        acc += bias[None, :]
        
        # Load weights for this input channel - shape [BLOCK_SIZE_C, K_H, K_W]
        weight_base = k * C_out * K_H * K_W
        weight_indices = weight_base + c_offset[:, None] * K_H * K_W + tl.arange(0, K_H * K_W)
        weight_mask = c_mask[:, None] & (tl.arange(0, K_H * K_W) < K_H * K_W)[None, :]
        
        # Load all weights for this input channel and flatten spatial dimensions
        weights_flat = tl.load(weight_ptr + weight_indices, mask=weight_mask, other=0.0).to(tl.float32)
        weights_flat = weights_flat.reshape(BLOCK_SIZE_C, K_H * K_W)
        
        # Process each spatial location
        for spat_idx in range(K_H * K_W):
            # Input coordinates for this spatial position
            hh = spat_idx // K_W  # height position
            ww = spat_idx % K_W   # width position
            
            # Compute input positions
            h_in = n_offset * stride_H + hh
            w_in = c_offset * stride_W + ww
            
            # Validity check
            h_valid = (h_in < H_out) & (hh < K_H)
            w_valid = (w_in < W_out) & (ww < K_W)
            
            # Generate mask for valid positions
            mask = h_valid & w_valid
            
            # Calculate input tensor index for each valid position
            base_offset = k * H_in * W_in
            input_coords = base_offset + h_in * W_in + w_in
            
            # Load input values
            x_vals = tl.load(x_ptr + input_coords, mask=mask, other=0.0).to(tl.float32)
            
            # Get weight values for this spatial position
            w_vals = weights_flat[:, spat_idx]
            
            # Convert to proper shapes for accumulation
            # x_vals is [BLOCK_SIZE_N] for valid batch/oc combinations
            # w_vals is [BLOCK_SIZE_C] 
            # We need to accumulate only for valid combinations
            
            # Replicate x_vals for all output channels where mask is valid
            if mask.any():
                # Create expanded indices for accumulation
                batch_oc_indices = tl.arange(BLOCK_SIZE_N) * BLOCK_SIZE_C + tl.arange(BLOCK_SIZE_C)
                batch_oc_valid = mask.reshape(BLOCK_SIZE_N * BLOCK_SIZE_C)
                
                # Reshape for broadcasting
                x_expanded = x_vals[mask]  # Values for valid positions
                w_expanded = w_vals  # All output channel weights
                
                # Perform accumulation if there are valid values
                if x_expanded.shape[0] > 0:
                    # Use 1D indexing to accumulate
                    acc_flat = acc.sum()  # Flatten for 1D operations
                    acc += 0  # Ensure gradients work, this logic needs simplification
                    
                    # Alternative: simpler element-wise approach for valid positions
                    # Process each valid batch/output channel combination
                    for bi in range(BLOCK_SIZE_N):
                        if h_valid[bi]:
                            for ci in range(BLOCK_SIZE_C):
                                if c_mask[ci] and w_valid:
                                    input_coord = base_offset + h_in[bi] * W_in + w_in
                                    x_val = tl.load(x_ptr + input_coord).to(tl.float32)
                                    acc[bi, ci] += x_val * w_vals[ci]
    
    # Store result
    out_idx = n_offset[:, None] * C_out + c_offset[None, :]
    tl.store(out_ptr + out_idx, acc, mask=n_mask[:, None] & c_mask[None, :])

@torch.fx.wrap
def optimized_conv2d(x, weight, bias):
    N, C_in, H_in, W_in = x.shape
    C_out, K_H, K_W = weight.shape[0], weight.shape[2], weight.shape[3]
    
    # Calculate output dimensions (with stride 16)
    H_out = (H_in - K_H) // 16 + 1
    W_out = (W_in - K_W) // 16 + 1
    
    # Reshape weight to match expected format [C_out, C_in, K_H, K_W]
    # The input weight is [768, 3, 16, 16] which is already in correct format
    
    # Launch kernel
    output = torch.empty((N, C_out, H_out, W_out), dtype=torch.float32, device=x.device)
    
    # Block sizes for the simplified kernel
    BLOCK_SIZE_N = 32      # Batch dimension 
    BLOCK_SIZE_C = 64      # Output channel dimension
    
    # Grid dimensions
    num_n_blocks = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_c_blocks = (C_out + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    grid = (num_n_blocks, num_c_blocks)
    
    conv2d_kernel[grid](
        x, weight, bias, output,
        N, C_in, H_in, W_in,
        C_out, 
        16, 16,  # stride_H, stride_W
        H_out, W_out,
        BLOCK_SIZE_N, BLOCK_SIZE_C, 16, 16  # K_H, K_W
    )
    
    return output

def replacement_func():
    return optimized_conv2d