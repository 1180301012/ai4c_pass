import torch
import triton
import triton.language as tl


def pattern(input_in, concat_tensor, running_mean, running_var, weight, bias):
    tmp_7 = torch.cat([input_in, concat_tensor], 1)
    tmp_8 = torch.nn.functional.batch_norm(tmp_7, running_mean, running_var, weight, bias, False, 0.1, 0.001)
    tmp_9 = torch.nn.functional.relu(tmp_8, inplace=False)
    return tmp_9


def replacement_args(input_in, concat_tensor, running_mean, running_var, weight, bias):
    return (input_in, concat_tensor, running_mean, running_var, weight, bias)


@triton.jit
def fused_cat_bn_relu_kernel(
    input1_ptr,      # First input to cat
    input2_ptr,      # Second input to cat 
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_input1_c,      # Number of channels in first input
    n_input2_c,      # Number of channels in second input  
    n_total_c,       # Total channels after concatenation (n_input1_c + n_input2_c)
    height,
    width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # Calculate grid coordinates
    m = tl.program_id(0)  # height position
    n = tl.program_id(1)  # width position
    k = tl.program_id(2)  # channel block
    
    # Each thread processes a block of channels for one pixel
    batch_base = k * BLOCK_SIZE_K
    batch_end = min(batch_base + BLOCK_SIZE_K, n_total_c)
    
    # Calculate position in output
    output_offset = m * width * n_total_c + n * n_total_c
    
    # Load batch normalization parameters for the channels in this block
    batch_means = tl.load(running_mean_ptr + batch_base).to(tl.float32)
    batch_vars = tl.load(running_var_ptr + batch_base).to(tl.float32)
    batch_weights = tl.load(weight_ptr + batch_base).to(tl.float32)
    batch_biases = tl.load(bias_ptr + batch_base).to(tl.float32)
    
    # Process each channel in the block
    for c in range(batch_base, batch_end):
        load_offset = output_offset + c
        
        # Load input values (need to load from both concatenated tensors)
        val1 = 0.0
        val2 = 0.0
        
        # Load from first tensor (channels 0 to n_input1_c-1)
        if c < n_input1_c:
            val1 = tl.load(input1_ptr + load_offset, mask=True)
        
        # Load from second tensor (channels n_input1_c to n_total_c-1)  
        if c >= n_input1_c:
            second_c = c - n_input1_c
            second_offset = m * width * n_input2_c + n * n_input2_c + second_c
            val2 = tl.load(input2_ptr + second_offset, mask=True)
        
        # Concatenated value
        x = val1 + val2
        
        # Normalize: (x - mean) / sqrt(var + eps)
        mean = batch_means[c - batch_base]
        var = batch_vars[c - batch_base]
        eps = 0.001
        
        # Handle potential numerical issues
        if var < 0:
            var = 0.0
        
        normalized = (x - mean) / tl.sqrt(var + eps)
        
        # Scale and shift: normalize * weight + bias
        scale = batch_weights[c - batch_base]
        shift = batch_biases[c - batch_base]
        result = normalized * scale + shift
        
        # ReLU activation
        if result > 0:
            final_result = result
        else:
            final_result = 0.0
        
        # Store result
        tl.store(output_ptr + load_offset, final_result)


@torch.fx.wrap  
def fused_cat_bn_relu(input_in, concat_tensor, running_mean, running_var, weight, bias):
    # Get tensor shapes
    shape1 = input_in.shape
    shape2 = concat_tensor.shape
    
    n_input1_c, n_input2_c = shape1[0], shape2[0]
    height, width = shape1[2], shape1[3]
    n_total_c = n_input1_c + n_input2_c
    
    # Create output tensor
    output_shape = [n_total_c, height, width]
    output = torch.empty(output_shape, dtype=input_in.dtype, device=input_in.device)
    
    # Configure block sizes for better GPU utilization
    BLOCK_SIZE_M = 8   # Process 8 rows per block
    BLOCK_SIZE_N = 8   # Process 8 columns per block
    BLOCK_SIZE_K = min(256, n_total_c)  # Process up to 256 channels per block
    
    # Calculate grid size
    grid_m = (height + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_k = (n_total_c + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    grid = (grid_m, grid_n, grid_k)
    
    # Launch kernel
    fused_cat_bn_relu_kernel[grid](
        input1_ptr=input_in,
        input2_ptr=concat_tensor,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_input1_c=n_input1_c,
        n_input2_c=n_input2_c,
        n_total_c=n_total_c,
        height=height,
        width=width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output


def replacement_func():
    return fused_cat_bn_relu