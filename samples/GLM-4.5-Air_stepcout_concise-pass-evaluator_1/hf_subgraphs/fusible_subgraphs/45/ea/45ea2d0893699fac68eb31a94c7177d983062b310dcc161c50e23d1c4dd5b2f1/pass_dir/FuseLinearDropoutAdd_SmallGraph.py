import torch
import triton
import triton.language as tl
import math

def pattern(input, weight, bias, residual):
    # Linear transformation using exact variable order from model.py
    tmp = torch.nn.functional.linear(input, weight, bias)
    # Dropout with p=0.1
    tmp_drop = torch.nn.functional.dropout(tmp, 0.1, False, False)
    # Residual addition
    result = residual + tmp_drop
    return (result, tmp_drop)

def replacement_args(input, weight, bias, residual):
    return (input, weight, bias, residual)

@triton.jit
def linear_dropout_add_kernel_small(
    input_ptr, weight_ptr, bias_ptr, residual_ptr, output_ptr, output_dropout_ptr,
    batch_size, seq_len, hidden_size, n_hidden,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ID for 3D blocking (batch, sequence, hidden)
    pid = tl.program_id(0)
    
    # Calculate total number of elements and split work
    total_elements = batch_size * seq_len * hidden_size
    num_programs = 1024  # Use 1024 programs for parallel processing
    elements_per_program = (total_elements + num_programs - 1) // num_programs
    start_idx = pid * elements_per_program
    end_idx = min((pid + 1) * elements_per_program, total_elements)
    
    # Process each assigned element
    for idx in range(start_idx, end_idx):
        # Convert linear index to 3D coordinates
        i = idx // (seq_len * hidden_size)  # batch index
        j = (idx % (seq_len * hidden_size)) // hidden_size  # sequence index
        k = idx % hidden_size  # hidden feature index
        
        if i < batch_size and j < seq_len and k < hidden_size:
            # Reshape input for linear operation: [batch_size, seq_len, hidden_size] -> [batch_size*seq_len, hidden_size]
            input_idx = (i * seq_len + j) * hidden_size + k
            
            # Compute linear operation for each hidden output dimension
            acc = 0.0
            for l in range(0, hidden_size, BLOCK_K):
                l_end = min(l + BLOCK_K, hidden_size)
                
                # Load input element
                input_val = tl.load(input_ptr + input_idx, 
                                  mask=l < hidden_size, other=0.0)
                # Load weight element: weight is [n_hidden, hidden_size]
                weight_idx = l * n_hidden + k if k < n_hidden else 0
                weight_val = tl.load(weight_ptr + weight_idx,
                                   mask=(l < hidden_size) and (k < n_hidden), other=0.0)
                
                acc += input_val * weight_val
            
            # Add bias
            bias_idx = k if k < n_hidden else 0
            bias_val = tl.load(bias_ptr + bias_idx,
                             mask=k < n_hidden, other=0.0)
            linear_output = acc + bias_val
            
            # Apply dropout p=0.1 (10% chance of setting to 0)
            dropout_scale = 1.0 / 0.9  # Scale since dropout sets 10% to 0
            # Use simple deterministic approach for reproducibility
            dropout_mask = 1.0
            # In real implementation we'd use proper random number generation
            # For now, just scale the output
            dropout_output = linear_output * dropout_scale
            
            # Add residual
            residual_val = tl.load(residual_ptr + input_idx,
                                 mask=k < hidden_size, other=0.0)
            final_output = residual_val + dropout_output
            
            # Store results
            tl.store(output_ptr + input_idx, final_output)
            
            # Store dropout output (needed for pattern return)
            tl.store(output_dropout_ptr + input_idx, dropout_output)

@torch.fx.wrap
def fused_linear_dropout_add_small(input, weight, bias, residual):
    # Get input tensor shapes
    # For BERT models: input is [batch_size, seq_len, hidden_size]
    batch_size = input.shape[0]
    seq_len = input.shape[1]
    hidden_size = input.shape[2]
    n_hidden = bias.shape[0]
    
    # Reshape input to 2D for linear operation: [batch_size*seq_len, hidden_size]
    input_2d = input.reshape(-1, hidden_size)
    residual_2d = residual.reshape(-1, hidden_size)
    
    # Create 2D output tensors
    output_2d = torch.empty((batch_size * seq_len, n_hidden), dtype=torch.float32, device=input.device)
    output_dropout_2d = torch.empty((batch_size * seq_len, n_hidden), dtype=torch.float32, device=input.device)
    
    # Launch kernel with 1024 programs
    num_programs = 1024
    
    # Launch kernel
    linear_dropout_add_kernel_small[(num_programs,)](
        input_ptr=input_2d,
        weight_ptr=weight,
        bias_ptr=bias,
        residual_ptr=residual_2d,
        output_ptr=output_2d,
        output_dropout_ptr=output_dropout_2d,
        batch_size=batch_size * seq_len,
        seq_len=1,  # Flattened
        hidden_size=hidden_size,
        n_hidden=n_hidden,
        BLOCK_M=8,
        BLOCK_N=8,
        BLOCK_K=16
    )
    
    # Reshape back to 3D
    output = output_2d.reshape(batch_size, seq_len, n_hidden)
    output_dropout = output_dropout_2d.reshape(batch_size, seq_len, n_hidden)
    
    return output, output_dropout

def replacement_func():
    return fused_linear_dropout_add_small