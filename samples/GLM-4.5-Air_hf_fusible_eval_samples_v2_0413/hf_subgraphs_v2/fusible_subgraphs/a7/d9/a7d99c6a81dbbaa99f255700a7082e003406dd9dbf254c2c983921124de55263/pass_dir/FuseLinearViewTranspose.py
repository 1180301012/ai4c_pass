import torch
import triton
import triton.language as tl

def pattern(hidden_states, weight, bias):
    # Linear transformation: [1, 1, 512] @ [512, 512] + [512] -> [1, 1, 512]
    linear = torch.nn.functional.linear(hidden_states, weight, bias)
    # Reshape to [1, 1, 8, 64] and transpose to [1, 8, 1, 64]
    reshaped = linear.view(1, 1, -1, 64)
    transposed = reshaped.transpose(1, 2)
    # Need to create both intermediate and final result for observable outputs
    # intermediate is the reshaped tensor, final is the transposed
    return linear, reshaped, transposed

@triton.jit
def linear_view_transpose_kernel(
    hidden_states_ptr,
    weight_ptr, 
    bias_ptr,
    output_ptr1,  # linear result [1, 1, 512]
    output_ptr2,  # reshaped result [1, 1, 8, 64] 
    output_ptr3,  # final transposed result [1, 8, 1, 64]
    hidden_states_size,
    weight_size_m,
    weight_size_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Matrix multiplication for linear transformation
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute offsets for matrix multiplication
    m_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N
    k_offset = tl.arange(0, BLOCK_SIZE_K)
    
    # Load weight matrix tile
    weight_ptrs = weight_ptr + n_offset * weight_size_m + k_offset
    weight_tile = tl.load(weight_ptrs, mask=k_offset < weight_size_n, other=0.0)
    
    # Load bias for this output column
    bias_ptr_n = bias_ptr + n_offset
    bias_n = tl.load(bias_ptr_n, mask=n_offset < weight_size_n, other=0.0)
    
    # Process each hidden_states batch
    # hidden_states is [1, 1, 512] -> flatten to [512]
    offsets_m = tl.arange(0, hidden_states_size)
    
    for k in range(0, hidden_states_size, BLOCK_SIZE_K):
        k_remaining = hidden_states_size - k
        k_block_size = min(BLOCK_SIZE_K, k_remaining)
        
        # Load hidden_states vector tile
        hidden_ptrs = hidden_states_ptr + k + k_offset
        hidden_tile = tl.load(hidden_ptrs, mask=k_offset < k_block_size, other=0.0)
        
        # Compute dot product for this column
        acc = tl.dot(hidden_tile, weight_tile[:k_block_size])
        
        if pid_m == 0:  # Only one batch in this case
            output_offset = tl.program_id(1) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            final_acc = acc + bias_n
            tl.store(output_ptr1 + output_offset, final_acc, mask=offsets_m < BLOCK_SIZE_N)
    
    # Now handle reshape and transpose with separate kernel launch
    # This is a simplified version for the specific shape [1, 1, 512] -> [1, 8, 1, 64]
    if pid_m == 0 and pid_n == 0:
        # Reshape linear result [1, 1, 512] to [1, 1, 8, 64]
        # and then transpose to [1, 8, 1, 64]
        # Since this is a small shape, we can handle it directly
        offset = tl.arange(0, 512)[:, None]
        row = offset // 64
        col = offset % 64
        
        # Store in reshaped format [1, 1, 8, 64] as flattened [512]
        tl.store(output_ptr2 + offset.flatten(), acc.flatten(), mask=offset < 512)
        
        # Transpose and store in final format [1, 8, 1, 64]
        transposed_offset = col * 8 + row
        tl.store(output_ptr3 + transposed_offset * 8 + tl.arange(0, 8)[None, :], 
                acc.flatten()[offset.flatten()], mask=offset < 512)

@torch.fx.wrap
def fused_linear_view_transpose(hidden_states, weight, bias):
    # Get tensor shapes
    hidden_size = hidden_states.shape[-1]  # 512
    weight_out = weight.shape[0]  # 512
    
    # Allocate output tensors
    linear_output = torch.empty((1, 1, hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)
    reshaped_output = torch.empty((1, 1, 8, 64), dtype=hidden_states.dtype, device=hidden_states.device)
    transposed_output = torch.empty((1, 8, 1, 64), dtype=hidden_states.dtype, device=hidden_states.device)
    
    # Launch Triton kernel
    grid = lambda meta: (1, weight_out // 64, 1)  # 1 batch, num_output_cols // block_size
    
    # simplified kernel for the specific small sizes
    @triton.jit
    def simple_fusion_kernel(
        hidden_states_ptr,
        weight_ptr,
        bias_ptr,
        linear_out_ptr,
        reshaped_out_ptr,
        transposed_out_ptr,
        hidden_size: tl.constexpr,
        weight_out: tl.constexpr,
    ):
        pid = tl.program_id(0)
        
        if pid < weight_out:
            # Load bias for this output column
            bias_val = tl.load(bias_ptr + pid)
            
            # Compute linear output for this column
            acc = bias_val
            for k in range(0, hidden_size, 16):
                k_remaining = hidden_size - k
                k_block_size = min(16, k_remaining)
                
                hidden_vals = tl.load(hidden_states_ptr + k + tl.arange(0, k_block_size))
                weight_vals = tl.load(weight_ptr + pid * hidden_size + k + tl.arange(0, k_block_size))
                
                acc += tl.dot(hidden_vals, weight_vals)
            
            # Store linear output
            tl.store(linear_out_ptr + pid, acc)
            
            # For reshape and transpose, handle the 8x64 conversion
            hidden_idx = pid
            seq_idx = hidden_idx // 64
            head_idx = hidden_idx % 64
            
            # Store in reshaped format [1, 1, 8, 64]
            reshaped_offset = seq_idx * 64 + head_idx
            tl.store(reshaped_out_ptr + reshaped_offset, acc)
            
            # Store in transposed format [1, 8, 1, 64] 
            transposed_offset = head_idx * 8 + seq_idx
            tl.store(transposed_out_ptr + transposed_offset, acc)
    
    # Launch with appropriate grid size
    simple_fusion_kernel[(weight_out,)](
        hidden_states_ptr=hidden_states,
        weight_ptr=weight,
        bias_ptr=bias,
        linear_out_ptr=linear_output,
        reshaped_out_ptr=reshaped_output.view(-1),
        transposed_out_ptr=transposed_output.view(-1),
        hidden_size=hidden_size,
        weight_out=weight_out,
    )
    
    return linear_output, reshaped_output, transposed_output

def replacement_args(hidden_states, weight, bias):
    return (hidden_states, weight, bias)

def replacement_func():
    return fused_linear_view_transpose