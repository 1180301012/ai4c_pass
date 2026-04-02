import torch
import triton
import triton.language as tl

def pattern(in_4):
    # Key states processing pattern: view -> transpose -> contiguous
    tmp_3 = in_4.view(1, 1, -1, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_9 = tmp_4.contiguous()
    return tmp_9

def replacement_args(in_4):
    return (in_4,)

@triton.jit
def optimized_key_states_kernel(
    key_ptr,
    output_ptr,
    n_keys,
    n_heads,
    head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each program handles one head
    if pid >= n_heads:
        return
    
    start_idx = pid * head_dim
    end_idx = start_idx + head_dim
    
    # Load key data for this head
    key_offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = key_offsets < end_idx
    
    # Load key data (originally [1, 1, 512] -> reshaped to [1, 8, 64])
    # We only need the head_dim part for our head
    key_data = tl.load(key_ptr + key_offsets, mask=mask, other=0.0)
    
    # Store directly in output layout (contiguous [1, 8, 1, 64])
    output_offsets = pid * head_dim + tl.arange(0, BLOCK_SIZE)
    output_mask = output_offsets < (n_heads * head_dim)
    
    tl.store(output_ptr + output_offsets, key_data, mask=output_mask)

@torch.fx.wrap
def optimized_key_states_processing(key):
    # key shape: [1, 1, 512]
    # output shape: [1, 8, 1, 64] contiguous
    
    n_keys = key.shape[-1]
    n_heads = n_keys // 64  # 512 // 64 = 8
    head_dim = 64
    
    # Reshape input from [1, 1, 512] to effectively [1, 1, 8, 64] in memory
    key_reshaped = key.view(1, 1, n_heads, head_dim)
    
    # Transpose from [1, 1, 8, 64] to [1, 8, 1, 64]
    key_transposed = key_reshaped.transpose(1, 2)
    
    # Output buffer (this already gives us contiguous layout)
    output = torch.empty((1, n_heads, 1, head_dim), dtype=key.dtype, device=key.device)
    
    # Launch kernel
    BLOCK_SIZE = 256  # Must be power of 2 for efficient memory access
    grid = (n_heads * ((head_dim + BLOCK_SIZE - 1) // BLOCK_SIZE),)
    
    if n_heads * head_dim > 0:
        optimized_key_states_kernel[grid](
            key_transposed,
            output,
            n_keys,
            n_heads,
            head_dim,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        output = key_transposed.contiguous()
    
    return output

def replacement_func():
    return optimized_key_states_processing