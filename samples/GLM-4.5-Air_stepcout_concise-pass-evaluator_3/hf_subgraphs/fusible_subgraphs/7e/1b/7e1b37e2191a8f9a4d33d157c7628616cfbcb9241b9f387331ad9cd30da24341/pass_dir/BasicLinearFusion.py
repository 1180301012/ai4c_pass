import torch
import triton
import triton.language as tl

def pattern(weight, hidden_states, key_states):
    # Very simple pattern - just the linear operation
    result = torch.nn.functional.linear(hidden_states, weight, None)
    return result

def replacement_args(weight, hidden_states, key_states):
    return (weight, hidden_states)

@triton.jit
def basic_linear_kernel(
    weight_ptr,
    hidden_states_ptr,
    output_ptr,
    batch_size,
    seq_len,
    in_features,
    out_features,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * seq_len
    
    # Reshape for linear operation
    hidden_flat = tl.load(hidden_states_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Simple matrix multiplication approximation
    output_flat = tl.zeros([tl.sum(mask)], dtype=tl.float32)
    for k in range(in_features):
        weight_k = tl.load(weight_ptr + offsets[:, None] * in_features + k, mask=mask[:, None], other=0.0).to(tl.float32)
        hidden_k = tl.load(hidden_states_ptr + offsets[:, None] * in_features + k, mask=mask[:, None], other=0.0).to(tl.float32)
        output_flat += tl.sum(weight_k * hidden_k, axis=1)
    
    tl.store(output_ptr + offsets, output_flat, mask=mask)

@torch.fx.wrap
def basic_linear_fused(weight, hidden_states, key_states):
    # Get tensor shapes
    batch_size, seq_len, in_features = hidden_states.shape
    out_features = weight.shape[0]
    
    output = torch.empty((batch_size, seq_len, out_features), dtype=hidden_states.dtype, device=hidden_states.device)
    
    BLOCK_SIZE = 1024
    num_programs = (batch_size * seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    basic_linear_kernel[(num_programs,)](
        weight_ptr=weight,
        hidden_states_ptr=hidden_states,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        in_features=in_features,
        out_features=out_features,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return basic_linear_fused