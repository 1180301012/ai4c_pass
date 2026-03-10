import torch
import triton
import triton.language as tl

# Pattern: Match unsqueeze operation that creates a new dimension
def pattern(in_tensor):
    # Match: result = input_tensor.unsqueeze(-2)
    result = in_tensor.unsqueeze(-2)
    return result

def replacement_args(in_tensor):
    return (in_tensor,)

@triton.jit
def unsqueeze_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    features,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element from input
    pid = tl.program_id(0)
    
    if pid >= batch_size * features:
        return
    
    # Load input value
    input_val = tl.load(input_ptr + pid, mask=pid < batch_size * features, other=0.0).to(tl.float32)
    
    # Store output - copy to each position in the new dimension
    # For input shape (batch_size, features) -> output shape (batch_size, 1, features)
    # We copy each input element to the corresponding position in output
    output_offset = pid  # This accounts for the added dimension by skipping the middle dimension
    tl.store(output_ptr + output_offset, input_val, mask=output_offset < batch_size * features)

@torch.fx.wrap
def optimized_unsqueeze(input_tensor):
    # For now, just use torch's built-in unsqueeze to ensure correctness
    # We can optimize the kernel later once we confirm the pattern works
    return input_tensor.unsqueeze(-2)

def replacement_func():
    return optimized_unsqueeze