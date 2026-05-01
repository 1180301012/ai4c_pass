import torch
import triton
import triton.language as tl

# Pattern matching

def pattern(in_0):
    softmax_out = torch.nn.functional.softmax(in_0, dim=1)
    linspace_vec = torch.linspace(0, 4, steps=5, device=in_0.device)
    mult_result = softmax_out * linspace_vec
    sum_result = mult_result.sum(dim=1)
    return 5 - sum_result

# Argument extraction

def replacement_args(in_0):
    return (in_0,)

# Triton kernel
@triton.jit
def weighted_softmax_kernel(x_ptr, output_ptr, batch_size, inner_size: tl.constexpr = 5):
    # Each block processes one batch element
    batch_idx = tl.program_id(0)
    elem_idx = tl.thread_id(0)
    if elem_idx >= inner_size:
        return

    # Load element
    x = tl.load(x_ptr + batch_idx * inner_size + elem_idx)
    exp_x = tl.exp(x)
    
    # Compute weighted and denominator components
    numerator_elem = exp_x * elem_idx
    denominator_elem = exp_x

    # Shared memory for reduction
    shared_mem = tl.shared_memory(shape=(2,), dtype=tl.float32)
    if tl.thread_id(0) == 0:
        shared_mem[0] = 0.0  # numerator
        shared_mem[1] = 0.0  # denominator
    tl.sync()

    # Accumulate results
    tl.atomic_add(shared_mem + 0, numerator_elem)
    tl.atomic_add(shared_mem + 1, denominator_elem)
    tl.sync()

    # Thread 0 computes result
    if tl.thread_id(0) == 0:
        numerator = shared_mem[0]
        denominator = shared_mem[1]
        y = numerator / denominator if denominator != 0.0 else 0.0
        output_val = 5.0 - y
        tl.store(output_ptr + batch_idx, output_val)


# Kernel wrapper
@torch.fx.wrap
def kernel_wrapper(in_0):
    batch_size = in_0.shape[0]
    # Convert to float32 for precision
    in_0_fp32 = in_0.float()
    output_fp32 = torch.empty((batch_size,), dtype=torch.float32, device=in_0.device)
    
    # Launch kernel
    grid = (batch_size,)
    weighted_softmax_kernel[grid](
        x_ptr=in_0_fp32,
        output_ptr=output_fp32,
        batch_size=batch_size,
        inner_size=5
    )
    
    # Convert back to original dtype
    return output_fp32.to(in_0.dtype)

# Replacement function

def replacement_func():
    return kernel_wrapper