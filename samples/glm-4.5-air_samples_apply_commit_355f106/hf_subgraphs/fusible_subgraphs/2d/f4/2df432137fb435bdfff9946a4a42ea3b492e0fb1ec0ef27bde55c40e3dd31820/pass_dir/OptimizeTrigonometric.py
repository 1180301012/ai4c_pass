import torch
import triton
import triton.language as tl
import math

def pattern(input_tensor):
    # The pattern: compute both cos and sin of the same input
    cos_result = input_tensor.cos()
    sin_result = input_tensor.sin()
    return cos_result, sin_result

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def trig_kernel(
    input_ptr,
    cos_ptr, sin_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute both trig functions
    cos_val = tl.math.cos(x)
    sin_val = tl.math.sin(x)
    
    # Store results
    tl.store(cos_ptr + offsets, cos_val, mask=mask)
    tl.store(sin_ptr + offsets, sin_val, mask=mask)

@triton.jit
def trig_kernel_autotuned(
    input_ptr,
    cos_ptr, sin_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute both trig functions
    cos_val = tl.math.cos(x)
    sin_val = tl.math.sin(x)
    
    # Store results
    tl.store(cos_ptr + offsets, cos_val, mask=mask)
    tl.store(sin_ptr + offsets, sin_val, mask=mask)

@torch.fx.wrap
def optimized_trigonometric(input_tensor):
    n_elements = input_tensor.numel()
    
    # Create output tensors
    cos_out = torch.empty_like(input_tensor)
    sin_out = torch.empty_like(input_tensor)
    
    # Use autotuning to find optimal block size
    @triton.autotune(
        configs=[
            triton.Config(num_warps=1, num_stages=1),
            triton.Config(num_warps=2, num_stages=1),
            triton.Config(num_warps=4, num_stages=1),
            triton.Config(num_warps=8, num_stages=1),
            triton.Config(num_warps=1, num_stages=2),
            triton.Config(num_warps=2, num_stages=2),
            triton.Config(num_warps=4, num_stages=2),
            triton.Config(num_warps=8, num_stages=2),
        ],
        key=['n_elements'],
    )
    @triton.heuristics({
        'BLOCK_SIZE': lambda args: 1024 if args['n_elements'] > 8192 else 512,
    })
    def autotuned_kernel(
        input_ptr,
        cos_ptr, sin_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        num_warps: tl.constexpr,
        num_stages: tl.constexpr,
    ):
        trig_kernel_autotuned[input_pid](
            input_ptr, cos_ptr, sin_ptr, n_elements, BLOCK_SIZE
        )
    
    # Set up launch parameters
    grid = (n_elements + 1023) // 1024,
    
    # Configure autotuned kernel
    class AutotunedKernelContext:
        def __init__(self, input_t, cos_out_t, sin_out_t):
            self.input = input_t
            self.cos_out = cos_out_t  
            self.sin_out = sin_out_t
            self.n_elements = n_elements
            
        def run(self):
            autotuned_kernel[(grid,)](
                self.input, self.cos_out, self.sin_out,
                self.n_elements,
                BLOCK_SIZE=1024,
                num_warps=8,
                num_stages=2
            )
    
    ctx = AutotunedKernelContext(input_tensor, cos_out, sin_out)
    ctx.run()
    
    return cos_out, sin_out

# Simpler version without complex autotuning for more predictable performance
@torch.fx.wrap  
def simple_optimized_trigonometric(input_tensor):
    n_elements = input_tensor.numel()
    
    # Create output tensors  
    cos_out = torch.empty_like(input_tensor)
    sin_out = torch.empty_like(input_tensor)
    
    # Choose block size based on tensor size
    BLOCK_SIZE = 1024 if n_elements > 8192 else 512
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    trig_kernel[grid_size](
        input_tensor, cos_out, sin_out,
        n_elements, BLOCK_SIZE
    )
    
    return cos_out, sin_out

def replacement_func():
    return simple_optimized_trigonometric