import torch
import triton
import triton.language as tl

def pattern(x):
    # Match GELU followed by two reshape operations
    tmp_0 = torch.nn.functional.gelu(x)
    tmp_1 = tmp_0.reshape(1, 124, 2, 768)
    tmp_2 = tmp_1.reshape(1, 248, 768)
    return tmp_2

def replacement_args(x):
    return (x,)

def replacement_func():
    @triton.jit
    def fused_gelu_reshape_kernel(
        x_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        # Each program handles a contiguous block of data
        block_start = tl.program_id(0) * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load input data
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # Apply GELU operation using Triton
        # GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x_sq = x * x
        x_cubed = x_sq * x
        inner = x + 0.044715 * x_cubed
        tanh_arg = tl.sqrt(2.0 / tl.math.pi) * inner
        tanh_result = tl.tanh(tanh_arg)
        gelu_result = x * 0.5 * (1.0 + tanh_result)
        
        # Store output data (reshape is just a view change, so no layout change needed)
        tl.store(out_ptr + offsets, gelu_result, mask=mask)
    
    @torch.fx.wrap
    def fused_gelu_reshape(x):
        # Input shape: [1, 124, 1536]
        # Output shape: [1, 248, 768]
        N = x.numel()
        BLOCK_SIZE = 1024
        num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        out = torch.empty(1, 248, 768, dtype=x.dtype, device=x.device)
        
        fused_gelu_reshape_kernel[(num_programs,)](
            x_ptr=x,
            out_ptr=out,
            n_elements=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return out
    
    return fused_gelu_reshape