import torch
import triton
import triton.language as tl

@triton.jit
def view_roll_kernel_64_64_384(
    input_ptr,
    output_ptr,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for 64x64 spatial dimensions with 384 features.
    """
    pid = tl.program_id(0)
    total_elements = batch_size * 64 * 64 * 384
    block_size = BLOCK_SIZE
    
    start_idx = pid * block_size
    end_idx = min(start_idx + block_size, total_elements)
    
    for idx in range(start_idx, end_idx):
        # Calculate dimension indices
        fid = idx % 384
        residual = idx // 384
        w_idx = residual % 64
        residual //= 64
        h_idx = residual % 64
        b_idx = residual // 64
        
        # Apply roll (4,4)
        rolled_w = (w_idx - 4) % 64
        rolled_h = (h_idx - 4) % 64
        
        # Calculate output index
        out_idx = ((b_idx * 64 + rolled_h) * 64 + rolled_w) * 384 + fid
        
        # Load and store
        input_val = tl.load(input_ptr + idx, other=0.0)
        tl.store(output_ptr + out_idx, input_val)

@torch.fx.wrap
def fused_view_roll_64_64_384(input_tensor):
    """Fused view + roll for 64x64 spatial with 384 features."""
    batch_size = input_tensor.shape[0]
    output = torch.empty_like(input_tensor)
    
    total_elements = batch_size * 64 * 64 * 384
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    view_roll_kernel_64_64_384[(num_programs,)](
        input_tensor,
        output,
        batch_size,
        BLOCK_SIZE
    )
    
    return output

def pattern(tmp_2):
    tmp_3 = tmp_2.view(-1, 64, 64, 384)
    tmp_4 = torch.roll(tmp_3, shifts=(4, 4), dims=(1, 2))
    return tmp_3, tmp_4

def replacement_args(tmp_2):
    return (tmp_2,)

def replacement_func():
    def fused_kernel_wrapper(tmp_2):
        return fused_view_roll_64_64_384(tmp_2)
    return fused_kernel_wrapper