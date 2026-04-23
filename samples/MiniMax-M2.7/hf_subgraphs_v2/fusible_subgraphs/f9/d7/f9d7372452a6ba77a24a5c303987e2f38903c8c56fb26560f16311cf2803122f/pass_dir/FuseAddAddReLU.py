import torch
import triton
import triton.language as tl

@triton.jit
def add_add_relu_kernel(
    in_0_ptr, in_2_ptr, in_3_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for:
    1. in_3 += in_0  (element-wise add)
    2. result += in_2  (element-wise add)
    3. relu(result)  (in-place ReLU)
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all three inputs
    in_0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_2 = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    in_3 = tl.load(in_3_ptr + offsets, mask=mask, other=0.0)
    
    # Perform the fused operations: in_3 + in_0 + in_2, then relu
    tmp = in_3 + in_0
    tmp = tmp + in_2
    out = tl.where(tmp > 0, tmp, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_add_add_relu(in_0, in_2, in_3):
    """
    Wrapper function that launches the fused add+add+relu kernel.
    
    Args:
        in_0: First input tensor [1, 128, 16, 12]
        in_2: Second input tensor [1, 128, 16, 12]
        in_3: Third input tensor [1, 128, 16, 12]
    
    Returns:
        ReLU(in_3 + in_0 + in_2)
    """
    # Flatten tensors for 1D grid
    in_0_flat = in_0.flatten()
    in_2_flat = in_2.flatten()
    in_3_flat = in_3.flatten()
    
    N = in_0_flat.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out_flat = torch.empty_like(in_0_flat)
    
    add_add_relu_kernel[(num_programs,)](
        in_0_ptr=in_0_flat,
        in_2_ptr=in_2_flat,
        in_3_ptr=in_3_flat,
        out_ptr=out_flat,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back to original shape
    return out_flat.view(in_0.shape)

def pattern(in_0, in_1, in_2, in_3):
    """
    Match the pattern:
    in_3 += in_0
    in_4 = in_3
    in_4 += in_2
    tmp_0 = in_4
    tmp_2 = relu(tmp_0)
    
    Returns (tmp_2, in_1) where tmp_2 is the relu output and in_1 is passed through
    """
    in_3_plus = in_3 + in_0
    in_4 = in_3_plus + in_2
    tmp_2 = torch.nn.functional.relu(in_4, inplace=False)
    return tmp_2, in_1

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

def replacement_func():
    return fused_add_add_relu