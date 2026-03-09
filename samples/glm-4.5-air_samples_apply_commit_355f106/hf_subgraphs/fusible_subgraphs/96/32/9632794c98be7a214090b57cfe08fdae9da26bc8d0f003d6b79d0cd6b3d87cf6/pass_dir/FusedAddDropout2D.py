import torch
import triton
import triton.language as tl

def pattern(in_4, in_3):
    # Match the exact pattern from the model:
    # tmp_3 = in_4 + in_3
    # tmp_4 = torch.nn.functional.dropout2d(tmp_3, 0.1, False, False)
    tmp_3 = in_4 + in_3
    tmp_4 = torch.nn.functional.dropout2d(tmp_3, 0.1, training=False, inplace=False)
    return tmp_4

def replacement_args(in_4, in_3):
    return (in_4, in_3)

@triton.jit
def fused_add_dropout_kernel(
    ptr1, ptr2, out_ptr,
    n_elements,
    p: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load both tensors
    x = tl.load(ptr1 + offsets, mask=mask, other=0.0)
    y = tl.load(ptr2 + offsets, mask=mask, other=0.0)
    
    # Element-wise addition: (x + y)
    sum_val = x + y
    
    # Apply dropout scaling: since training=False, we just scale by (1-p)
    dropout_scale = 1.0 - p
    out = sum_val * dropout_scale
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_add_dropout2d(tensor1, tensor2, p=0.5, training=True, inplace=False):
    """
    Fused addition followed by dropout2d operation
    Note: For training=False, dropout is just scaling by (1-p)
    """
    # Ensure tensors have the same shape
    if tensor1.shape != tensor2.shape:
        raise ValueError("Input tensors must have the same shape")
    
    # For training=True, we would need more complex handling, but our pattern matches training=False
    if training:
        raise NotImplementedError("Training mode not implemented in this fused pass")
    
    # Use the larger tensor size
    n_elements = tensor1.numel()
    
    # Optimized block size for tensor sizes 64x64
    if tensor1.numel() >= 4096:  # 64x64 or larger
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 512
    
    # Calculate number of programs needed
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(tensor1)
    
    # Launch fused kernel
    fused_add_dropout_kernel[(num_programs,)](
        ptr1=tensor1,
        ptr2=tensor2,
        out_ptr=output,
        n_elements=n_elements,
        p=p,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_add_dropout2d