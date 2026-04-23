import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Simple add kernel for element-wise addition.
    """
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    result = x + y
    tl.store(output_ptr + offsets, result, mask=mask)


def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern for ViT model: add + reshape(-1, 768) + layer_norm(768)
    """
    tmp_2 = in_2 + in_3
    tmp_3 = tmp_2.reshape(-1, 768)
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (768,), in_1, in_0, 1e-05)
    return tmp_3, tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@torch.fx.wrap
def optimized_add_reshape_layernorm(in_0, in_1, in_2, in_3):
    """
    Optimized version:
    1. Use Triton for the element-wise add operation
    2. Then use PyTorch for reshape + layer_norm (which torch.compile will optimize)
    """
    # Get shapes
    batch_size, seq_len, hidden_dim = in_2.shape
    n_elements = in_2.numel()
    
    # Flatten for the add kernel
    in_2_flat = in_2.flatten()
    in_3_flat = in_3.flatten()
    
    # Allocate output for add
    add_output = torch.empty_like(in_2_flat)
    
    # Launch add kernel
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    add_kernel[(num_programs,)](in_2_flat, in_3_flat, add_output, n_elements, BLOCK_SIZE)
    
    # Reshape the result
    tmp_3 = add_output.reshape(-1, 768)
    
    # Layer norm
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (768,), in_1, in_0, 1e-05)
    
    return tmp_3, tmp_4


def replacement_func():
    return optimized_add_reshape_layernorm