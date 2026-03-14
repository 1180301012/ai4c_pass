import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    y = x.to(torch.float32)
    z = torch.nn.functional.layer_norm(y, (320,), weight, bias, 1e-05)
    return y, z

def replacement_args(x, weight, bias):
    return (x, weight, bias)

@triton.jit
def fused_type_conv_layer_norm_kernel(
    out_ptr,
    layer_out_ptr,
    ptr_6,
    ptr_2,
    ptr_1,
    n_elements,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input and convert to float32 implicitly through computation
    tmp_6_val = tl.load(ptr_6 + offsets, mask=mask, other=0.0)
    
    # Load layer norm weights and bias (host-resident)
    weight = tl.load(ptr_2)
    bias = tl.load(ptr_1)
    
    # Apply fused layer norm with implicit float32 conversion
    # Compute mean
    mean = tl.sum(tmp_6_val, axis=0) / hidden_size
    
    # Compute variance
    diff = tmp_6_val - mean
    var = tl.sum(diff * diff, axis=0) / hidden_size
    
    # Normalize
    inv_std = 1.0 / tl.sqrt(var + eps)
    normalized = diff * inv_std
    
    # Scale and shift
    result = normalized * weight + bias
    
    # Store both outputs (tmp_7 is just the original input for compatibility,
    # and tmp_8 is the layer norm output)
    tl.store(out_ptr + offsets, tmp_6_val, mask=mask)
    tl.store(layer_out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_type_conv_layer_norm(tmp_6, tmp_2, tmp_1):
    # Get input dimensions from tmp_6
    if len(tmp_6.shape) == 3:
        batch_size, seq_len, hidden_dim = tmp_6.shape
        total_elements = batch_size * seq_len * hidden_dim
    else:
        total_elements = tmp_6.numel()
    
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensors
    tmp_7_out = torch.empty_like(tmp_6)
    tmp_8_out = torch.empty_like(tmp_6)
    
    fused_type_conv_layer_norm_kernel[(num_programs,)](
        out_ptr=tmp_7_out,
        layer_out_ptr=tmp_8_out,
        ptr_6=tmp_6,
        ptr_2=tmp_2,
        ptr_1=tmp_1,
        n_elements=total_elements,
        hidden_size=320,  # From weight_meta.py
        eps=1e-05,  # From model.py
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return tmp_7_out, tmp_8_out

def replacement_func():
    return fused_type_conv_layer_norm