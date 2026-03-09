import torch
import triton
import triton.language as tl

def pattern(self, x, weight, bias, eps):
    # Dropout + LayerNorm fusion pattern
    # tmp_3 = dropout(tmp_2, 0.1, False, False)
    tmp_3 = torch.nn.functional.dropout(x, 0.1, False, False)
    # tmp_4 = layer_norm(tmp_3, (64,), weight, bias, eps)
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (64,), weight, bias, eps)
    # Return both values as per the graph return: (tmp_3, tmp_4)
    return tmp_3, tmp_4

def replacement_args(x, weight, bias, eps):
    return (x, weight, bias, eps)

@triton.jit
def fused_dropout_layernorm_kernel_64(
    x_ptr, 
    weight_ptr, 
    bias_ptr, 
    eps,
    out_ptr,
    dropout_out_ptr,
    n_elements,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply dropout (p=0.1, training=False) -> no dropout applied during inference
    dropout_mask = 1.0  # Since training=False, dropout is disabled
    dropout_x = x * dropout_mask
    
    # Store dropout output
    tl.store(dropout_out_ptr + offsets, dropout_x, mask=mask)
    
    # Load layer norm parameters
    weight = tl.load(weight_ptr + (offsets % hidden_size), mask=mask, other=1.0)
    bias = tl.load(bias_ptr + (offsets % hidden_size), mask=mask, other=0.0)
    
    # Layer normalization computation
    mean = tl.sum(x, axis=0) / n_elements
    var = tl.sum((x - mean) * (x - mean), axis=0) / n_elements
    x_norm = (x - mean) * tl.math.rsqrt(var + eps)
    out = x_norm * weight + bias
    
    # Store layer norm output
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_dropout_layernorm_64(x, weight, bias, eps=1e-12):
    # Get input tensor info
    n_elements = x.numel()
    batch_size, seq_len = x.shape[0] if len(x.shape) > 2 else 1, x.shape[-2] if len(x.shape) > 2 else 1
    hidden_size = 64
    
    # Determine optimal block size based on tensor size
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensors
    dropout_out = torch.empty_like(x)
    layernorm_out = torch.empty_like(x)
    
    # Launch kernel
    fused_dropout_layernorm_kernel_64[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        eps=eps,
        out_ptr=layernorm_out,
        dropout_out_ptr=dropout_out,
        n_elements=n_elements,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return dropout_out, layernorm_out

def replacement_func():
    return fused_dropout_layernorm_64