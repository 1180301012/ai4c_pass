import torch
import triton
import triton.language as tl

def pattern(in_2, in_3, weight, bias, in_4, eps):
    # Match the complete computation pattern from the graph
    tmp_2 = in_2 + in_3
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (256,), weight, bias, eps)
    tmp_4 = tmp_3 + in_4
    return tmp_3, tmp_4

def replacement_args(in_2, in_3, weight, bias, in_4, eps):
    return (in_2, in_3, weight, bias, in_4, eps)

@triton.jit
def fused_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    z_ptr,
    out_ptr,
    out2_ptr,
    eps,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    feature_size: tl.constexpr,
):
    # Each program handles one sequence position
    pid = tl.program_id(0)
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    
    if batch_idx >= batch_size or seq_idx >= seq_len:
        return
    
    # Step 1: Perform addition x + y for this sequence position
    sum_x = 0.0
    sum_x2 = 0.0
    
    # Compute mean and variance over the feature dimension
    for i in range(feature_size):
        offset = pid * feature_size + i
        x_val = tl.load(x_ptr + offset)
        y_val = tl.load(y_ptr + offset)
        sum_val = x_val + y_val
        sum_x += sum_val
        sum_x2 += sum_val * sum_val
    
    mean = sum_x / feature_size
    var = (sum_x2 / feature_size) - (mean * mean)
    var = tl.maximum(var, 1e-8)
    
    # Step 2: Apply LayerNorm and add z
    for i in range(feature_size):
        offset = pid * feature_size + i
        x_val = tl.load(x_ptr + offset)
        y_val = tl.load(y_ptr + offset)
        weight_val = tl.load(weight_ptr + i)
        bias_val = tl.load(bias_ptr + i)
        z_val = tl.load(z_ptr + offset)
        
        # Addition + LayerNorm
        sum_val = x_val + y_val
        normalized = (sum_val - mean) / tl.sqrt(var + eps)
        output1 = normalized * weight_val + bias_val
        
        # Final addition
        output2 = output1 + z_val
        
        # Store both outputs
        tl.store(out_ptr + offset, output1)
        tl.store(out2_ptr + offset, output2)

@torch.fx.wrap  
def fused_operation(x, y, weight, bias, z, eps=1e-05):
    batch_size, seq_len, feature_size = x.shape
    
    # Total sequence positions
    num_programs = batch_size * seq_len
    
    # Create output tensors
    out1 = torch.empty_like(x)  # This is tmp_3 (LayerNorm output)
    out2 = torch.empty_like(x)  # This is tmp_4 (final addition)
    
    # Launch the fused kernel
    fused_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        weight_ptr=weight,
        bias_ptr=bias,
        z_ptr=z,
        out_ptr=out1,
        out2_ptr=out2,
        eps=eps,
        batch_size=batch_size,
        seq_len=seq_len,
        feature_size=feature_size,
    )
    
    return out1, out2

def replacement_func():
    return fused_operation