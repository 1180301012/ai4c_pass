import torch
import triton
import triton.language as tl

def pattern(in_2, in_1, in_0):
    tmp_2 = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = tmp_2.view(1, 1, -1)
    tmp_4 = tmp_3.softmax(dim=-1)
    return tmp_4

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

@triton.jit
def conv2d_kernel(input_ptr, weight_ptr, bias_ptr, output_ptr, \
                 batch, channels, h_in, w_in, h_out, w_out, \
                 BLOCK_SIZE: tl.constexpr):
    # Calculate index for current output position
    pid = tl.program_id(0)
    num_elements = h_out * w_out
    batch_idx = pid // num_elements
    pos = pid % num_elements
    h = pos // w_out
    w = pos % w_out

    # Compute convolution for this output position
    val = tl.zeros((1,), dtype=tl.float32)
    for c in range(channels):
        # Calculate input index
        input_idx = batch_idx * channels * h_in * w_in + c * h_in * w_in + h * w_in + w
        input_val = tl.load(input_ptr + input_idx)
        # Load weight (1x1 kernel)
        weight_val = tl.load(weight_ptr + c)
        val += input_val * weight_val
    # Add bias
    bias_val = tl.load(bias_ptr)
    val += bias_val
    
    # Store result in output position (flattened)
    output_idx = batch_idx * (h_out * w_out) + pos
    tl.store(output_ptr + output_idx, val)

@triton.jit
def softmax_kernel(input_ptr, output_ptr, batch, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Process all elements for this batch
    start_idx = pid * n_elements
    end_idx = (pid + 1) * n_elements
    
    # Load all elements for this batch
    vals = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(0, n_elements, BLOCK_SIZE):
        idx = start_idx + i
        mask = (idx < end_idx)
        vals = tl.load(input_ptr + idx, mask=mask, other=-float('inf'))
        
    # Compute max
    max_val = tl.max(vals)
    # Subtract max and exponentiate
    exp_vals = tl.exp(vals - max_val)
    # Compute sum
    sum_val = tl.sum(exp_vals)
    # Normalize
    output_vals = exp_vals / sum_val
    
    # Store result
    tl.store(output_ptr + start_idx, output_vals, mask=mask)

@torch.fx.wrap
def fused_conv_softmax(in_2, in_1, in_0):
    batch, channels, h_in, w_in = in_2.shape
    h_out, w_out = h_in, w_in
    n_elements = h_out * w_out
    
    # Allocate output for convolution (flattened)
    conv_out = torch.empty((batch, n_elements), dtype=in_2.dtype, device=in_2.device)
    
    # Convolution parameters
    BLOCK_SIZE = 1024
    num_blocks = (batch * n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Execute convolution
    conv2d_kernel[(num_blocks,)](
        input_ptr=in_2, 
        weight_ptr=in_1.squeeze(0).view(-1),
        bias_ptr=in_0, 
        output_ptr=conv_out,
        batch=batch,
        channels=channels,
        h_in=h_in,
        w_in=w_in,
        h_out=h_out,
        w_out=w_out,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Execute softmax
    softmax_out = torch.empty_like(conv_out)
    softmax_kernel[(batch,)](
        input_ptr=conv_out, 
        output_ptr=softmax_out,
        batch=batch,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape to [batch, 1, n_elements]
    return softmax_out.view(batch, 1, n_elements)

def replacement_func():
    return fused_conv_softmax