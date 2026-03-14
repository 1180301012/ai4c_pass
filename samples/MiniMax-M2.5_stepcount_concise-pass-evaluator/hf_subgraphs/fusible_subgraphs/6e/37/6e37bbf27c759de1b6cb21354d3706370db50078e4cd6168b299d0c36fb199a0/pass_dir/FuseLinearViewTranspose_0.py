import torch
import triton
import triton.language as tl


@triton.jit
def fused_linear_view_transpose_kernel_0(
    hidden_ptr, weight_ptr, output_ptr,
    batch_size, seq_len, num_heads, head_dim,
    hidden_stride, weight_stride, output_stride,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid: (batch_size * num_heads * seq_len)
    pid = tl.program_id(0)
    
    # Calculate position indices
    head_idx = pid % num_heads
    remaining = pid // num_heads
    seq_idx = remaining % seq_len
    batch_idx = remaining // seq_len
    
    # Load hidden tensor row offset
    hidden_offset = batch_idx * hidden_stride[0] + seq_idx * hidden_stride[1]
    
    # Compute the output feature for this head
    out_feature_idx = head_idx
    
    # Weight is [512, 2048], we need row out_feature_idx: [2048]
    weight_offset = out_feature_idx * weight_stride[0]
    
    # Accumulator for the matmul result
    acc = tl.zeros((head_dim,), dtype=tl.float32)
    
    # Perform the matrix multiplication
    for k in range(0, 2048, BLOCK_SIZE):
        k_offsets = k + tl.arange(0, BLOCK_SIZE)
        mask = k_offsets < 2048
        
        hidden_vals = tl.load(
            hidden_ptr + hidden_offset + k_offsets,
            mask=mask,
            other=0.0
        )
        
        weight_vals = tl.load(
            weight_ptr + weight_offset + k_offsets,
            mask=mask,
            other=0.0
        )
        
        acc += hidden_vals * weight_vals
    
    # Store result
    out_base = batch_idx * output_stride[0] + head_idx * output_stride[1] + seq_idx * output_stride[2]
    tl.store(output_ptr + out_base + tl.arange(0, head_dim), acc)


@torch.fx.wrap
def fused_linear_view_transpose_kernel_wrapper_0(hidden_states, weight, output_shape, cos_in, sin_in):
    """
    Fused kernel for graph 0: in_2 [1, 64, 2048] -> view(1, 64, -1, 128) -> transpose -> [1, 4, 64, 128]
    """
    batch, seq, hidden_dim = hidden_states.shape
    num_heads = output_shape[1]
    head_dim = output_shape[3]
    
    output = torch.empty(output_shape, dtype=hidden_states.dtype, device=hidden_states.device)
    
    output_stride = output.stride()
    hidden_stride = hidden_states.stride()
    weight_stride = weight.stride()
    
    grid = (batch * num_heads * seq,)
    BLOCK_SIZE = 128
    
    fused_linear_view_transpose_kernel_0[grid](
        hidden_states, weight, output,
        batch, seq, num_heads, head_dim,
        hidden_stride, weight_stride, output_stride,
        BLOCK_SIZE,
    )
    
    cos_out = cos_in.unsqueeze(1)
    sin_out = sin_in.unsqueeze(1)
    
    return cos_out, sin_out, output


def pattern(in_0, in_1, in_2, in_3):
    """
    Match the pattern: linear + view + transpose
    Graph 0 variant: in_2 shape [1, 64, 2048] -> view(1, 64, -1, 128) -> transpose
    """
    tmp_1 = torch.nn.functional.linear(in_2, in_0, None)
    tmp_2 = tmp_1.view((1, 64, -1, 128))
    tmp_3 = tmp_2.transpose(1, 2)
    tmp_4 = in_1.unsqueeze(1)
    tmp_5 = in_3.unsqueeze(1)
    return tmp_4, tmp_5, tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    batch = in_2.shape[0]
    seq = in_2.shape[1]
    num_heads = 4
    head_dim = 128
    output_shape = (batch, num_heads, seq, head_dim)
    return (in_2, in_0, output_shape, in_1, in_3)


def replacement_func():
    return fused_linear_view_transpose_kernel_wrapper_0