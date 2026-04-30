import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """
    Match the pattern: conv2d + add + permute + contiguous + view
    This is the pattern found in Nystromformer models.
    
    Args:
        in_0: conv weight [groups, 1, kH, kW] - typically [4 or 12, 1, 65, 1]
        in_1: context_layer [batch, groups, seq_len, head_dim] - mutable via +=
        in_2: value_layer [batch, groups, seq_len, head_dim]
    
    Returns:
        Final reshaped output tensor
    """
    # Depthwise conv2d: groups parameter varies (4 or 12)
    # stride=(1, 1), padding=(32, 0), dilation=(1, 1)
    conv_out = torch.conv2d(in_2, in_0, None, (1, 1), (32, 0), (1, 1), 4)
    
    # In-place add
    in_1 += conv_out
    
    # Permute: (B, G, S, H) -> (B, S, G, H)
    permuted = in_1.permute(0, 2, 1, 3)
    
    # Make contiguous
    cont = permuted.contiguous()
    
    # Get output shape from shape metadata - will be passed via replacement_args
    # The shape depends on the specific graph, extracted at runtime
    return cont


def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments needed for the replacement kernel.
    
    Returns:
        Tuple of (in_0, in_1, in_2, groups, out_batch, out_seq, out_dim)
    """
    groups = in_0.shape[0]  # First dim of weight is the groups count
    batch, g, seq, head = in_1.shape
    out_dim = g * head  # Final dim after fusion: groups * head_dim
    return (in_0, in_1, in_2, groups, batch, seq, out_dim)


@triton.jit
def fused_conv2d_add_permute_view_kernel(
    # Pointers
    weight_ptr, context_ptr, value_ptr, output_ptr,
    # Tensor dimensions
    batch: tl.constexpr, seq: tl.constexpr, head: tl.constexpr, groups: tl.constexpr,
    kernel_size: tl.constexpr,
    # Strides
    weight_batch_stride, weight_kH_stride, weight_kW_stride,
    context_batch_stride, context_groups_stride, context_seq_stride, context_head_stride,
    value_batch_stride, value_groups_stride, value_seq_stride, value_head_stride,
    output_batch_stride, output_seq_stride, output_dim_stride,
    # Meta
    OUT_BATCH: tl.constexpr, OUT_SEQ: tl.constexpr, OUT_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: depthwise conv2d + in-place add + permute + view
    
    The computation:
    1. For each (b, g, s, h) in context, we need to compute:
       context[b,g,s,h] += sum over kH of weight[g,0,kH,0] * value[b,g,s+kH-32,kH]
       (with padding considerations)
    
    2. Then reshape from (B, S, G, H) to (B, S, G*H)
    
    This is a depthwise convolution with padding=(32,0), stride=(1,1)
    """
    # Program ID for the output element
    pid = tl.program_id(0)
    num_elements = OUT_BATCH * OUT_SEQ * OUT_DIM
    
    if pid >= num_elements:
        return
    
    # Decode the output position
    out_idx = pid
    b = out_idx // (OUT_SEQ * OUT_DIM)
    out_idx = out_idx % (OUT_SEQ * OUT_DIM)
    s = out_idx // OUT_DIM
    d = out_idx % OUT_DIM
    
    # d = g * head_dim + h
    g = d // head
    h = d % head
    
    # Get initial context value for this position
    context_idx = b * context_batch_stride + g * context_groups_stride + s * context_seq_stride + h * context_head_stride
    acc = tl.load(context_ptr + context_idx).to(tl.float32)
    
    # Depthwise conv: for each position in the kernel (kH)
    # With padding=(32,0), value index is s + kH - 32
    # Valid when 0 <= s + kH - 32 < seq, i.e., 32-kH <= s < seq+32-kH
    for kH in range(kernel_size):
        weight_idx = g * weight_batch_stride + kH * weight_kH_stride
        w_val = tl.load(weight_ptr + weight_idx)
        
        # Input position in value tensor
        v_s = s + kH - 32
        if 0 <= v_s and v_s < seq:
            v_idx = b * value_batch_stride + g * value_groups_stride + v_s * value_seq_stride + h * value_head_stride
            v_val = tl.load(value_ptr + v_idx)
            acc += w_val * v_val
    
    # Store result: after permute (B, S, G, H) and view to (B, S, G*H)
    out_idx_final = b * output_batch_stride + s * output_seq_stride + d * output_dim_stride
    tl.store(output_ptr + out_idx_final, acc)


def fused_conv2d_add_permute_view(weight, context, value, groups, out_batch, out_seq, out_dim):
    """
    Wrapper function that launches the fused kernel.
    
    Args:
        weight: conv weight tensor [groups, 1, kH, 1]
        context: context tensor [batch, groups, seq, head_dim] - modified in place
        value: value tensor [batch, groups, seq, head_dim]
        groups: number of depthwise groups
        out_batch, out_seq, out_dim: output shape
    
    Returns:
        Output tensor [out_batch, out_seq, out_dim]
    """
    batch, g, seq, head = context.shape
    kernel_size = weight.shape[2]  # kH from weight [groups, 1, kH, 1]
    
    # Allocate output
    output = torch.empty((out_batch, out_seq, out_dim), dtype=context.dtype, device=context.device)
    
    # Calculate strides
    weight_batch_stride = weight.stride(0)
    weight_kH_stride = weight.stride(2)
    weight_kW_stride = weight.stride(3)
    
    context_batch_stride = context.stride(0)
    context_groups_stride = context.stride(1)
    context_seq_stride = context.stride(2)
    context_head_stride = context.stride(3)
    
    value_batch_stride = value.stride(0)
    value_groups_stride = value.stride(1)
    value_seq_stride = value.stride(2)
    value_head_stride = value.stride(3)
    
    output_batch_stride = output.stride(0)
    output_seq_stride = output.stride(1)
    output_dim_stride = output.stride(2)
    
    # Grid: one program per output element
    num_elements = out_batch * out_seq * out_dim
    grid = (num_elements,)
    
    fused_conv2d_add_permute_view_kernel[grid](
        weight, context, value, output,
        batch, seq, head, groups,
        kernel_size,
        weight_batch_stride, weight_kH_stride, weight_kW_stride,
        context_batch_stride, context_groups_stride, context_seq_stride, context_head_stride,
        value_batch_stride, value_groups_stride, value_seq_stride, value_head_stride,
        output_batch_stride, output_seq_stride, output_dim_stride,
        out_batch, out_seq, out_dim,
        BLOCK_SIZE=64,
    )
    
    return output


@torch.fx.wrap
def kernel_wrapper(weight, context, value, groups, out_batch, out_seq, out_dim):
    """
    FX wrap for the fused kernel.
    """
    return fused_conv2d_add_permute_view(weight, context, value, groups, out_batch, out_seq, out_dim)


def replacement_func():
    """
    Returns the replacement function.
    """
    return kernel_wrapper