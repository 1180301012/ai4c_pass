import torch
import triton
import triton.language as tl

@triton.jit
def fused_linear_dropout_transpose_kernel(
    input_ptr, weight_ptr, bias_ptr,
    output_dropout_ptr, output_transpose_ptr,
    B: tl.constexpr, S: tl.constexpr, D_in: tl.constexpr, D_out: tl.constexpr,
    dropout_p: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused kernel for: linear + dropout + transpose
    Input: [B, S, D_in]
    Weight: [D_out, D_in] 
    Bias: [D_out]
    Output (dropout): [B, S, D_out]
    Output (transpose): [B, D_out, S]
    """
    # Get program IDs
    batch_pid = tl.program_id(0)
    out_feature_pid = tl.program_id(1)
    
    # Calculate output offsets for this program
    # Each program computes one output feature across all batches
    # Output layout: [B, D_out, S] for transposed, [B, S, D_out] for dropout output
    
    # Offsets for reading weight (one row per program)
    weight_offsets = out_feature_pid * D_in + tl.arange(0, BLOCK_SIZE_K)
    weight_mask = weight_offsets < D_out * D_in
    
    # Offsets for writing outputs
    # For dropout output: [B, S, D_out] -> offset = b * S * D_out + s * D_out + d
    # For transpose output: [B, D_out, S] -> offset = b * D_out * S + d * S + s
    
    # Process all batches and sequences
    for batch in range(B):
        for seq in range(S):
            # Compute the linear output for this (batch, seq, out_feature)
            # We need to compute: sum over k of input[b, s, k] * weight[out_feature, k]
            
            # Input offsets for this batch, sequence
            input_offsets = batch * S * D_in + seq * D_in + tl.arange(0, BLOCK_SIZE_K)
            input_mask = input_offsets < B * S * D_in
            
            # Load weight row for this output feature
            w_offsets = out_feature_pid * D_in + tl.arange(0, BLOCK_SIZE_K)
            w_mask = w_offsets < D_out * D_in
            weight = tl.load(weight_ptr + w_offsets, mask=w_mask, other=0.0)
            
            # Load input vector
            inp = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
            
            # Compute matmul: sum over k of inp[k] * weight[k]
            # Using reduce for accumulation
            logits = tl.sum(inp * weight)
            
            # Add bias if present (bias_ptr is not None)
            if bias_ptr is not None:
                bias = tl.load(bias_ptr + out_feature_pid)
                logits = logits + bias
            
            # Apply dropout (training mode is False in all cases, so dropout is deterministic)
            # But we need to match PyTorch's behavior for correctness
            # In inference mode (training=False), dropout returns input directly
            if dropout_p > 0.0:
                # Random dropout mask
                random = tl.rand(tl.program_id(0) * 1000 + batch * 100 + seq, 0.0)
                mask = random > dropout_p
                logits = tl.where(mask, logits / (1.0 - dropout_p), 0.0)
            
            # Store to dropout output [B, S, D_out]
            dropout_offset = batch * S * D_out + seq * D_out + out_feature_pid
            tl.store(output_dropout_ptr + dropout_offset, logits)
            
            # Store to transpose output [B, D_out, S]
            transpose_offset = batch * D_out * S + out_feature_pid * S + seq
            tl.store(output_transpose_ptr + transpose_offset, logits)


@torch.fx.wrap
def fused_linear_dropout_transpose_kernel_wrapper(
    in_0, in_1, in_2, dropout_p
):
    """
    Wrapper for the fused kernel.
    in_0: bias tensor [D_out]
    in_1: weight tensor [D_out, D_in]
    in_2: input tensor [B, S, D_in]
    dropout_p: dropout probability (0.0 means no dropout)
    Returns: (dropout_output [B, S, D_out], transpose_output [B, D_out, S])
    """
    B, S, D_in = in_2.shape
    D_out = in_1.shape[0]
    
    # Create output tensors
    output_dropout = torch.empty(B, S, D_out, dtype=in_2.dtype, device=in_2.device)
    output_transpose = torch.empty(B, D_out, S, dtype=in_2.dtype, device=in_2.device)
    
    # Grid configuration
    BLOCK_SIZE_M = 64  # For output features
    BLOCK_SIZE_K = 32  # For reduction dimension
    
    grid = (B, D_out)  # (batch, output_features)
    
    # Call the kernel
    fused_linear_dropout_transpose_kernel[grid](
        in_2, in_1, in_0,
        output_dropout, output_transpose,
        B, S, D_in, D_out,
        dropout_p,
        BLOCK_SIZE_M,
        BLOCK_SIZE_K,
    )
    
    return output_dropout, output_transpose


def pattern(in_0, in_1, in_2):
    """
    Match the pattern: linear + dropout + transpose
    """
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.nn.functional.dropout(linear, 0.1, False, False)
    tmp_4 = tmp_3.transpose(1, 2)
    return tmp_3, tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, 0.1)


def replacement_func():
    return fused_linear_dropout_transpose_kernel_wrapper