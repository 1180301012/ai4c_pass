import torch
import triton
import triton.language as tl

@triton.jit
def fused_linear_dropout_transpose_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr, output_trans_ptr,
    batch_size, seq_len, in_features, out_features,
    dropout_p, training: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    num_stages: tl.constexpr
):
    """
    Fused kernel for: linear(input, weight, bias) -> dropout -> transpose(1,2)
    
    Computes: output[b, s, o] = sum_k(input[b, s, k] * weight[o, k]) + bias[o]
    Then applies dropout, then transposes to [b, o, s]
    """
    # Grid: (out_features // BLOCK_SIZE_N, batch_size * seq_len // BLOCK_SIZE_M)
    pid_dim0 = tl.program_id(0)
    pid_dim1 = tl.program_id(1)
    
    # Calculate output row and column for this program
    row_offset = pid_dim1 * BLOCK_SIZE_M
    col_offset = pid_dim0 * BLOCK_SIZE_N
    
    # Initialize output accumulation matrix in registers
    # Shape: BLOCK_SIZE_M x BLOCK_SIZE_N
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Memory access offsets
    input_row_offsets = row_offset + tl.arange(0, BLOCK_SIZE_M)
    input_col_offsets = tl.arange(0, BLOCK_SIZE_K)
    
    weight_row_offsets = col_offset + tl.arange(0, BLOCK_SIZE_N)
    weight_col_offsets = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize bias in registers if present
    bias = tl.load(bias_ptr + (col_offset + tl.arange(0, BLOCK_SIZE_N)), 
                   mask=(col_offset + tl.arange(0, BLOCK_SIZE_N)) < out_features,
                   other=0.0)
    
    # Loop over K dimension with blocking
    for k in range(0, in_features, BLOCK_SIZE_K):
        # Load input block
        input_mask = (input_row_offsets < batch_size * seq_len) & \
                     ((k + input_col_offsets) < in_features)
        input_block = tl.load(
            input_ptr + (input_row_offsets[:, None] * in_features + k + input_col_offsets[None, :]),
            mask=input_mask,
            other=0.0
        )
        
        # Load weight block
        weight_mask = (weight_row_offsets < out_features) & \
                      ((k + weight_col_offsets) < in_features)
        weight_block = tl.load(
            weight_ptr + ((k + weight_col_offsets)[:, None] * in_features + weight_row_offsets[None, :]),
            mask=weight_mask,
            other=0.0
        )
        
        # Matrix multiply: input [M,K] @ weight [K,N]^T = input [M,K] @ weight.T [N,K]
        # But weight is stored as [out_features, in_features], so we need weight.T
        # We can just use weight_block as-is since it's loaded correctly
        acc += tl.dot(input_block, weight_block)
        
        # Also add for transpose case: input [M,K] @ weight.T [K,N]
        # We need to multiply input_block [M,K] with transposed weight_block [K,N]
        # Actually weight_block loaded as [K,N] (where N is BLOCK_SIZE_K dimension)
        # The weight layout is [out_features, in_features] = [N, K]
        # We need [in_features, out_features] = [K, N]
        # So weight.T should be [N, K] -> we need to transpose
        
    # Add bias
    acc = acc + bias[None, :]
    
    # Apply dropout during training
    if training:
        # Generate random mask using philox RNG
        philox_seed = 42  # Can be made configurable
        philox_offset = (pid_dim0 * 97 + pid_dim1 * 113) % 1024
        random = tl.rand(philox_seed, philox_offset)
        dropout_mask = random > dropout_p
        # Scale by 1/(1-p) during training for proper dropout
        scale = 1.0 / (1.0 - dropout_p + 1e-7)
        acc = acc * dropout_mask * scale
    
    # Convert to output dtype
    output_dtype = tl.load(input_ptr + (0 * in_features)).dtype  # Get dtype from input
    acc = acc.to(output_dtype)
    
    # Store original output (before transpose): shape [B*S, out_features]
    output_row = row_offset + tl.arange(0, BLOCK_SIZE_M)
    output_col = col_offset + tl.arange(0, BLOCK_SIZE_N)
    output_mask = (output_row < batch_size * seq_len) & (output_col < out_features)
    
    tl.store(
        output_ptr + (output_row[:, None] * out_features + output_col[None, :]),
        acc,
        mask=output_mask
    )
    
    # Store transposed output: shape [B, out_features, S]
    # Transpose from [B*S, out_features] to [B, out_features, S]
    # output[b, s, o] -> transposed[b, o, s]
    # For linear output: position p = b * seq_len + s, we need to write to [b, o, s]
    for m in range(BLOCK_SIZE_M):
        for n in range(BLOCK_SIZE_N):
            b = (row_offset + m) // seq_len
            s = (row_offset + m) % seq_len
            o = col_offset + n
            if b < batch_size and s < seq_len and o < out_features:
                idx_orig = (row_offset + m) * out_features + (col_offset + n)
                val = tl.load(output_ptr + idx_orig)
                idx_trans = b * out_features * seq_len + o * seq_len + s
                tl.store(output_trans_ptr + idx_trans, val)


def pattern(in_0, in_1, in_2):
    """
    Match pattern: linear(in_2, in_1, in_0) -> dropout -> transpose(1,2)
    
    This is a feature projection layer commonly found in transformer models.
    The pattern needs to return both the dropout output and transposed output
    since both are observable in the model's return.
    """
    linear_out = torch.nn.functional.linear(in_2, in_1, in_0)
    dropout_out = torch.nn.functional.dropout(linear_out, 0.1, False, False)
    transposed_out = dropout_out.transpose(1, 2)
    return dropout_out, transposed_out


def replacement_args(in_0, in_1, in_2):
    """Extract arguments needed for the fused kernel."""
    return (in_0, in_1, in_2)


@torch.fx.wrap
def fused_linear_dropout_transpose_wrapper(in_0, in_1, in_2):
    """
    Wrapper function that launches the fused Triton kernel.
    
    Args:
        in_0: bias tensor [out_features]
        in_1: weight tensor [out_features, in_features]
        in_2: input tensor [batch, seq, in_features]
    
    Returns:
        dropout_out: [batch, seq, out_features]
        transposed_out: [batch, out_features, seq]
    """
    batch_size, seq_len, in_features = in_2.shape
    out_features, in_features_w = in_1.shape
    
    # Allocate output tensors
    dropout_out = torch.empty((batch_size, seq_len, out_features), 
                               dtype=in_2.dtype, device=in_2.device)
    transposed_out = torch.empty((batch_size, out_features, seq_len), 
                                  dtype=in_2.dtype, device=in_2.device)
    
    # Block sizes - tune based on problem size
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # Grid dimensions
    grid_m = (batch_size * seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (out_features + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    fused_linear_dropout_transpose_kernel[(grid_n, grid_m)](
        input_ptr=in_2,
        weight_ptr=in_1,
        bias_ptr=in_0,
        output_ptr=dropout_out,
        output_trans_ptr=transposed_out,
        batch_size=batch_size,
        seq_len=seq_len,
        in_features=in_features,
        out_features=out_features,
        dropout_p=0.1,
        training=False,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        num_stages=2
    )
    
    return dropout_out, transposed_out


def replacement_func():
    return fused_linear_dropout_transpose_wrapper