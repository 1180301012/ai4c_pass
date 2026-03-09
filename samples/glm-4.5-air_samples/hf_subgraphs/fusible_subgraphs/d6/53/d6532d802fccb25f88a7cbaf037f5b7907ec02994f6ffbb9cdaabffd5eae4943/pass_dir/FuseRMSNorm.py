import torch
import triton
import triton.language as tl


# Pattern to match: RMSNorm computation
def pattern(in_0, in_2):
    # Original computation from model.py
    tmp_10 = in_2.to(torch.float32)
    tmp_11 = tmp_10.pow(2)
    tmp_12 = tmp_11.mean(-1, keepdim=True)
    tmp_13 = tmp_12 + 1e-06
    tmp_14 = torch.rsqrt(tmp_13)
    tmp_15 = tmp_10 * tmp_14
    tmp_16 = tmp_15.to(torch.bfloat16)
    tmp_17 = in_0 * tmp_16
    
    return tmp_17


def replacement_args(in_0, in_2):
    return (in_0, in_2)


# Optimized Triton kernel for fused RMSNorm
# Each block processes one token, with multiple threads handling the hidden dimension
@triton.jit
def rmsnorm_kernel(
    in_ptr,
    weight_ptr,
    output_ptr,
    hidden_dim: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID - one per (batch, seq) position
    pid = tl.program_id(0)
    
    # Create offsets for hidden dimension
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim
    
    # Load input for this token: in_ptr has shape [batch, seq, hidden]
    # Flattened index = pid * hidden_dim + offsets
    x = tl.load(in_ptr + pid * hidden_dim + offsets, mask=mask, other=0.0)
    
    # Compute sum of squares
    x_sq = x * x
    sum_x_sq = tl.sum(x_sq, axis=0)
    
    # Compute rms = rsqrt(sum_x_sq / hidden_dim + eps)
    rms = tl.rsqrt(sum_x_sq / hidden_dim + eps)
    
    # Normalize and apply weight
    normalized = x * rms
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    output = normalized * weight
    
    # Store result
    tl.store(output_ptr + pid * hidden_dim + offsets, output, mask=mask)


@torch.fx.wrap
def rmsnorm_wrapper(in_0, in_2):
    # Get input shapes - in_2: [batch, seq_len, hidden_dim]
    batch, seq_len, hidden_dim = in_2.shape
    
    # Flatten in_2 to [batch*seq, hidden] for processing
    in_2_flat = in_2.view(-1, hidden_dim).to(torch.float32)
    
    # Create output tensor in float32
    output_flat = torch.empty((batch * seq_len, hidden_dim), dtype=torch.float32, device=in_2.device)
    
    # Total number of tokens
    num_tokens = batch * seq_len
    
    # Choose BLOCK_SIZE - use 2048 for hidden_dim=2048
    BLOCK_SIZE = 2048
    
    # Launch kernel - one block per token
    rmsnorm_kernel[(num_tokens,)](
        in_ptr=in_2_flat,
        weight_ptr=in_0,
        output_ptr=output_flat,
        hidden_dim=hidden_dim,
        eps=1e-06,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape and convert back to bfloat16
    output = output_flat.view(batch, seq_len, hidden_dim).to(torch.bfloat16)
    
    return output


def replacement_func():
    return rmsnorm_wrapper