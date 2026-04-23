import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    tmp_3 = torch.ops.aten.embedding.default(in_2, in_1, -1, False, False)
    tmp_4 = torch.ops.aten._to_copy.default(in_0, dtype=torch.int64)
    return (tmp_3, tmp_4)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Optimized kernel: each program handles one index position and loads the entire embedding row
@triton.jit
def embedding_row_kernel(
    indices_ptr,
    weight_ptr,
    output_ptr,
    num_indices,
    embed_dim,
    stride_weight_0,
    stride_weight_1,
    stride_output_0,
    stride_output_2,
    BLOCK_E: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < num_indices:
        # Load index value from flat contiguous array
        idx_val = tl.load(indices_ptr + pid)
        
        # Load entire embedding row
        e_offsets = tl.arange(0, BLOCK_E)
        e_mask = e_offsets < embed_dim
        row_ptr = weight_ptr + idx_val.to(tl.int64) * stride_weight_0
        val = tl.load(row_ptr + e_offsets * stride_weight_1, mask=e_mask, other=0.0)
        
        # Store output row
        out_row_ptr = output_ptr + pid.to(tl.int64) * stride_output_0
        tl.store(out_row_ptr + e_offsets * stride_output_2, val, mask=e_mask)


@torch.fx.wrap
def fused_embedding_long(in_0, in_1, in_2):
    # Compute embedding using Triton kernel
    # in_0: attention_mask [batch, seq_len], int64 (for .long() cast)
    # in_1: indices [batch, seq_len], int64 (contiguous)
    # in_2: weight [vocab, embed_dim], bfloat16 (contiguous)
    
    embed_dim_val = in_2.shape[1]
    num_indices_val = in_1.numel()
    batch_size = in_1.shape[0]
    seq_len = in_1.shape[1]
    
    # Create output in the same dtype as weight
    output = torch.empty((batch_size, seq_len, embed_dim_val), dtype=in_2.dtype, device=in_2.device)

    BLOCK_E = triton.next_power_of_2(embed_dim_val)
    grid = (num_indices_val,)

    # Get strides
    sw0 = in_2.stride(0)
    sw1 = in_2.stride(1)
    so0 = output.stride(0)
    so2 = output.stride(2)

    embedding_row_kernel[grid](
        indices_ptr=in_1,
        weight_ptr=in_2,
        output_ptr=output,
        num_indices=num_indices_val,
        embed_dim=embed_dim_val,
        stride_weight_0=sw0,
        stride_weight_1=sw1,
        stride_output_0=so0,
        stride_output_2=so2,
        BLOCK_E=BLOCK_E,
    )

    # in_0.long() - just identity since in_0 is already int64
    tmp_4 = in_0.long()

    return (output, tmp_4)

def replacement_func():
    return fused_embedding_long