import torch
import triton
import triton.language as tl

# Pattern matching function - must exactly match the graph operations
# Note: We match the specific embedding operations and addition chain
@torch.fx.wrap
def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13):
    tmp_15 = in_2[:, :256]
    tmp_16 = torch.nn.functional.embedding(in_0, in_9, 0, None, 2.0, False, False)
    tmp_17 = torch.nn.functional.embedding(tmp_15, in_6, None, None, 2.0, False, False)
    tmp_18 = in_13[:, :, 0]
    tmp_19 = torch.nn.functional.embedding(tmp_18, in_10, None, None, 2.0, False, False)
    tmp_20 = in_13[:, :, 1]
    tmp_21 = torch.nn.functional.embedding(tmp_20, in_11, None, None, 2.0, False, False)
    tmp_22 = in_13[:, :, 2]
    tmp_23 = torch.nn.functional.embedding(tmp_22, in_10, None, None, 2.0, False, False)
    tmp_24 = in_13[:, :, 3]
    tmp_25 = torch.nn.functional.embedding(tmp_24, in_11, None, None, 2.0, False, False)
    tmp_26 = in_13[:, :, 3]
    tmp_27 = in_13[:, :, 1]
    tmp_28 = tmp_26 - tmp_27
    tmp_29 = torch.nn.functional.embedding(tmp_28, in_5, None, None, 2.0, False, False)
    tmp_30 = in_13[:, :, 2]
    tmp_31 = in_13[:, :, 0]
    tmp_32 = tmp_30 - tmp_31
    tmp_33 = torch.nn.functional.embedding(tmp_32, in_8, None, None, 2.0, False, False)
    tmp_34 = torch.nn.functional.embedding(in_1, in_7, None, None, 2.0, False, False)

    tmp_35 = tmp_16 + tmp_17
    tmp_36 = tmp_35 + tmp_19
    tmp_37 = tmp_36 + tmp_21
    tmp_38 = tmp_37 + tmp_23
    tmp_39 = tmp_38 + tmp_25
    tmp_40 = tmp_39 + tmp_29
    tmp_41 = tmp_40 + tmp_33
    tmp_42 = tmp_41 + tmp_34

    return tmp_42

# Argument extraction for the kernel
def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13)

# Optimized Triton kernel - fuses all embeddings and additions
@triton.jit
def embed_fusion_kernel(
    in_0_ptr,  # input ids [batch, seq_len]
    in_1_ptr,  # token type ids [batch, seq_len]
    in_2_ptr,  # position ids [1, 512]
    in_13_ptr, # bbox [batch, seq_len, 4]
    weights_0_ptr,  # word embeddings [vocab_size, 768]
    weights_1_ptr,  # position embeddings [512, 768]
    weights_2_ptr,  # x_position embeddings [1024, 768]
    weights_3_ptr,  # y_position embeddings [1024, 768]
    weights_4_ptr,  # h_position embeddings [1024, 768]
    weights_5_ptr,  # w_position embeddings [1024, 768]
    weights_6_ptr,  # token type embeddings [2, 768]
    out_ptr,  # output [batch, seq_len, 768]
    batch_size: tl.int32,
    seq_len: tl.int32,
    embed_dim: tl.int32,
    num_embeddings_0: tl.int32,
    num_embeddings_1: tl.int32,
    num_embeddings_2: tl.int32,
    num_embeddings_3: tl.int32,
    num_embeddings_4: tl.int32,
    num_embeddings_5: tl.int32,
    num_embeddings_6: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    # Determine program ID for sequence dimension
    batch = tl.program_id(0)
    seq_offset = tl.program_id(1) * BLOCK_SIZE
    offsets = seq_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < seq_len

    # Process each position in the sequence
    for i in range(BLOCK_SIZE):
        if offsets[i] >= seq_len:
            continue
        
        # Get all the indices for this position
        idx0 = tl.load(in_0_ptr + batch * seq_len + offsets[i])
        # Position IDs - use first 256 from in_2 (shape [1, 512])
        idx1 = tl.load(in_2_ptr + offsets[i])
        idx2 = tl.load(in_13_ptr + batch * seq_len * 4 + offsets[i] * 4)
        idx3 = tl.load(in_13_ptr + batch * seq_len * 4 + offsets[i] * 4 + 1)
        idx4 = tl.load(in_13_ptr + batch * seq_len * 4 + offsets[i] * 4 + 2)
        idx5 = tl.load(in_13_ptr + batch * seq_len * 4 + offsets[i] * 4 + 3)
        # For the position differences
        idx6 = idx5 - idx3  # in_13[:,:,3] - in_13[:,:,1]
        idx7 = idx4 - idx2  # in_13[:,:,2] - in_13[:,:,0]
        idx8 = tl.load(in_1_ptr + batch * seq_len + offsets[i])

        # Load all embedding vectors
        emb0 = tl.load(weights_0_ptr + idx0 * embed_dim + tl.arange(0, embed_dim), mask=mask, other=0.0)
        emb1 = tl.load(weights_1_ptr + idx1 * embed_dim + tl.arange(0, embed_dim), mask=mask, other=0.0)
        emb2 = tl.load(weights_2_ptr + idx2 * embed_dim + tl.arange(0, embed_dim), mask=mask, other=0.0)
        emb3 = tl.load(weights_3_ptr + idx3 * embed_dim + tl.arange(0, embed_dim), mask=mask, other=0.0)
        emb4 = tl.load(weights_2_ptr + idx4 * embed_dim + tl.arange(0, embed_dim), mask=mask, other=0.0)  # Reusing x_position for idx4
        emb5 = tl.load(weights_3_ptr + idx5 * embed_dim + tl.arange(0, embed_dim), mask=mask, other=0.0)  # Reusing y_position for idx5
        emb6 = tl.load(weights_4_ptr + idx6 * embed_dim + tl.arange(0, embed_dim), mask=mask, other=0.0)
        emb7 = tl.load(weights_5_ptr + idx7 * embed_dim + tl.arange(0, embed_dim), mask=mask, other=0.0)
        emb8 = tl.load(weights_6_ptr + idx8 * embed_dim + tl.arange(0, embed_dim), mask=mask, other=0.0)

        # Sum all embeddings
        result = emb0 + emb1 + emb2 + emb3 + emb4 + emb5 + emb6 + emb7 + emb8

        # Store the result
        tl.store(out_ptr + batch * seq_len * embed_dim + offsets[i] * embed_dim + tl.arange(0, embed_dim), 
                 result, mask=mask)

# Wrapper to set up the Triton kernel execution
@torch.fx.wrap
def embed_fusion(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13):
    batch_size, seq_len = in_0.shape
    embed_dim = 768

    # Get all the shapes for the weight matrices
    num_embeddings_0 = in_9.shape[0]  # word embeddings
    num_embeddings_1 = in_6.shape[0]  # position embeddings
    num_embeddings_2 = in_10.shape[0]  # x_position
    num_embeddings_3 = in_11.shape[0]  # y_position
    num_embeddings_4 = in_5.shape[0]  # h_position
    num_embeddings_5 = in_8.shape[0]  # w_position
    num_embeddings_6 = in_7.shape[0]  # token type

    # Create output tensor
    out = torch.empty((batch_size, seq_len, embed_dim), dtype=in_0.dtype, device=in_0.device)

    # Set block size (adjust based on performance needs)
    BLOCK_SIZE = 128
    num_programs = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    embed_fusion_kernel[(batch_size, num_programs), BLOCK_SIZE](
        in_0_ptr=in_0, 
        in_1_ptr=in_1, 
        in_2_ptr=in_2, 
        in_13_ptr=in_13,
        weights_0_ptr=in_9, 
        weights_1_ptr=in_6, 
        weights_2_ptr=in_10, 
        weights_3_ptr=in_11, 
        weights_4_ptr=in_5, 
        weights_5_ptr=in_8, 
        weights_6_ptr=in_7,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        embed_dim=embed_dim,
        num_embeddings_0=num_embeddings_0,
        num_embeddings_1=num_embeddings_1,
        num_embeddings_2=num_embeddings_2,
        num_embeddings_3=num_embeddings_3,
        num_embeddings_4=num_embeddings_4,
        num_embeddings_5=num_embeddings_5,
        num_embeddings_6=num_embeddings_6,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

# Replacement function - returns the kernel wrapper
def replacement_func():
    return embed_fusion