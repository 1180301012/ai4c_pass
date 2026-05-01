import torch
import triton
import triton.language as tl

def pattern(a, b, c, d, e, f, g, h, i, j, k, l, m, n):
    tmp_16 = torch.nn.functional.embedding(a, j, 0, None, 2.0, False, False)
    tmp_17 = torch.nn.functional.embedding(c[:, :256], g, None, None, 2.0, False, False)
    tmp_19 = torch.nn.functional.embedding(n[:, :, 0], k, None, None, 2.0, False, False)
    tmp_21 = torch.nn.functional.embedding(n[:, :, 1], l, None, None, 2.0, False, False)
    tmp_23 = torch.nn.functional.embedding(n[:, :, 2], k, None, None, 2.0, False, False)
    tmp_25 = torch.nn.functional.embedding(n[:, :, 3], l, None, None, 2.0, False, False)
    tmp_29 = torch.nn.functional.embedding(n[:, :, 3] - n[:, :, 1], f, None, None, 2.0, False, False)
    tmp_33 = torch.nn.functional.embedding(n[:, :, 2] - n[:, :, 0], i, None, None, 2.0, False, False)
    tmp_34 = torch.nn.functional.embedding(b, h, None, None, 2.0, False, False)
    
    tmp_35 = tmp_16 + tmp_17
    tmp_36 = tmp_35 + tmp_19
    tmp_37 = tmp_36 + tmp_21
    tmp_38 = tmp_37 + tmp_23
    tmp_39 = tmp_38 + tmp_25
    tmp_40 = tmp_39 + tmp_29
    tmp_41 = tmp_40 + tmp_33
    tmp_42 = tmp_41 + tmp_34
    
    return tmp_42

def replacement_args(a, b, c, d, e, f, g, h, i, j, k, l, m, n):
    index1 = a
    index2 = c[:, :256]
    index3 = n[:, :, 0]
    index4 = n[:, :, 1]
    index5 = n[:, :, 2]
    index6 = n[:, :, 3]
    index7 = n[:, :, 3] - n[:, :, 1]
    index8 = n[:, :, 2] - n[:, :, 0]
    index9 = b
    
    weight1 = j
    weight2 = g
    weight3 = k
    weight4 = l
    weight5 = k
    weight6 = l
    weight7 = f
    weight8 = i
    weight9 = h
    
    return (index1, index2, index3, index4, index5, index6, index7, index8, index9, weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, weight9)

@triton.jit
def embed_and_accumulate_kernel(
    index1, index2, index3, index4, index5, index6, index7, index8, index9,
    weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, weight9,
    output,
    batch_size, seq_len, feature_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Thread index
    pid = tl.program_id(0)
    seq_id = pid % seq_len
    batch_id = pid // seq_len
    
    # Skip out of bounds
    if batch_id >= batch_size or seq_id >= seq_len:
        return
    
    # Load index values
    idx1 = tl.load(index1 + batch_id * seq_len + seq_id)
    idx2 = tl.load(index2 + batch_id * seq_len + seq_id)
    idx3 = tl.load(index3 + batch_id * seq_len + seq_id)
    idx4 = tl.load(index4 + batch_id * seq_len + seq_id)
    idx5 = tl.load(index5 + batch_id * seq_len + seq_id)
    idx6 = tl.load(index6 + batch_id * seq_len + seq_id)
    idx7 = tl.load(index7 + batch_id * seq_len + seq_id)
    idx8 = tl.load(index8 + batch_id * seq_len + seq_id)
    idx9 = tl.load(index9 + batch_id * seq_len + seq_id)
    
    # Load weight vectors
    w1 = tl.load(weight1 + idx1 * feature_dim + tl.arange(0, feature_dim))
    w2 = tl.load(weight2 + idx2 * feature_dim + tl.arange(0, feature_dim))
    w3 = tl.load(weight3 + idx3 * feature_dim + tl.arange(0, feature_dim))
    w4 = tl.load(weight4 + idx4 * feature_dim + tl.arange(0, feature_dim))
    w5 = tl.load(weight5 + idx5 * feature_dim + tl.arange(0, feature_dim))
    w6 = tl.load(weight6 + idx6 * feature_dim + tl.arange(0, feature_dim))
    w7 = tl.load(weight7 + idx7 * feature_dim + tl.arange(0, feature_dim))
    w8 = tl.load(weight8 + idx8 * feature_dim + tl.arange(0, feature_dim))
    w9 = tl.load(weight9 + idx9 * feature_dim + tl.arange(0, feature_dim))
    
    # Accumulate embeddings
    result = w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8 + w9
    
    # Store result
    tl.store(output + (batch_id * seq_len + seq_id) * feature_dim + tl.arange(0, feature_dim), result)

@torch.fx.wrap
def fused_embedding_and_accumulate(*args):
    (index1, index2, index3, index4, index5, index6, index7, index8, index9, 
     weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, weight9) = args
    batch_size, seq_len = index1.shape
    feature_dim = weight1.shape[1]
    
    BLOCK_SIZE = 128
    grid = (batch_size * seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty((batch_size, seq_len, feature_dim), dtype=weight1.dtype)
    
    embed_and_accumulate_kernel[grid](
        index1, index2, index3, index4, index5, index6, index7, index8, index9,
        weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, weight9,
        output,
        batch_size, seq_len, feature_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_embedding_and_accumulate