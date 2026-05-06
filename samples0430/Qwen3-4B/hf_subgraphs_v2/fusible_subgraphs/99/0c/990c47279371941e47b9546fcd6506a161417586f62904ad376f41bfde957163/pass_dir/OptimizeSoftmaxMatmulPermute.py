import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    scaled = 0.0625 * in_0
    softmaxed = torch.nn.functional.softmax(scaled, dim=-1)
    matmul_res = torch.matmul(softmaxed, in_1)
    permuted = matmul_res.permute(0, 2, 1)
    return (permuted,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

def optimize_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    n_batch: tl.int32,
    n_feat: tl.int32,
    n_heads: tl.int32,
    n_out: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the starting index for this block
    block_id = tl.program_id(0)
    start = block_id * BLOCK_SIZE
    # Calculate the number of elements in the block
    end = min((block_id + 1) * BLOCK_SIZE, n_batch * n_out)
    
    # Initialize output to zeros
    output = tl.zeros((n_out,), dtype=tl.float16)
    
    # Process each element in the block
    for i in range(start, end):
        # Load input feature vector (B, C, H)
        in_0 = tl.load(input_ptr + i)
        # Scale by 0.0625
        scaled = in_0 * 0.0625
        # Compute softmax along last dimension (H)
        # Implementation for softmax (simplified for this example)
        exp_vals = tl.exp(scaled)
        sums = tl.sum(exp_vals, axis=-1, keepdim=True)
        softmaxed = exp_vals / sums
        # Matrix multiply (B, H, C) * (B, C, D)
        matmul_res = tl.dot(softmaxed, weight_ptr)
        # Permute result (B, C, D) to (B, D, C)
        permuted = tl.permute(matmul_res, [0, 2, 1])
        # Store in output
        tl.store(output_ptr + i, permuted)

def kernel_wrapper(in_0, in_1):
    batch = in_0.shape[0]
    seq_len = in_0.shape[1]
    head_size = in_0.shape[2]
    out_size = in_1.shape[2]
    
    n_batch = batch
    n_feat = seq_len
    n_heads = head_size
    n_out = out_size
    
    output = torch.empty((batch, out_size, head_size),
                         dtype=in_0.dtype,
                         device=in_0.device)
    
    grid = (batch, )
    optimize_kernel[grid](
        input_ptr=in_0,
        weight_ptr=in_1,
        output_ptr=output,
        n_batch=batch,
        n_feat=seq_len,
        n_heads=head_size,
        n_out=out_size,
        BLOCK_SIZE=1024,
    )
    
    return (output,)

def replacement_func():
    return kernel_wrapper