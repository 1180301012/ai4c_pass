import torch
import triton
import triton.language as tl

@triton.jit
def fused_softmax_kernel(
    in_1_ptr,
    in_2_ptr,
    in_3_ptr,
    out_ptr,
    BLOCK_SIZE_K: tl.constexpr,
    N_L: tl.constexpr,
):
    j = tl.program_id(0)
    k = tl.thread_id(0)
    
    # Compute sum over l for (j,k)
    sum_val = 0
    for l in range(N_L):
        in1 = tl.load(in_1_ptr + (j * BLOCK_SIZE_K * N_L + k * N_L + l))
        in2 = tl.load(in_2_ptr + (k * N_L + l))
        diff = in1 - in2
        sum_val += diff * diff
    
    in3 = tl.load(in_3_ptr + k)
    sum_val *= in3
    
    # Store to shared memory for reduction
    shmem = tl.shared.zeros((BLOCK_SIZE_K,), tl.float16)
    shmem[k] = sum_val
    tl.sync()
    
    # Reduce to find max
    max_val = -float('inf')
    for i in range(BLOCK_SIZE_K):
        max_val = tl.maximum(max_val, shmem[i])
    
    # Compute sum of exp
    sum_exp = 0.0
    for i in range(BLOCK_SIZE_K):
        exp_val = tl.exp(shmem[i] - max_val)
        sum_exp += exp_val
    
    # Compute softmax and store
    exp_val = tl.exp(sum_val - max_val)
    output_val = exp_val / sum_exp
    tl.store(out_ptr + (j * BLOCK_SIZE_K + k), output_val)

@torch.fx.wrap
def fused_softmax_wrapper(in_1, in_2, in_3):
    batch_size = in_1.shape[0]
    n_j = in_1.shape[1]
    n_k = in_1.shape[2]
    n_l = in_1.shape[3]
    
    assert batch_size == 1
    assert in_2.shape == (1, 1, n_k, n_l)
    assert in_3.shape == (1, 1, n_k)
    
    x1_flat = in_1.view(-1)
    x2_flat = in_2.view(-1)
    x3_flat = in_3.view(-1)
    
    out_shape = (1, n_j, n_k)
    out_flat = torch.empty(out_shape, dtype=in_1.dtype)
    
    num_blocks = n_j
    BLOCK_SIZE_K = n_k
    N_L = n_l
    
    fused_softmax_kernel[(num_blocks,)](
        x1_flat,
        x2_flat,
        x3_flat,
        out_flat,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        N_L=N_L
    )
    
    return out_flat

def pattern(in_1, in_2, in_3):
    tmp_1 = in_1 - in_2
    tmp_2 = tmp_1.pow(2)
    tmp_3 = tmp_2.sum(dim = 3)
    tmp_4 = in_3 * tmp_3
    tmp_5 = torch.nn.functional.softmax(tmp_4, dim = 2)
    return tmp_5

def replacement_args(in_1, in_2, in_3):
    return (in_1, in_2, in_3)

def replacement_func():
    return fused_softmax_wrapper