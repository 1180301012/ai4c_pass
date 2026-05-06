import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_1 = in_0.unsqueeze(1)
    tmp_2 = tmp_1.transpose(2, 3)
    return (tmp_2,)

def replacement_args(in_0):
    return (in_0,)

def kernel_wrapper(in_0):
    batch_size = in_0.shape[0]
    sequence_size = in_0.shape[1]
    feature_size = in_0.shape[2]
    
    out = torch.empty((batch_size, 1, feature_size, sequence_size), 
                      dtype=in_0.dtype, 
                      device=in_0.device)
    
    @triton.jit
    def optimized_kernel(in_ptr, out_ptr, n_batch, n_sequence, n_feature, BLOCK_SIZE):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < (n_sequence * n_feature)
        
        for i in offsets:
            if not mask[i]:
                continue
            input_idx = block_start + i
            s = input_idx // n_feature
            f = input_idx % n_feature
            output_idx = f * n_sequence + s
            tl.store(out_ptr + output_idx, tl.load(in_ptr + input_idx), mask=mask[i])
    
    optimized_kernel[(1,)](        in_ptr=in_0,
        out_ptr=out,
        n_batch=batch_size,
        n_sequence=sequence_size,
        n_feature=feature_size,
        BLOCK_SIZE=1024,
    )
    
    return out
def replacement_func():
    return kernel_wrapper