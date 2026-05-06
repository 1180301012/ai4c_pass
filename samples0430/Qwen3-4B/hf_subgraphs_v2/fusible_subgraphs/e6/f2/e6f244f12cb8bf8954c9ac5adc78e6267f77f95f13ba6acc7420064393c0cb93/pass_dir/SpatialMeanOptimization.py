import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Matches the exact computation pattern: in_0 + in_1
    tmp = in_0 + in_1
    return tmp, tmp.mean((2, 3), keepdim=True)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

class SpatialMeanKernel:
    @triton.jit
    def mean_kernel(
        input_ptr,
        output_ptr,
        N: tl.int32,
        C: tl.int32,
        H: tl.int32,
        W: tl.int32,
        BLOCK_SIZE: tl.constexpr = 128
    ):
        pid = tl.program_id(0)
        start_h = pid * BLOCK_SIZE
        end_h = min(start_h + BLOCK_SIZE, H)
        
        total = 0.0
        count = 0
        
        for h in range(start_h, end_h):
            for w in range(W):
                idx = h * W + w
                val = tl.load(input_ptr + idx)
                total += val
                count += 1
        
        mean_val = total / count if count > 0 else 0.0
        tl.store(output_ptr, mean_val)

    @torch.fx.wrap
    def kernel_wrapper(x):
        N, C, H, W = x.shape
        output = torch.empty((N, C), x.dtype)
        
        # Launch kernel with proper grid
        SpatialMeanKernel.mean_kernel[torch.int32(N * C)][(N * C,)](
            x,
            output,
            N,
            C,
            H,
            W
        )
        
        return output

def replacement_func():
    return SpatialMeanKernel.kernel_wrapper