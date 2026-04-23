import torch
import triton
import triton.language as tl

# Pattern matching function (matches the specific 32x32 meshgrid pattern)
def pattern(in_0, in_1):
    tmp_0 = torch.cat([in_1, in_0])
    tmp_1 = torch.arange(32)
    tmp_2 = torch.arange(32)
    meshgrid = torch.functional.meshgrid(tmp_1, tmp_2, indexing = 'ij')
    tmp_4 = meshgrid[0]
    tmp_5 = meshgrid[1]
    tmp_6 = torch.stack((tmp_4, tmp_5))
    tmp_7 = torch.flatten(tmp_6, 1)
    tmp_8 = tmp_7[(slice(None, None, None), slice(None, None, None), None)]
    tmp_9 = tmp_7[(slice(None, None, None), None, slice(None, None, None))]
    tmp_10 = tmp_8 - tmp_9
    tmp_11 = tmp_10.permute(1, 2, 0)
    tmp_12 = tmp_11.contiguous()
    tmp_13 = tmp_12[(slice(None, None, None), slice(None, None, None), 0)]
    tmp_13 += 31
    tmp_14 = tmp_13
    tmp_12[(slice(None, None, None), slice(None, None, None), 0)] = tmp_14
    tmp_16 = tmp_12[(slice(None, None, None), slice(None, None, None), 1)]
    tmp_16 += 31
    tmp_17 = tmp_16
    tmp_12[(slice(None, None, None), slice(None, None, None), 1)] = tmp_17
    tmp_19 = tmp_12[(slice(None, None, None), slice(None, None, None), 0)]
    tmp_19 *= 63
    tmp_20 = tmp_19
    tmp_12[(slice(None, None, None), slice(None, None, None), 0)] = tmp_20
    tmp_22 = torch.zeros(size = (1025, 1025), dtype = torch.int64)
    tmp_23 = tmp_12.sum(-1)
    tmp_22[(slice(1, None, None), slice(1, None, None))] = tmp_23
    tmp_22[(0, slice(0, None, None))] = 3969
    tmp_22[(slice(0, None, None), 0)] = 3970
    tmp_22[(0, 0)] = 3971
    tmp_28 = tmp_22.view(-1)
    return (tmp_0, tmp_28)

# Extract necessary parameters for the kernel

def replacement_args(in_0, in_1):
    # Parameters extracted from the pattern:
    # N = 32, offset = 31, scale = 63, matrix_size = 1025
    return (32, 31, 63, 1025)

# Triton kernel for efficient coordinate matrix calculation
@triton.jit
def coordinate_matrix_kernel(out_ptr, N: tl.constexpr, offset: tl.constexpr, scale: tl.constexpr, matrix_size: tl.constexpr):
    i = tl.program_id(0)
    j = tl.program_id(1)
    
    if i == 0 and j == 0:
        value = 3971
    elif i == 0:
        value = 3969
    elif j == 0:
        value = 3970
    else:
        i_idx = i - 1
        j_idx = j - 1
        x = (i_idx + offset) * scale
        y = j_idx + offset
        value = x + y

    tl.store(out_ptr + i * matrix_size + j, value)

# Kernel wrapper
@torch.fx.wrap
def coordinate_matrix_wrapper(N, offset, scale, matrix_size):
    out = torch.empty((matrix_size, matrix_size), dtype=torch.int64, device=torch.device('cuda:0'))
    grid = (matrix_size, matrix_size)
    coordinate_matrix_kernel[grid](out, N, offset, scale, matrix_size)
    return out

# Return the kernel wrapper as the replacement function

def replacement_func():
    return coordinate_matrix_wrapper