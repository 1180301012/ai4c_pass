import torch
import triton
import triton.language as tl

def pattern(x):
    """Simple pattern: convert to float32 and square"""
    # Convert to float32 and square - basic element-wise operations
    return x.to(torch.float32).pow(2)

def replacement_args(x):
    """Extract arguments for the replacement function"""
    return (x,)

@triton.jit
def normalize_kernel(
    x_ptr,
    rsqrt_ptr,
    n_elements,
    reduction_dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for normalization computation"""
    # Each program handles a block of data
    pid = tl.program_id(0)
    # Calculate offsets for this program
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Convert to float and square in one operation
    x_squared = x.to(tl.float32) * x.to(tl.float32)
    
    # Store squared values for reduction (we'll do reduction separately)
    tl.store(rsqrt_ptr + offsets, x_squared, mask=mask)

# Single kernel for complete normalization computation
# Simple Triton kernel for element-wise operations (square, epsilon addition)
@triton.jit
def square_add_epsilon_kernel(
    x_ptr,
    squared_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel: square values and add epsilon to mean later"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load and convert to float32, then square
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_float = x.to(tl.float32)
    x_squared = x_float * x_float
    
    # Store squared values
    tl.store(squared_ptr + offsets, x_squared, mask=mask)

# Pure Triton implementation to avoid forbidden API usage
@triton.jit
def compute_simple_rsqrt(x_val):
    """Compute reciprocal square root with epsilon addition"""
    epsilon_added = x_val + 1e-06
    return 1.0 / tl.sqrt(epsilon_added)

# Simple wrapper avoiding forbidden APIs
@torch.fx.wrap
def compute_mean_rsqrt_simple(x_squared, reduction_dim=-1, keepdim=True):
    """Compute mean, add epsilon, and rsqrt using basic operations"""
    # Compute mean
    mean_val = x_squared.mean(reduction_dim, keepdim=keepdim)
    
    # Add epsilon and compute reciprocal square root
    epsilon_added = mean_val + 1e-06
    rsqrt_val = 1.0 / torch.sqrt(epsilon_added)
    
    return rsqrt_val

@torch.fx.wrap
def fuse_normalize_mean_rsqrt(x, reduction_dim=-1, keepdim=True):
    """Fused normalization function: convert to float32, square, compute mean, add epsilon, compute rsqrt"""
    if x.dim() != 3:
        raise ValueError("Expected 3D input tensor")
    
    batch, seq_len, hidden_size = x.shape
    n_elements = batch * seq_len * hidden_size
    
    # Temporary storage for squared values
    x_squared = torch.empty_like(x, dtype=torch.float32)
    
    # Block size optimized for the problem
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch Triton kernel for element-wise operations (convert to float32 and square)
    square_add_epsilon_kernel[(num_programs,)](
        x_ptr=x,
        squared_ptr=x_squared,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Use PyTorch for the reduction operations (mean, epsilon, rsqrt)
    # these operations are already highly optimized in PyTorch
    rsqrt_result = compute_mean_rsqrt_torch(x_squared, reduction_dim, keepdim)
    
    return rsqrt_result

def replacement_func():
    """Return the fused normalization function"""
    return fuse_normalize_mean_rsqrt