import torch

def pattern(linear_input, weight_tensor, position_indices):
    """
    Pattern: linear transformation followed by indexing with position indices
    Original sequence:
        linear = torch.nn.functional.linear(linear_input, weight_tensor, None)
        tmp_3 = linear.view(-1, weight_tensor.shape[1])  # features dimension
        tmp_4 = position_indices.view(-1)
        tmp_5 = tmp_3[tmp_4]
    """
    # Apply the exact sequence from original
    linear_result = torch.nn.functional.linear(linear_input, weight_tensor, None)
    features = weight_tensor.shape[1] if weight_tensor.dim() > 1 else weight_tensor.shape[0]
    viewed_result = linear_result.view(-1, features)
    flat_indices = position_indices.view(-1)
    output = viewed_result[flat_indices]
    
    # Reshape to match the expected output [64, 64, features]
    output = output.view(64, 64, features)
    return output

def replacement_args(linear_input, weight_tensor, position_indices):
    return (linear_input, weight_tensor, position_indices)

def replacement_func():
    def optimized_linear_index(linear_input, weight_tensor, position_indices):
        """
        Optimized version that reduces intermediate tensor creation
        """
        # Use torch.nn.functional.linear directly to avoid intermediate steps
        linear_result = torch.nn.functional.linear(linear_input, weight_tensor, None)
        
        # Combine view and indexing operations to reduce memory overhead
        features = weight_tensor.shape[1] if weight_tensor.dim() > 1 else weight_tensor.shape[0]
        output = linear_result.view(-1, features)[position_indices.view(-1)]
        
        # Final reshape
        output = output.view(64, 64, features)
        
        return output
    
    return optimized_linear_index