import torch
import torch.fx as fx

# This script is for debugging purposes and won't be used as an optimization pass
# The file is needed to analyze the computational graph structure

def analyze_graph():
    # Load one of the models to understand the graph structure
    model_path = "./graphs/hf_subgraphs_v2/fusible_subgraphs/bfloat16/6/samples/mmpose/ipr_resnet_50/_decomposed/ipr_resnet_50_start184_end193_5/model.py"
    
    # Create the model
    import sys
    sys.path.append(model_path.rsplit('/', 1)[0])
    from model import GraphModule
    
    model = GraphModule()
    model.eval()
    
    # Create dummy inputs matching the weight metadata
    in_0 = torch.randn(1, 1, 1, 64, dtype=torch.bfloat16, device='cuda')
    in_1 = torch.randn(1, 1, 64, 1, dtype=torch.bfloat16, device='cuda')  
    in_2 = torch.randn(128, 17, 4096, dtype=torch.bfloat16, device='cuda')
    
    # Get the computation graph
    graph_model = fx.symbolic_trace(model)
    
    print("Graph structure:")
    print(graph_model.graph)
    print("\nNodes:")
    for node in graph_model.graph.nodes:
        print(f"  {node}")
    
    try:
        # Run forward pass to see outputs
        with torch.no_grad():
            outputs = model(in_0, in_1, in_2)
            print(f"\nOutput shapes: {[out.shape for out in outputs]}")
    except Exception as e:
        print(f"Error running model: {e}")

if __name__ == "__main__":
    analyze_graph()