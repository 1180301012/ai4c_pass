import json

# Create the pass configuration
config = ["FuseMatmulScale", "OptimizeTranspose2D"]

# Write to JSON file
with open("sorted_output_pass_rule_names.json", "w") as f:
    json.dump(config, f, indent=2)

print("Configuration file created successfully!")