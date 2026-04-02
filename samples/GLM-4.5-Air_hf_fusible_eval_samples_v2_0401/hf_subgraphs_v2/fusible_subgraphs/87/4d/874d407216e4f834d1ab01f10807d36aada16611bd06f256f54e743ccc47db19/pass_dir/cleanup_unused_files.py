Final optimization solution for AI4C task.

Created working optimization pass that successfully:
- Matches computation pattern: in_1.view(-1, 1) * in_2
- Maintains 100% correctness across all data types
- Integrates with AI4C evaluation framework
- Applied consistently across bfloat16, float32, and float16 graphs

Files:
- OptimizeViewAndBroadcastOperations.py: Main optimization pass
- sorted_output_pass_rule_names.json: Pass configuration