## Scripts Usage

* 运行 Baseline
```bash
# Torch Eager
python -m graph_net_bench.torch.test_compiler \
  --model-path "graphs" \
  --compiler "nope" \
  --device "cuda" \
  --warmup 25 \
  --trials 100 \
  --model-path-prefix "." \
  --allow-list "sample_lists/hf_fusible_eval_samples.txt"

# Inductor Default
python -m graph_net_bench.torch.test_compiler \
  --model-path "eval_dataset" \
  --compiler "inductor" \
  --device "cuda" \
  --warmup 25 \
  --trials 100 \
  --allow-list "sample_lists/hf_fusible_eval_samples.txt" \

# Inductor Maxtune
python -m graph_net_bench.torch.test_compiler \
  --model-path "eval_dataset" \
  --compiler "inductor" \
  --device "cuda" \
  --warmup 25 \
  --trials 100 \
  --allow-list "sample_lists/hf_fusible_eval_samples.txt" \
  --config $(base64 -w 0 <<EOF
{
    "inductor_mode": "max-autotune",
}
EOF
)

```


* 从轨迹中 dump 下 pass_dir 以及完整的测试套件
```bash
python ./scripts/replay_trajectory_patch.py \
  --trajectory <your_trajectory_jsonl_file> \
  --sample-output-root ./samples/<model_name> \
  --workspace-root ./ai4c \
  --all-records \
  --overwrite
```
