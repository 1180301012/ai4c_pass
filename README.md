# AI4C Pass

This repository stores AI4C-generated pass artifacts and evaluates pass quality against baselines.

## Usage

- `scripts/replay_trajectory_patch.py`
- `scripts/eval_ai4c_samples.py`


## 1) Rebuild Sample Bundles from Trajectory

Use this script to replay patch outputs and reconstruct sample directories.

```bash
python scripts/replay_trajectory_patch.py \
  --trajectory <your_trajectory_jsonl_file> \
  --sample-output-root samples/<model_name> \
  --workspace-root . \
  --all-records \
  --overwrite
```

## 2) Batch Evaluate AI4C Samples

Use this script to evaluate AI4C samples and generate one JSON array with per-sample metrics.

Script:

- `scripts/eval_ai4c_samples.py`

Per-sample output fields include:

- `is_pattern_match`
- `es_t_scores`
- `es_overall_score`
- `return_code`
- `error`

### Mode A: Evaluate from sample list

```bash
python -m scripts.eval_ai4c_samples \
  --sample-list sample_lists/hf_fusible_eval_samples.txt \
  --compiler inductor \
  --device cuda \
  --warmup 25 \
  --trials 100 \
  --output-root ./tmp/ai4c_eval_logs
```

### Mode B: Evaluate from dumped sample root

```bash
python scripts/eval_ai4c_samples.py \
  --samples-root samples/deepseek-v3.2 \
  --compiler pass_mgr \
  --device cuda \
  --warmup 25 \
  --trials 100 \
  --output-root /tmp/ai4c_eval_deepseek
```

### Output layout

If `--output-root /tmp/ai4c_eval_logs`:

- `/tmp/ai4c_eval_logs/logs/`: one log file per sample
- `/tmp/ai4c_eval_logs/pass_match/`: pass-match intermediate outputs
- `/tmp/ai4c_eval_logs/results.json`: final JSON array


## Key Defaults for `pass_mgr`

`eval_ai4c_samples.py` defaults:

- `--pass-input-dir-name const_pass_dir`
- `--pass-output-dir-name pass_dir`
- `--output-pass-pattern-limit 100`
- `--output-pass-replacement-func-limit 1`
