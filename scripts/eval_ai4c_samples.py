#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph_net_bench import analysis_util
from graph_net_bench.aggregate_es_scores import get_weights
from graph_net_bench.positive_tolerance_interpretation_manager import (
    get_positive_tolerance_interpretation,
    get_supported_positive_tolerance_interpretation_types,
)


def resolve_path(path_text: str, workspace_root: Path) -> Path:
    p = Path(path_text)
    if p.is_absolute():
        return p
    return (workspace_root / p).resolve()


def encode_config(config: dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=True)
    return base64.b64encode(payload.encode("utf-8")).decode("utf-8")


def _parse_json_with_optional_trailing_commas(text: str) -> dict[str, Any]:
    normalized = text.strip()
    # Accept relaxed JSON snippets with trailing commas before } or ].
    normalized = re.sub(r",\s*([}\]])", r"\1", normalized)
    config = json.loads(normalized)
    if not isinstance(config, dict):
        raise ValueError("--config must decode to a JSON object")
    return config


def decode_config_arg(config_text: str | None) -> dict[str, Any]:
    if config_text is None:
        return {}

    text = config_text.strip()
    if not text or text == "None":
        return {}

    # If it looks like inline JSON, parse directly (with relaxed trailing-comma support).
    if text.startswith("{") or text.startswith("["):
        return _parse_json_with_optional_trailing_commas(text)

    # Otherwise treat as base64-encoded JSON.
    try:
        decoded = base64.b64decode(text, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as e:
        raise ValueError(
            "--config is neither valid JSON nor valid base64-encoded JSON"
        ) from e

    try:
        return _parse_json_with_optional_trailing_commas(decoded)
    except json.JSONDecodeError as e:
        raise ValueError("Decoded --config is not valid JSON") from e


def run_test_compiler_for_sample(
    args: argparse.Namespace,
    sample_dir: Path,
    log_path: Path,
    pass_match_result_path: Path,
) -> int:
    allow_list = sample_dir / "graph_list.txt"
    if not allow_list.is_file():
        raise FileNotFoundError(f"graph_list.txt not found: {allow_list}")

    cmd = [
        sys.executable,
        "-m",
        "graph_net_bench.torch.test_compiler",
        "--model-path",
        args.model_path,
        "--model-path-prefix",
        str(sample_dir),
        "--allow-list",
        str(allow_list),
        "--compiler",
        args.compiler,
        "--device",
        args.device,
        "--warmup",
        str(args.warmup),
        "--trials",
        str(args.trials),
        "--log-prompt",
        args.log_prompt,
    ]

    if args.compiler == "pass_mgr":
        input_pass_rule_dir = sample_dir / args.pass_input_dir_name
        output_pass_rule_dir = sample_dir / args.pass_output_dir_name
        config = {
            "pass_match_result_file_path": str(pass_match_result_path),
            "input_pass_rule_dir": str(input_pass_rule_dir),
            "output_pass_rule_dir": str(output_pass_rule_dir),
            "output_pass_pattern_limit": args.output_pass_pattern_limit,
            "output_pass_replacement_func_limit": args.output_pass_replacement_func_limit,
        }

        # User config takes precedence over default pass_mgr config.
        config.update(args.user_config)
        cmd.extend(["--config", encode_config(config)])
    elif args.user_config:
        # For non-pass_mgr backends (e.g. inductor), pass user config through.
        cmd.extend(["--config", encode_config(args.user_config)])

    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_env = os.environ.copy()
    run_env["MKL_THREADING_LAYER"] = "GNU"
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=str(args.workspace_root),
            env=run_env,
            check=False,
        )
    return proc.returncode


def parse_pattern_match_from_log(log_text: str) -> bool | None:
    processing_count = log_text.count("[Processing]")
    if processing_count == 0:
        return None

    no_match_count = log_text.count(
        "[PassMgrBackend] Warning: No passes modified the graph. Returning original."
    )
    return no_match_count < processing_count


def compute_es_t_scores(
    log_path: Path,
    p: float,
    b: float,
    interpretation_type: str,
) -> dict[str, float]:
    samples = analysis_util.parse_logs_to_data(str(log_path))
    if not samples:
        return {}

    positive_tolerance_interpretation = get_positive_tolerance_interpretation(
        interpretation_type
    )
    es_scores = analysis_util.calculate_scores(
        samples,
        positive_tolerance_interpretation=positive_tolerance_interpretation,
        p=p,
        b=b,
        type="ESt",
    )
    return {str(k): float(v) for k, v in es_scores.items()}


def compute_overall_es_score(es_t_scores: dict[str, float]) -> float | None:
    if not es_t_scores:
        return None

    weights = get_weights()
    if set(str(k) for k in weights.keys()) != set(es_t_scores.keys()):
        return None

    weighted_sum = sum(
        weights[tolerance] * np.log(es_t_scores[str(tolerance)]) / np.log(10)
        for tolerance in weights.keys()
    )
    return float(10 ** float(weighted_sum))


def load_sample_lines(sample_list_path: Path) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    with sample_list_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            rows.append((idx, text))
    return rows


def discover_sample_dirs(samples_root: Path) -> list[tuple[int, str]]:
    graph_lists = sorted(samples_root.rglob("graph_list.txt"))
    rows = [(idx, str(path.parent)) for idx, path in enumerate(graph_lists, start=1)]
    return rows


def main(args) -> None:
    args.workspace_root = args.workspace_root.resolve()
    args.output_root = args.output_root.resolve()
    args.user_config = decode_config_arg(args.config)

    if args.sample_list is None and args.samples_root is None:
        raise ValueError("Please provide either --sample-list or --samples-root")

    output_json = (
        args.output_json.resolve()
        if args.output_json is not None
        else args.output_root / "results.json"
    )

    logs_dir = args.output_root / "logs"
    pass_match_dir = args.output_root / "pass_match"
    logs_dir.mkdir(parents=True, exist_ok=True)
    pass_match_dir.mkdir(parents=True, exist_ok=True)

    sample_rows: list[tuple[int, str]] = []
    if args.sample_list is not None:
        args.sample_list = args.sample_list.resolve()
        if not args.sample_list.is_file():
            raise FileNotFoundError(f"sample list not found: {args.sample_list}")
        sample_rows.extend(load_sample_lines(args.sample_list))

    if args.samples_root is not None:
        args.samples_root = args.samples_root.resolve()
        if not args.samples_root.is_dir():
            raise FileNotFoundError(f"samples root not found: {args.samples_root}")
        discovered = discover_sample_dirs(args.samples_root)
        sample_rows.extend(discovered)

    if not sample_rows:
        raise ValueError("No samples found from --sample-list/--samples-root")

    # Normalize line numbers after mixing multiple sources.
    sample_rows = [
        (idx, sample_path_text)
        for idx, (_, sample_path_text) in enumerate(sample_rows, start=1)
    ]
    results = []

    for line_no, sample_path_text in sample_rows:
        sample_dir = resolve_path(sample_path_text, args.workspace_root)
        log_path = logs_dir / f"{line_no:06d}.log"
        pass_match_file = pass_match_dir / f"{line_no:06d}.txt"

        item = {
            "line_no": line_no,
            "sample_path": str(sample_dir),
            "log_path": str(log_path),
            "return_code": None,
            "is_pattern_match": None,
            "es_t_scores": {},
            "es_overall_score": None,
            "error": None,
        }

        try:
            ret = run_test_compiler_for_sample(
                args=args,
                sample_dir=sample_dir,
                log_path=log_path,
                pass_match_result_path=pass_match_file,
            )
            item["return_code"] = ret

            log_text = log_path.read_text(encoding="utf-8", errors="ignore")
            item["is_pattern_match"] = parse_pattern_match_from_log(log_text)
            item["es_t_scores"] = compute_es_t_scores(
                log_path=log_path,
                p=args.negative_speedup_penalty,
                b=args.fpdb,
                interpretation_type=args.interpretation_type,
            )
            item["es_overall_score"] = compute_overall_es_score(item["es_t_scores"])
        except Exception as e:  # Keep processing other samples.
            item["error"] = str(e)

        results.append(item)
        print(
            f"line={line_no} ret={item['return_code']} pattern={item['is_pattern_match']} es_count={len(item['es_t_scores'])} es_overall={item['es_overall_score']}",
            flush=True,
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=True)

    print(f"Saved results to: {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-list",
        type=Path,
        required=False,
        help="Path to txt file that lists AI4C sample directories, one per line.",
    )
    parser.add_argument(
        "--samples-root",
        type=Path,
        default=None,
        help="Root directory of dumped AI4C samples (e.g., samples/deepseek-v3.2).",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=REPO_ROOT,
        help="Workspace root used to resolve relative sample paths.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "tmp" / "ai4c_eval_logs",
        help="Output root directory for per-line logs and final json.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output json file path. Default: <output-root>/results.json",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="graphs",
        help="Value passed to --model-path in test_compiler.",
    )
    parser.add_argument(
        "--compiler",
        type=str,
        default="pass_mgr",
        choices=["pass_mgr", "nope", "inductor"],
        help="Compiler backend passed to test_compiler.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Compiler config in base64-encoded JSON (preferred) or raw JSON string. "
            "For pass_mgr, this config overrides default pass_mgr config keys."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument(
        "--log-prompt",
        type=str,
        default="graph-net-test-compiler-log",
    )
    parser.add_argument(
        "--negative-speedup-penalty",
        type=float,
        default=0.0,
        help="Penalty power p used by ES(t) calculation.",
    )
    parser.add_argument(
        "--fpdb",
        type=float,
        default=0.001,
        help="Base penalty b used by ES(t) calculation.",
    )
    parser.add_argument(
        "--positive-tolerance-interpretation",
        dest="interpretation_type",
        choices=get_supported_positive_tolerance_interpretation_types(),
        default="default",
    )
    parser.add_argument(
        "--pass-input-dir-name",
        type=str,
        default="const_pass_dir",
        help="Relative dir name under sample dir for pass_mgr input_pass_rule_dir.",
    )
    parser.add_argument(
        "--pass-output-dir-name",
        type=str,
        default="pass_dir",
        help="Relative dir name under sample dir for pass_mgr output_pass_rule_dir.",
    )
    parser.add_argument(
        "--output-pass-pattern-limit",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--output-pass-replacement-func-limit",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    main(args)
