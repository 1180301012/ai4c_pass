#!/usr/bin/env python3
"""Replay patch outputs from AI4C/R2E trajectory files.

This script reads a trajectory JSON/JSONL file, extracts a patch-like payload
(typically `output_patch`), and reconstructs pass files.

It supports two workflows:
1) pass-only replay into a provided `--target-dir`
2) full sample build (`--sample-output-root`) that creates a sample directory
    similar to `ai4c_agent/tmp/samples_glm45air`, including:
    - patched `pass_dir/*`
    - copied `graph_list.txt` / `sample_uids.txt` from source sample
    - optional symlinks for `graph_net_bench` and `entry.sh`
    - `graphs/` subset copied strictly from `graph_list.txt`

"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


HEADER_RE = re.compile(r"^-{8,}\s+(.+?)\s+-{8,}\s*$")
DEFAULT_PATCH_FIELDS = (
    "output_patch",
    "patch",
    "final_patch",
    "generated_patch",
)


@dataclass
class PatchRecord:
    index: int
    patch_text: str
    source_field: str
    row: dict


@dataclass
class ReplaySummary:
    records_total: int = 0
    records_done: int = 0
    records_failed: int = 0
    pass_candidates: int = 0
    pass_written: int = 0
    pass_skipped: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay patch sections from an R2E/AI4C trajectory file into a target directory."
        )
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        required=True,
        help="Path to trajectory file (.jsonl or .json).",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=None,
        help="Directory that contains sorted_output_pass_rule_names.json and pass *.py files.",
    )
    parser.add_argument(
        "--record-index",
        type=int,
        default=-1,
        help=(
            "Index of selected patch record among all records containing patch text. "
            "Supports negative index, default: -1 (last)."
        ),
    )
    parser.add_argument(
        "--all-records",
        action="store_true",
        help="Replay all records containing patch text instead of selecting one record.",
    )
    parser.add_argument(
        "--patch-field",
        type=str,
        default="",
        help=(
            "Explicit field name containing patch text. "
            "If omitted, auto-detect from common fields."
        ),
    )
    parser.add_argument(
        "--strip-prefix",
        type=str,
        default="pass_dir/",
        help="Strip this prefix from each patch section path before writing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files. Default: skip existing files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned writes without modifying files.",
    )
    parser.add_argument(
        "--include-non-pass-files",
        action="store_true",
        help=(
            "Also materialize non-pass files (e.g. score.txt). By default only "
            "*.py and sorted_output_pass_rule_names.json are written."
        ),
    )
    parser.add_argument(
        "--sample-output-root",
        type=Path,
        default=None,
        help=(
            "If set, build a full sample directory under this root (e.g. "
            "ai4c_agent/tmp/samples_glm45air) using ds.sample_dir from trajectory."
        ),
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path.cwd(),
        help="Workspace root that contains samples/, graphs/, graph_net_bench/, entry_scripts/.",
    )
    parser.add_argument(
        "--source-sample-root",
        type=Path,
        default=None,
        help=(
            "Optional override for source sample root. If omitted, source sample path "
            "is resolved from workspace-root and ds.sample_dir."
        ),
    )
    parser.add_argument(
        "--no-link-graph-net-bench",
        action="store_true",
        help="Do not create graph_net_bench symlink in full-sample mode.",
    )
    parser.add_argument(
        "--no-link-entry-sh",
        action="store_true",
        help="Do not create entry.sh symlink in full-sample mode.",
    )
    parser.add_argument(
        "--skip-copy-graphs",
        action="store_true",
        help="Skip creating subset graphs/ from graph_list.txt in full-sample mode.",
    )
    parser.add_argument(
        "--sample-dir",
        type=str,
        default="",
        help="Explicit sample_dir override (relative path like samples/.../instance_id).",
    )
    return parser.parse_args()


def load_json_or_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # Ignore malformed lines to keep robust on noisy logs.
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
        return rows

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        return [obj]

    raise ValueError(f"Unsupported JSON root type in {path}: {type(obj).__name__}")


def iter_patch_records(rows: list[dict], patch_field: str) -> Iterable[PatchRecord]:
    for idx, row in enumerate(rows):
        fields = (patch_field,) if patch_field else DEFAULT_PATCH_FIELDS
        for field in fields:
            value = row.get(field)
            if isinstance(value, str) and value.strip():
                yield PatchRecord(index=idx, patch_text=value, source_field=field, row=row)
                break


def pick_record(records: list[PatchRecord], record_index: int) -> PatchRecord:
    if not records:
        raise ValueError("No patch text found in trajectory.")

    try:
        return records[record_index]
    except IndexError as exc:
        raise IndexError(
            f"record-index {record_index} is out of range for {len(records)} records"
        ) from exc


def parse_output_patch_sections(patch_text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_path: str | None = None
    current_lines: list[str] = []

    for line in patch_text.splitlines():
        match = HEADER_RE.match(line)
        if match:
            if current_path is not None:
                sections.append((current_path, "\n".join(current_lines).rstrip("\n") + "\n"))
            current_path = match.group(1).strip()
            current_lines = []
            continue

        if current_path is not None:
            current_lines.append(line)

    if current_path is not None:
        sections.append((current_path, "\n".join(current_lines).rstrip("\n") + "\n"))

    return sections


def is_allowed_output(path_str: str, include_non_pass_files: bool) -> bool:
    name = Path(path_str).name
    if name == "sorted_output_pass_rule_names.json":
        return True
    if name == "score.txt":
        return True
    if path_str.endswith(".py"):
        return True
    if include_non_pass_files:
        return True
    return False


def secure_join(base: Path, rel: Path) -> Path:
    base_resolved = base.resolve()
    dst = (base / rel).resolve()
    if not str(dst).startswith(str(base_resolved) + "/") and dst != base_resolved:
        raise ValueError(f"Unsafe path (outside target-dir): {rel}")
    return dst


def map_section_path(path_str: str, strip_prefix: str) -> Path:
    normalized = path_str.strip().replace("\\", "/")
    if strip_prefix and normalized.startswith(strip_prefix):
        normalized = normalized[len(strip_prefix) :]
    normalized = normalized.lstrip("/")
    return Path(normalized)


def normalize_sample_dir(sample_dir: str) -> Path:
    text = sample_dir.strip().replace("\\", "/").lstrip("/")
    return Path(text)


def sample_output_relpath(sample_dir: Path) -> Path:
    parts = sample_dir.parts
    if parts and parts[0] == "samples":
        return Path(*parts[1:])
    return sample_dir


def remove_path(path: Path, dry_run: bool) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if dry_run:
        print(f"[DRY ] remove: {path}")
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    shutil.rmtree(path)


def copy_file_if_exists(src: Path, dst: Path, overwrite: bool, dry_run: bool) -> bool:
    if not src.is_file():
        return False
    if dst.exists() and not overwrite:
        print(f"[SKIP] exists: {dst}")
        return False
    if dry_run:
        print(f"[DRY ] copy : {src} -> {dst}")
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[COPY] {src} -> {dst}")
    return True


def ensure_symlink(dst: Path, src: Path, overwrite: bool, dry_run: bool) -> bool:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            print(f"[SKIP] exists: {dst}")
            return False
        remove_path(dst, dry_run=dry_run)
    rel_target = Path(os.path.relpath(str(src), str(dst.parent)))
    if dry_run:
        print(f"[DRY ] link : {dst} -> {rel_target}")
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(rel_target)
    print(f"[LINK] {dst} -> {rel_target}")
    return True


def read_graph_list(graph_list_file: Path) -> list[Path]:
    if not graph_list_file.is_file():
        return []
    lines = graph_list_file.read_text(encoding="utf-8").splitlines()
    out: list[Path] = []
    for raw in lines:
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        out.append(Path(text.replace("\\", "/")))
    return out


def copy_graph_subset(
    graph_list_entries: list[Path],
    workspace_root: Path,
    sample_dir_out: Path,
    overwrite: bool,
    dry_run: bool,
) -> tuple[int, int, int]:
    candidate = 0
    copied = 0
    missing = 0
    for rel_path in graph_list_entries:
        candidate += 1
        rel_norm = Path(str(rel_path).lstrip("/"))
        src = secure_join(workspace_root, rel_norm)
        dst = secure_join(sample_dir_out, rel_norm)

        if not src.exists():
            missing += 1
            print(f"[MISS] graph case not found: {src}")
            continue

        if dst.exists() and not overwrite:
            print(f"[SKIP] exists: {dst}")
            continue

        if dst.exists() and overwrite:
            remove_path(dst, dry_run=dry_run)

        if dry_run:
            print(f"[DRY ] graph: {src} -> {dst}")
            copied += 1
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        copied += 1
        print(f"[COPY] graph: {src} -> {dst}")
    return candidate, copied, missing


def build_full_sample(
    record: PatchRecord,
    sections: list[tuple[str, str]],
    workspace_root: Path,
    sample_output_root: Path,
    source_sample_root: Path | None,
    sample_dir_override: str,
    strip_prefix: str,
    overwrite: bool,
    dry_run: bool,
    include_non_pass_files: bool,
    link_graph_net_bench: bool,
    link_entry_sh: bool,
    copy_graphs: bool,
) -> tuple[int, int, int]:
    ds = record.row.get("ds") if isinstance(record.row.get("ds"), dict) else {}
    sample_dir_text = sample_dir_override.strip() or str(ds.get("sample_dir") or "")
    if not sample_dir_text:
        raise ValueError("sample_dir not found. Provide --sample-dir or ensure ds.sample_dir exists.")

    sample_dir_rel = normalize_sample_dir(sample_dir_text)
    sample_out = secure_join(sample_output_root.resolve(), sample_output_relpath(sample_dir_rel))
    source_sample = (
        secure_join(source_sample_root.resolve(), sample_dir_rel)
        if source_sample_root is not None
        else secure_join(workspace_root.resolve(), sample_dir_rel)
    )

    if not dry_run:
        sample_out.mkdir(parents=True, exist_ok=True)
    print(f"Sample source: {source_sample}")
    print(f"Sample output: {sample_out}")

    pass_target_dir = sample_out / "pass_dir"
    if not dry_run:
        pass_target_dir.mkdir(parents=True, exist_ok=True)

    total, written, skipped = replay_sections(
        sections=sections,
        target_dir=pass_target_dir,
        strip_prefix=strip_prefix,
        overwrite=overwrite,
        dry_run=dry_run,
        include_non_pass_files=include_non_pass_files,
    )

    copied_meta = 0
    copied_meta += int(
        copy_file_if_exists(
            source_sample / "graph_list.txt",
            sample_out / "graph_list.txt",
            overwrite=overwrite,
            dry_run=dry_run,
        )
    )
    copied_meta += int(
        copy_file_if_exists(
            source_sample / "sample_uids.txt",
            sample_out / "sample_uids.txt",
            overwrite=overwrite,
            dry_run=dry_run,
        )
    )

    link_count = 0
    if link_graph_net_bench:
        link_count += int(
            ensure_symlink(
                sample_out / "graph_net_bench",
                workspace_root / "graph_net_bench",
                overwrite=overwrite,
                dry_run=dry_run,
            )
        )
    if link_entry_sh:
        link_count += int(
            ensure_symlink(
                sample_out / "entry.sh",
                workspace_root / "entry_scripts" / "entry.sh",
                overwrite=overwrite,
                dry_run=dry_run,
            )
        )

    graph_candidate = graph_copied = graph_missing = 0
    if copy_graphs:
        graph_list_file = source_sample / "graph_list.txt" if dry_run else sample_out / "graph_list.txt"
        graph_entries = read_graph_list(graph_list_file)
        if not graph_entries:
            print(f"[WARN] graph_list is empty or missing: {graph_list_file}")
        graph_candidate, graph_copied, graph_missing = copy_graph_subset(
            graph_list_entries=graph_entries,
            workspace_root=workspace_root,
            sample_dir_out=sample_out,
            overwrite=overwrite,
            dry_run=dry_run,
        )

    print("\nFull-sample done.")
    print(f"  pass candidate files: {total}")
    print(f"  pass written:         {written}")
    print(f"  pass skipped:         {skipped}")
    print(f"  meta copied:          {copied_meta}")
    print(f"  links created:        {link_count}")
    if copy_graphs:
        print(f"  graph candidates:     {graph_candidate}")
        print(f"  graph copied:         {graph_copied}")
        print(f"  graph missing:        {graph_missing}")
    return total, written, skipped


def record_instance_tag(record: PatchRecord) -> str:
    ds = record.row.get("ds") if isinstance(record.row.get("ds"), dict) else {}
    instance_id = ds.get("instance_id")
    if isinstance(instance_id, str) and instance_id.strip():
        return instance_id.strip()
    return f"record_{record.index:04d}"


def parse_sections_or_raise(record: PatchRecord) -> list[tuple[str, str]]:
    sections = parse_output_patch_sections(record.patch_text)
    if not sections:
        raise ValueError(
            "Selected patch text does not match expected section format. "
            "Expected headers like: -------------------- pass_dir/Foo.py --------------------"
        )
    return sections


def replay_sections(
    sections: list[tuple[str, str]],
    target_dir: Path,
    strip_prefix: str,
    overwrite: bool,
    dry_run: bool,
    include_non_pass_files: bool,
) -> tuple[int, int, int]:
    total = 0
    written = 0
    skipped = 0

    for src_path, content in sections:
        if not is_allowed_output(src_path, include_non_pass_files):
            continue

        total += 1
        rel = map_section_path(src_path, strip_prefix=strip_prefix)
        dst = secure_join(target_dir, rel)

        if dst.exists() and not overwrite:
            skipped += 1
            print(f"[SKIP] exists: {dst}")
            continue

        if dry_run:
            print(f"[DRY ] write: {dst} ({len(content.encode('utf-8'))} bytes)")
            written += 1
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(content, encoding="utf-8")
        print(f"[WRITE] {dst}")
        written += 1

    return total, written, skipped


def main() -> None:
    args = parse_args()

    rows = load_json_or_jsonl(args.trajectory)
    records = list(iter_patch_records(rows, patch_field=args.patch_field.strip()))
    selected_records: list[PatchRecord]
    if args.all_records:
        selected_records = records
    else:
        selected_records = [pick_record(records, args.record_index)]

    if args.target_dir is None and args.sample_output_root is None:
        raise ValueError("Either --target-dir or --sample-output-root must be provided.")

    summary = ReplaySummary(records_total=len(selected_records))
    workspace_root = args.workspace_root.resolve()
    sample_output_root = args.sample_output_root.resolve() if args.sample_output_root else None
    source_sample_root = (
        args.source_sample_root.resolve() if args.source_sample_root is not None else None
    )

    for i, record in enumerate(selected_records, start=1):
        try:
            sections = parse_sections_or_raise(record)
            print(
                f"\n[{i}/{len(selected_records)}] record row={record.index}, "
                f"field={record.source_field}, sections={len(sections)}"
            )

            if sample_output_root is not None:
                total, written, skipped = build_full_sample(
                    record=record,
                    sections=sections,
                    workspace_root=workspace_root,
                    sample_output_root=sample_output_root,
                    source_sample_root=source_sample_root,
                    sample_dir_override=args.sample_dir,
                    strip_prefix=args.strip_prefix,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                    include_non_pass_files=args.include_non_pass_files,
                    link_graph_net_bench=not args.no_link_graph_net_bench,
                    link_entry_sh=not args.no_link_entry_sh,
                    copy_graphs=not args.skip_copy_graphs,
                )
            else:
                target_root = args.target_dir.resolve()
                if args.all_records:
                    target_dir = target_root / record_instance_tag(record)
                else:
                    target_dir = target_root

                if not args.dry_run:
                    target_dir.mkdir(parents=True, exist_ok=True)
                print(f"Target dir: {target_dir}")

                total, written, skipped = replay_sections(
                    sections=sections,
                    target_dir=target_dir,
                    strip_prefix=args.strip_prefix,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                    include_non_pass_files=args.include_non_pass_files,
                )

                print("Done.")
                print(f"  candidate files: {total}")
                print(f"  written:         {written}")
                print(f"  skipped:         {skipped}")

            summary.records_done += 1
            summary.pass_candidates += total
            summary.pass_written += written
            summary.pass_skipped += skipped
        except Exception as exc:
            summary.records_failed += 1
            print(f"[ERROR] record row={record.index} failed: {exc}")
            if not args.all_records:
                raise

    print("\nBatch summary")
    print(f"  records total:   {summary.records_total}")
    print(f"  records done:    {summary.records_done}")
    print(f"  records failed:  {summary.records_failed}")
    print(f"  pass candidates: {summary.pass_candidates}")
    print(f"  pass written:    {summary.pass_written}")
    print(f"  pass skipped:    {summary.pass_skipped}")


if __name__ == "__main__":
    main()
