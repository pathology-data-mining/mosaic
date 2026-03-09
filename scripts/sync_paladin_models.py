#!/usr/bin/env python3
"""Sync paladin model checkpoints to HuggingFace.

Idempotent script for exporting, uploading, and registering paladin models.
Works for any project name (paladin-redo, or future variants).

Phases:
  gen-configs  -- Generate Hydra pkl configs for SLURM export jobs
  upload       -- Upload exported pkl files to HF repo (idempotent)
  update-map   -- Update paladin_model_map.csv locally and push to HF

Usage:
    python scripts/sync_paladin_models.py --project paladin-redo --phase gen-configs
    python scripts/sync_paladin_models.py --project paladin-redo --phase upload
    python scripts/sync_paladin_models.py --project paladin-redo --phase update-map

    # Or run all phases sequentially (gen-configs + upload + update-map),
    # except gen-configs requires SLURM to complete first:
    python scripts/sync_paladin_models.py --project paladin-redo
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MOSAIC_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUROC_CSV = MOSAIC_ROOT / "aggregated_auroc.freeze.corrected.csv"
MODEL_MAP_CSV = MOSAIC_ROOT / "data" / "paladin_model_map.csv"
HF_CRED_ENV = Path.home() / ".hf_cred.env"

# Paladin worktree (dev branch) — used for SLURM jobs and Hydra configs
# The main repo (/repos/paladin) is on feature/webdataset-integration which has
# an incompatible cancer_type API change. The dev worktree uses the stable API.
PALADIN_REPO = Path("/gpfs/mskmind_ess/limr/repos/paladin-dev")
PALADIN_CONF_DIR = PALADIN_REPO / "conf" / "train"
PALADIN_VENV = PALADIN_REPO / "venv"

# Boehmk's paladin storage (read-only for us — checkpoints live here)
PALADIN_STORAGE_BASE = Path("/gpfs/mskmind_ess/boehmk/paladin/storage")

# Our writable output dir for exported pkl files (boehmk's storage is read-only)
PKL_OUTPUT_BASE = Path("/gpfs/mskmind_ess/limr/paladin-redo-pkl")

HF_REPO_ID = "PDM-Group/paladin-aeon-models"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_hf_token() -> str:
    """Load HF_TOKEN from ~/.hf_cred.env or environment."""
    if "HF_TOKEN" in os.environ:
        return os.environ["HF_TOKEN"]
    if HF_CRED_ENV.exists():
        for line in HF_CRED_ENV.read_text().splitlines():
            line = line.strip()
            if line.startswith("export HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError(
        "HF_TOKEN not found. Set it in the environment or in ~/.hf_cred.env"
    )


def get_best_run_per_combo(project: str, csv_path: Path) -> dict[tuple[str, str], str]:
    """Return {(histology, target_name): run_id} for best-auroc run per combo."""
    df = pd.read_csv(csv_path)
    proj = df[df["project"] == project]
    if proj.empty:
        raise ValueError(f"No rows found for project={project!r} in {csv_path}")
    best = proj.sort_values("auroc", ascending=False).drop_duplicates(
        subset=["histology", "target_name"]
    )
    return dict(zip(zip(best["histology"], best["target_name"]), best["run_id"]))


def find_ckpt_zip(storage_dir: Path, run_id: str) -> Path:
    """Find the (single) .ckpt.zip file for a run_id."""
    pattern = str(storage_dir / run_id / "checkpoints" / "*.ckpt.zip")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No .ckpt.zip found for {run_id} in {storage_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple .ckpt.zip files found for {run_id}: {matches}")
    return Path(matches[0])


def parse_epoch_step(filename: str) -> str:
    """Parse 'epoch=E-step=S' from a checkpoint filename."""
    m = re.search(r"(epoch=\d+-step=\d+)", filename)
    if not m:
        raise ValueError(f"Cannot parse epoch/step from filename: {filename}")
    return m.group(1)


def find_pkl(storage_dir: Path, run_id: str) -> Path | None:
    """Return the pkl file path if it exists, else None.

    Checks both the dedicated PKL_OUTPUT_BASE (our writable area) and
    the original boehmk storage dir (in case pkls were already generated there).
    """
    # Primary: our output area
    out_dir = PKL_OUTPUT_BASE / run_id / "checkpoints"
    pattern = str(out_dir / "*.model.pkl")
    matches = glob.glob(pattern)
    if matches:
        return Path(matches[0])
    # Fallback: boehmk's storage (read-only, but may have pre-existing pkls)
    pattern = str(storage_dir / run_id / "checkpoints" / "*.model.pkl")
    matches = glob.glob(pattern)
    return Path(matches[0]) if matches else None


# ---------------------------------------------------------------------------
# Phase 1: gen-configs
# ---------------------------------------------------------------------------

PKL_YAML_TEMPLATE = """\
callbacks:
- _target_: lightning.pytorch.callbacks.EarlyStopping
  mode: ${{train.monitor.mode}}
  monitor: ${{train.monitor.metric}}
  patience: 5
  verbose: false
- _target_: lightning.pytorch.callbacks.ModelCheckpoint
  mode: ${{train.monitor.mode}}
  monitor: ${{train.monitor.metric}}
  save_top_k: 5
  verbose: false
- _target_: lightning.pytorch.callbacks.LearningRateMonitor
  log_momentum: false
  logging_interval: step
- _target_: lightning.pytorch.callbacks.progress.tqdm_progress.TQDMProgressBar
  refresh_rate: 5
- _target_: paladin.pl_modules.callbacks.ExportCallback
- _target_: paladin.pl_modules.callbacks.ConvertToONNX
  export_path: {export_path}
command: test
deterministic: false
logging:
  logger:
    _target_: lightning.pytorch.loggers.WandbLogger
    entity: null
    log_model: ${{..upload.run_files}}
    mode: disabled
    name: pkl_export
    project: paladin
    tags: ${{core.tags}}
  upload:
    run_files: false
    source: false
  wandb_watch:
    log: all
    log_freq: 5
monitor:
  metric: loss/val_epoch
  mode: min
restore:
  ckpt_or_run_path: {ckpt_path}
  mode: continue
seed_index: 0
trainer:
  accelerator: gpu
  deterministic: ${{train.deterministic}}
  devices: 1
  fast_dev_run: false
  max_epochs: 500
  max_steps: 1000000
  num_sanity_val_steps: 0
  precision: 16
  strategy: ddp
  val_check_interval: 1.0
"""

SBATCH_TEMPLATE = """\
#!/bin/bash
# Submit SLURM pickle-export jobs for {project} models.
# Generated by scripts/sync_paladin_models.py --phase gen-configs
# Run this from {paladin_repo}

PALADIN_REPO="{paladin_repo}"
PALADIN_CONDA_ENV="{paladin_venv}"

cd "$PALADIN_REPO"

RUN_IDS=(
{run_id_list}
)

echo "Submitting ${{#RUN_IDS[@]}} pickle-export jobs..."

for run_id in "${{RUN_IDS[@]}}"; do
    config_name="pkl_${{run_id}}"
    config_file="{conf_dir}/pkl_${{run_id}}.yaml"

    if [ ! -f "$config_file" ]; then
        echo "WARNING: config not found, skipping: $config_file"
        continue
    fi

    sbatch \\
        --nodes=1 \\
        --ntasks=1 \\
        --cpus-per-task=40 \\
        --mem=100G \\
        --time=1:00:00 \\
        --output="{paladin_repo}/logs/pickle_%A_%a.out" \\
        --error="{paladin_repo}/logs/pickle_%A_%a.err" \\
        --qos=premium \\
        --job-name="pkl_${{run_id}}" \\
        --gres=gpu:1 \\
        --wrap="source ~/.bashrc && conda activate {paladin_venv} && cd {paladin_repo} && WANDB_MODE=disabled python src/paladin/run.py train=${{config_name}} nn=joint nn/module=jointbb"

    sleep 0.5
done

echo "Done submitting!"
"""


def phase_gen_configs(project: str, csv_path: Path) -> None:
    """Generate Hydra pkl configs and sbatch submit script."""
    storage_dir = PALADIN_STORAGE_BASE / project
    if not storage_dir.exists():
        raise FileNotFoundError(f"Storage directory not found: {storage_dir}")

    combos = get_best_run_per_combo(project, csv_path)
    print(f"Found {len(combos)} (histology, target_name) combos for project={project!r}")

    PALADIN_CONF_DIR.mkdir(parents=True, exist_ok=True)
    logs_dir = PALADIN_REPO / "logs"
    logs_dir.mkdir(exist_ok=True)

    skipped, written, errors = [], [], []

    for (histology, target), run_id in combos.items():
        pkl_path = find_pkl(storage_dir, run_id)
        if pkl_path:
            skipped.append(run_id)
            continue

        try:
            ckpt_zip = find_ckpt_zip(storage_dir, run_id)
        except FileNotFoundError as e:
            errors.append((run_id, str(e)))
            continue

        epoch_step = parse_epoch_step(ckpt_zip.name)
        # Write pkl to our own writable area (boehmk's storage is read-only)
        out_ckpt_dir = PKL_OUTPUT_BASE / run_id / "checkpoints"
        out_ckpt_dir.mkdir(parents=True, exist_ok=True)
        export_path = str(out_ckpt_dir / f"{run_id}.{epoch_step}.model.onnx")

        yaml_content = PKL_YAML_TEMPLATE.format(
            export_path=export_path,
            ckpt_path=str(ckpt_zip),
        )

        config_file = PALADIN_CONF_DIR / f"pkl_{run_id}.yaml"
        config_file.write_text(yaml_content)
        written.append(run_id)

    print(f"  Written: {len(written)} configs")
    print(f"  Skipped (pkl exists): {len(skipped)}")
    if errors:
        print(f"  ERRORS ({len(errors)}):")
        for run_id, msg in errors:
            print(f"    {run_id}: {msg}")

    # Write the sbatch submit script
    needs_export = [
        run_id
        for run_id in combos.values()
        if not find_pkl(storage_dir, run_id)
    ]

    run_id_list = "\n".join(f"    {r}" for r in needs_export)
    sbatch_content = SBATCH_TEMPLATE.format(
        project=project,
        paladin_repo=str(PALADIN_REPO),
        paladin_venv=str(PALADIN_VENV),
        conf_dir=str(PALADIN_CONF_DIR),
        run_id_list=run_id_list,
    )

    submit_script = MOSAIC_ROOT / "scripts" / f"submit_{project.replace('-', '_')}_pickles.sh"
    submit_script.write_text(sbatch_content)
    submit_script.chmod(0o755)
    print(f"\nSubmit script written to: {submit_script}")
    print(f"Run it from: {PALADIN_REPO}")
    print("  cd /gpfs/mskmind_ess/limr/repos/paladin")
    print(f"  bash {submit_script}")


# ---------------------------------------------------------------------------
# Phase 2: upload
# ---------------------------------------------------------------------------


def phase_upload(project: str, csv_path: Path) -> None:
    """Upload pkl files from storage to HuggingFace (idempotent).

    Batches all files into a single commit for efficiency.
    """
    from huggingface_hub import HfApi, CommitOperationAdd

    token = load_hf_token()
    api = HfApi(token=token)

    storage_dir = PALADIN_STORAGE_BASE / project
    hf_folder = project.replace("-", "_")  # paladin-redo -> paladin_redo

    # Get existing files in HF repo for this project folder
    print(f"Listing existing files in HF repo ({hf_folder}/)...")
    try:
        repo_files = set(api.list_repo_files(HF_REPO_ID, repo_type="model"))
    except Exception as e:
        raise RuntimeError(f"Failed to list HF repo files: {e}") from e

    existing_hf = {f for f in repo_files if f.startswith(f"{hf_folder}/")}
    print(f"  Already in HF ({hf_folder}/): {len(existing_hf)}")

    # Find all pkl files — check our writable output area first, then boehmk's storage
    pkl_files = glob.glob(str(PKL_OUTPUT_BASE / "*" / "checkpoints" / "*.model.pkl"))
    pkl_files += glob.glob(str(storage_dir / "*" / "checkpoints" / "*.model.pkl"))
    print(f"  Found {len(pkl_files)} pkl files (output area + boehmk storage)")

    combos = get_best_run_per_combo(project, csv_path)
    expected_run_ids = set(combos.values())

    # Deduplicate by hf_path (output area takes priority over boehmk storage)
    seen_hf_paths: set[str] = set()
    to_upload: list[tuple[str, Path]] = []  # (hf_path, local_path)

    for pkl_path_str in pkl_files:
        pkl_path = Path(pkl_path_str)
        run_id = pkl_path.parent.parent.name  # .../paladin-redo/{run_id}/checkpoints/...

        if run_id not in expected_run_ids:
            continue  # not a best-run pkl, skip

        hf_path = f"{hf_folder}/{pkl_path.name}"

        if hf_path in seen_hf_paths:
            continue  # dedup
        seen_hf_paths.add(hf_path)

        if hf_path in existing_hf:
            continue  # already uploaded

        to_upload.append((hf_path, pkl_path))

    skipped = len(seen_hf_paths) - len(to_upload)
    print(f"  To upload: {len(to_upload)}  |  Already in HF (skipped): {skipped}")

    # Check for missing pkls (SLURM not done yet)
    found_run_ids = {
        Path(p).parent.parent.name
        for p in pkl_files
        if Path(p).parent.parent.name in expected_run_ids
    }
    missing = expected_run_ids - found_run_ids
    if missing:
        print(f"  WARNING: Missing pkl files (SLURM not done?): {len(missing)}")
        for r in sorted(missing)[:10]:
            print(f"    {r}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")
        sys.exit(1)

    if not to_upload:
        print("Nothing to upload.")
        return

    # Batch into commits of up to BATCH_SIZE files to avoid oversized commits
    BATCH_SIZE = 50
    total_uploaded = 0
    for batch_start in range(0, len(to_upload), BATCH_SIZE):
        batch = to_upload[batch_start : batch_start + BATCH_SIZE]
        batch_end = batch_start + len(batch)
        print(f"  Committing files {batch_start + 1}–{batch_end} of {len(to_upload)} ...", flush=True)
        operations = [
            CommitOperationAdd(path_in_repo=hf_path, path_or_fileobj=str(local_path))
            for hf_path, local_path in batch
        ]
        api.create_commit(
            repo_id=HF_REPO_ID,
            repo_type="model",
            operations=operations,
            commit_message=f"Add {project} pkl files ({batch_start + 1}–{batch_end})",
        )
        total_uploaded += len(batch)
        print(f"    ok ({total_uploaded} uploaded so far)")

    print(f"\nUpload summary:")
    print(f"  Uploaded: {total_uploaded}")
    print(f"  Already in HF (skipped): {skipped}")


# ---------------------------------------------------------------------------
# Phase 3: update-map
# ---------------------------------------------------------------------------


def phase_update_map(project: str, csv_path: Path) -> None:
    """Update paladin_model_map.csv and push to HuggingFace."""
    from huggingface_hub import HfApi

    token = load_hf_token()
    api = HfApi(token=token)

    storage_dir = PALADIN_STORAGE_BASE / project
    hf_folder = project.replace("-", "_")  # paladin-redo -> paladin_redo

    combos = get_best_run_per_combo(project, csv_path)
    print(f"Processing {len(combos)} combos for project={project!r}")

    # Build (histology, target_name) -> model_path mapping
    new_entries: list[dict] = []
    missing: list[str] = []

    for (histology, target_name), run_id in combos.items():
        pkl = find_pkl(storage_dir, run_id)
        if pkl is None:
            missing.append(run_id)
            continue

        model_path = f"data/{hf_folder}/{pkl.name}"
        new_entries.append(
            {
                "cancer_subtype": histology,
                "target_name": target_name,
                "model_path": model_path,
            }
        )

    if missing:
        print(f"ERROR: {len(missing)} run_ids have no pkl file yet (run upload phase first?):")
        for r in sorted(missing)[:10]:
            print(f"  {r}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
        sys.exit(1)

    # Load existing model map
    model_map = pd.read_csv(MODEL_MAP_CSV)
    existing_keys = set(zip(model_map["cancer_subtype"], model_map["target_name"]))

    replaced, added = 0, 0
    for entry in new_entries:
        key = (entry["cancer_subtype"], entry["target_name"])
        if key in existing_keys:
            idx = model_map[
                (model_map["cancer_subtype"] == entry["cancer_subtype"])
                & (model_map["target_name"] == entry["target_name"])
            ].index
            model_map.loc[idx, "model_path"] = entry["model_path"]
            replaced += 1
        else:
            new_row = pd.DataFrame([entry])
            model_map = pd.concat([model_map, new_row], ignore_index=True)
            added += 1

    model_map.to_csv(MODEL_MAP_CSV, index=False)
    print(f"\npaladin_model_map.csv updated:")
    print(f"  Replaced: {replaced}")
    print(f"  Added: {added}")
    print(f"  Total entries: {len(model_map)}")
    print(f"  Saved to: {MODEL_MAP_CSV}")

    # Push updated CSV to HF
    print(f"\nUploading paladin_model_map.csv to HF...")
    api.upload_file(
        path_or_fileobj=str(MODEL_MAP_CSV),
        path_in_repo="paladin_model_map.csv",
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message=f"Update paladin_model_map.csv for {project}",
    )
    print("Done!")
    print(f"\nNext step: commit and push the updated model map:")
    print(f"  git add {MODEL_MAP_CSV.relative_to(MOSAIC_ROOT)}")
    print(f"  git commit -m 'feat: add {project} models to paladin_model_map.csv'")
    print(f"  git push")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--project",
        required=True,
        help="Project name in aggregated_auroc CSV (e.g. paladin-redo)",
    )
    parser.add_argument(
        "--phase",
        choices=["gen-configs", "upload", "update-map"],
        default=None,
        help="Phase to run. If omitted, runs upload + update-map (gen-configs requires SLURM).",
    )
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_AUROC_CSV),
        help=f"Path to aggregated AUROC CSV (default: {DEFAULT_AUROC_CSV})",
    )
    args = parser.parse_args()
    csv_path = Path(args.csv)

    if args.phase == "gen-configs":
        phase_gen_configs(args.project, csv_path)
    elif args.phase == "upload":
        phase_upload(args.project, csv_path)
    elif args.phase == "update-map":
        phase_update_map(args.project, csv_path)
    else:
        # Run upload + update-map (gen-configs requires separate SLURM step)
        print("Running upload + update-map phases (skipping gen-configs)...")
        phase_upload(args.project, csv_path)
        phase_update_map(args.project, csv_path)


if __name__ == "__main__":
    main()
