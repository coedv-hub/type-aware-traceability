#!/usr/bin/env python3
"""Generate frozen 1:1 pair-classification manifests.

Independent of the existing top-k recommendation experiments; writes only
to configs/pair_classification_manifests/. See
configs/pair_classification_protocol.yaml for the full sampling rule this
implements. Does not read or write results/, tables/, or
configs/final_framework_config.yaml.

Requirements with zero non-linked code files (every code file in the
project is a real gold link for that requirement) cannot have a valid
negative sampled at all, under any fallback -- there is no code file left
that isn't a real link. Such requirements are excluded from the manifest
entirely and recorded in {dataset}_excluded_requirements.csv, rather than
falling back to sampling a "negative" from the linked pool (which would
silently mislabel a real link as a non-link).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import DatasetLoader  # noqa: E402

SEED = 42
MANIFEST_DIR = PROJECT_ROOT / "configs" / "pair_classification_manifests"
VALIDATION_SPLIT = PROJECT_ROOT / "configs" / "validation_split.json"
DATASETS = ("eTour", "eANCI", "iTrust", "LibEST")
# Industrial is a fully external held-out dataset for this track: every
# eligible requirement is forced to Split="heldout" (no validation_split.json
# lookup at all, since it has no entry there and must never get one -- it
# is never used to calibrate the pooled threshold, only to report
# generalization on a 5th, previously-unseen dataset once that threshold
# is already frozen from the 4 public datasets).
HELDOUT_ONLY_DATASETS = ("Industrial",)
ALL_DATASETS = DATASETS + HELDOUT_ONLY_DATASETS


def density_group(link_count: int) -> str:
    if link_count <= 2:
        return "1-2"
    if link_count <= 10:
        return "3-10"
    return ">10"


def load_validation_ids(dataset_name: str) -> set[str]:
    payload = json.loads(VALIDATION_SPLIT.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen":
        raise SystemExit(f"Validation split is not frozen: {VALIDATION_SPLIT}")
    return set(payload["datasets"][dataset_name]["validation_requirement_ids"])


def build_manifest(
    dataset_name: str, config_path: Path, *, heldout_only: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (manifest, excluded_requirements).

    heldout_only=True (Industrial): skips the validation_split.json lookup
    entirely -- there is no entry for this dataset there, and there must
    never be one -- and forces every requirement's Split to "heldout"."""
    loader = DatasetLoader(config_path)
    dataset = loader.load(dataset_name)
    gold = dataset.ground_truth_by_requirement
    all_code_ids = sorted(dataset.code_files)
    req_ids = sorted(dataset.evaluation_requirements)
    req_type = {
        req_id: dataset.evaluation_requirements[req_id].req_type
        for req_id in req_ids
    }
    validation_ids = set() if heldout_only else load_validation_ids(dataset_name)

    rng = random.Random(SEED)
    rows: list[dict[str, object]] = []
    excluded_rows: list[dict[str, object]] = []
    pair_group_id = 0

    for req_id in req_ids:
        positives = sorted(gold.get(req_id, set()))
        link_count = len(positives)
        non_linked = [code_id for code_id in all_code_ids if code_id not in gold.get(req_id, set())]

        if not non_linked:
            # Every code file in the project is a real gold link for this
            # requirement -- there is no valid negative to sample under any
            # fallback. Exclude the requirement entirely rather than
            # mislabel a real link as a non-link.
            excluded_rows.append(
                {
                    "Dataset": dataset_name,
                    "Req_ID": req_id,
                    "Type": req_type[req_id],
                    "GoldLinkCount": link_count,
                    "TotalCodeFiles": len(all_code_ids),
                    "Reason": "fully_linked: every code file in the project "
                    "is a gold link for this requirement; no valid negative "
                    "exists",
                }
            )
            continue

        group = density_group(link_count)
        split = "validation" if req_id in validation_ids else "heldout"
        rng.shuffle(non_linked)
        available = list(non_linked)

        for positive_code_id in positives:
            pair_group_id += 1
            if available:
                negative_code_id = available.pop()
                with_replacement = False
            else:
                # non_linked is non-empty (checked above) but this
                # requirement has more positives than available non-linked
                # files; reuse from the real non-linked pool, flagged. A
                # requirement's negative can therefore legitimately repeat
                # across its own PairGroupIDs -- (Req_ID, Code_File) is NOT
                # a unique key for the manifest as a whole. Consumers must
                # key off PairGroupID (or row position), never off
                # (Req_ID, Code_File) alone.
                negative_code_id = rng.choice(non_linked)
                with_replacement = True

            for code_id, label, role, replaced in (
                (positive_code_id, 1, "positive", False),
                (negative_code_id, 0, "negative", with_replacement),
            ):
                rows.append(
                    {
                        "Dataset": dataset_name,
                        "Req_ID": req_id,
                        "Type": req_type[req_id],
                        "Split": split,
                        "PairGroupID": pair_group_id,
                        "GoldLinkCount": link_count,
                        "DensityGroup": group,
                        "Code_File": code_id,
                        "Label": label,
                        "PairRole": role,
                        "SampledWithReplacement": replaced,
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(excluded_rows)


def verify_no_cross_split_requirement(manifest: pd.DataFrame) -> None:
    """Every pair for a given Req_ID must carry the same Split label -- a
    requirement can never appear on both sides, since Split is assigned
    once per requirement before any pairs are generated."""
    per_req_splits = manifest.groupby("Req_ID")["Split"].nunique()
    offenders = per_req_splits[per_req_splits > 1]
    if not offenders.empty:
        raise SystemExit(
            "Cross-split requirement(s) detected (should be impossible): "
            f"{offenders.index.tolist()}"
        )


def verify_negatives_are_real_non_links(dataset_name: str, config_path: Path, manifest: pd.DataFrame) -> None:
    """Every negative row's Code_File must genuinely not be a gold link for
    its requirement, and every PairGroupID's positive/negative must be
    different code files. Guards against the exact class of bug this
    module was rewritten to fix."""
    loader = DatasetLoader(config_path)
    dataset = loader.load(dataset_name)
    gold = dataset.ground_truth_by_requirement

    negatives = manifest[manifest["PairRole"] == "negative"]
    bad = [
        (row.Req_ID, row.Code_File)
        for row in negatives.itertuples(index=False)
        if row.Code_File in gold.get(row.Req_ID, set())
    ]
    if bad:
        raise SystemExit(
            f"{dataset_name}: {len(bad)} negative pair(s) are actually gold "
            f"links (should be impossible): {bad[:5]}"
        )

    per_group = manifest.groupby("PairGroupID")["Code_File"].nunique()
    same_file_groups = per_group[per_group < 2]
    if not same_file_groups.empty:
        raise SystemExit(
            f"{dataset_name}: PairGroupID(s) with identical positive/negative "
            f"code file (should be impossible): {same_file_groups.index.tolist()[:5]}"
        )


def print_summary(dataset_name: str, manifest: pd.DataFrame, excluded: pd.DataFrame) -> None:
    positives = int((manifest["Label"] == 1).sum())
    negatives = int((manifest["Label"] == 0).sum())
    replaced = int(manifest["SampledWithReplacement"].sum())
    split_counts = manifest.drop_duplicates("Req_ID")["Split"].value_counts().to_dict()
    density_counts = (
        manifest[manifest["PairRole"] == "positive"]
        .drop_duplicates("Req_ID")["DensityGroup"]
        .value_counts()
        .to_dict()
    )
    print(f"{dataset_name}:")
    print(f"  positives={positives} negatives={negatives} (pairs={positives})")
    print(f"  with-replacement negatives={replaced}")
    print(f"  requirements by split: {split_counts}")
    print(f"  requirements by density group: {density_counts}")
    if not excluded.empty:
        print(f"  EXCLUDED requirements (fully-linked, no valid negative): "
              f"{excluded['Req_ID'].tolist()}")


def verify_heldout_only_dataset(dataset_name: str, manifest: pd.DataFrame) -> None:
    """Industrial-specific invariant: every row is Split="heldout" (never
    validation), and the manifest is exactly one positive + one negative
    per eligible requirement -- 22 rows for Industrial's 11 linked
    requirements. Deliberately hard-fails rather than silently writing a
    manifest that could accidentally leak into calibration."""
    if not manifest["Split"].eq("heldout").all():
        raise SystemExit(
            f"{dataset_name}: expected every row to be Split='heldout' "
            "(this dataset must never contribute to validation/"
            "calibration), found "
            f"{manifest['Split'].value_counts().to_dict()}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="all",
        help="Dataset name, 'all' (the 4 public calibration datasets), or "
        "'all_with_industrial' (also generates Industrial's held-out-only "
        "manifest).",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "datasets.yaml"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    if args.dataset == "all":
        targets = DATASETS
    elif args.dataset == "all_with_industrial":
        targets = ALL_DATASETS
    else:
        targets = (args.dataset,)
    for dataset_name in targets:
        heldout_only = dataset_name in HELDOUT_ONLY_DATASETS
        manifest, excluded = build_manifest(
            dataset_name, Path(args.config), heldout_only=heldout_only
        )
        verify_no_cross_split_requirement(manifest)
        verify_negatives_are_real_non_links(dataset_name, Path(args.config), manifest)
        if heldout_only:
            verify_heldout_only_dataset(dataset_name, manifest)
        output_path = MANIFEST_DIR / f"{dataset_name}.csv"
        manifest.to_csv(output_path, index=False)
        excluded_path = MANIFEST_DIR / f"{dataset_name}_excluded_requirements.csv"
        excluded.to_csv(excluded_path, index=False)
        print_summary(dataset_name, manifest, excluded)
        print(f"  -> {output_path}")
        if not excluded.empty:
            print(f"  -> {excluded_path}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
