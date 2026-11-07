#!/usr/bin/env python3
"""Screen external teacher probabilities and attach optional SCD pseudo masks.

The probability manifest is {"samples": {"record_id": "relative/pair.npz"}}.
Each NPZ contains `pre` and `post` float class probabilities in CxHxW order.
This utility does not load or distribute a teacher architecture or weights.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from statevid.data import ChangeDataset,load_manifest,resolve_path
from statevid.pseudo import ClasswisePseudoLabeler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--probabilities",type=Path,required=True)
    p.add_argument("--cache-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True,help="New manifest with pseudo_pre/pseudo_post paths")
    p.add_argument("--split",default="train")
    p.add_argument("--threshold-state",type=Path,help="Optional accumulator state to continue")
    args = p.parse_args()
    source = load_manifest(args.manifest,"scd")
    classes = source["num_classes"]
    if classes > 255:
        p.error("Cached PNG labels reserve 255 for ignored pixels; num_classes must be <=255")
    if args.output.exists() or (args.cache_dir.exists() and any(args.cache_dir.iterdir())):
        raise FileExistsError("Choose a new output manifest and an empty cache directory")
    probability_map = json.loads(args.probabilities.read_text(encoding="utf-8"))["samples"]
    dataset = ChangeDataset(args.manifest,args.split,augment=False,crop_size=None)
    labeler = ClasswisePseudoLabeler(classes)
    if args.threshold_state:
        labeler.load_state_dict(json.loads(args.threshold_state.read_text(encoding="utf-8")))
    result = deepcopy(source)
    result.pop("root",None)
    # The new manifest may live elsewhere; rebase all original source paths.
    keys = ("pre","post","label","label_pre","label_post","valid_mask","localization","damage","pseudo_pre","pseudo_post")
    for records in result["splits"].values():
        for record in records:
            for key in keys:
                if record.get(key):
                    path = resolve_path(args.manifest,record[key],source.get("root","."))
                    record[key] = Path(os.path.relpath(path.resolve(),args.output.parent.resolve())).as_posix()
    args.cache_dir.mkdir(parents=True,exist_ok=True)
    summary = {"samples":0,"candidate_pixels":0,"accepted_pixels":0}
    for index,record in enumerate(source["splits"][args.split]):
        item = dataset[index]
        if record["id"] not in probability_map:
            raise ValueError(f"Missing teacher probabilities for {record['id']!r}")
        path = Path(probability_map[record["id"]])
        path = path if path.is_absolute() else args.probabilities.parent/path
        with np.load(path,allow_pickle=False) as values:
            probabilities = [torch.from_numpy(np.array(values[key],dtype=np.float32))[None] for key in ("pre","post")]
        screened = labeler.generate(*probabilities,item["label_pre"][None],item["label_post"][None],valid=item["valid_mask"][None])
        accepted = screened["accepted"][0].numpy()
        for key,target in (("pseudo_pre","target_pre"),("pseudo_post","target_post")):
            mask = np.full(accepted.shape,255,np.uint8)
            mask[accepted] = screened[target][0].numpy()[accepted]
            output = args.cache_dir/f"{index:08d}_{key}.png"
            Image.fromarray(mask).save(output)
            result["splits"][args.split][index][key] = Path(os.path.relpath(output.resolve(),args.output.parent.resolve())).as_posix()
        summary["samples"] += 1
        for key in ("candidate_pixels","accepted_pixels"):
            summary[key] += screened["diagnostics"][key]
    (args.cache_dir/"thresholds.json").write_text(json.dumps(labeler.state_dict(),indent=2)+"\n")
    result["pseudo_labels"] = {"screening":"count-weighted classwise 60th-percentile confidence; agreed nonzero endpoints on valid GT zero/zero pixels",**summary}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"manifest":str(args.output),**summary}))


if __name__ == "__main__":
    main()
