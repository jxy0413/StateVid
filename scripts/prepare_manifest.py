#!/usr/bin/env python3
"""Create portable manifests from paired folders or LEVIR-style caption JSON."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
DEFAULT_DIRS = {
    "bcd": {"pre":"A", "post":"B", "label":"label"},
    "scd": {"pre":"im1", "post":"im2", "label_pre":"label1", "label_post":"label2"},
    "bda": {"pre":"pre", "post":"post", "localization":"localization", "damage":"damage"},
    "cc": {"pre":"A", "post":"B"},
}


def relative(path, output):
    return Path(os.path.relpath(path.resolve(),output.parent.resolve())).as_posix()


def image_index(folder):
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    found = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in SUFFIXES:
            if path.stem in found:
                raise ValueError(f"Duplicate image stems in {folder}")
            found[path.stem] = path
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task",choices=DEFAULT_DIRS,required=True)
    p.add_argument("--dataset",required=True)
    p.add_argument("--root",type=Path,required=True,help="Folder containing train/val/test subfolders")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--num-classes",type=int)
    p.add_argument("--metadata",type=Path,help="Optional JSON containing class_names/palette_rgb/normalization settings")
    p.add_argument("--splits",nargs="+",default=["train","val","test"])
    p.add_argument("--captions-json",type=Path,help="LEVIR-style {'images': [...]} annotation JSON for captioning")
    p.add_argument("--image-prefix",default="",help="Optional subfolder under root before split/A and split/B")
    for key in ("pre","post","label","label-pre","label-post","localization","damage"):
        p.add_argument("--"+key+"-dir",help="Override the paired subfolder name")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    data = {"schema_version":1,"dataset":args.dataset,"task":args.task,"splits":{x:[] for x in args.splits}}
    if args.metadata:
        extra = json.loads(args.metadata.read_text(encoding="utf-8"))
        if set(extra)&{"splits","task","dataset","root"}:
            raise ValueError("Metadata may define label/normalization settings, not paths or splits")
        data.update(extra)
    if args.num_classes:
        data["num_classes"] = args.num_classes
    if args.task == "scd" and not data.get("num_classes"):
        p.error("SCD requires --num-classes or num_classes in --metadata")
    dirs = {key:getattr(args,key+"_dir",None) or value for key,value in DEFAULT_DIRS[args.task].items()}
    base = args.root/args.image_prefix
    if args.task == "cc":
        if not args.captions_json:
            p.error("Captioning requires --captions-json")
        annotations = json.loads(args.captions_json.read_text(encoding="utf-8"))
        images = annotations.get("images") if isinstance(annotations,dict) else annotations
        if not isinstance(images,list):
            raise ValueError("Caption annotations need an images list")
        for entry in images:
            split = entry["split"]
            if split not in data["splits"]:
                continue
            filename = entry.get("filename",entry.get("id"))
            refs = entry.get("captions",entry.get("sentences"))
            if not isinstance(filename,str) or not isinstance(refs,list) or not refs:
                raise ValueError("Each caption record needs filename, split, and captions/sentences")
            captions = [x.get("tokens",x.get("raw",x.get("text"))) if isinstance(x,dict) else x for x in refs]
            record = {"id":filename,"captions":captions}
            for key,folder in dirs.items():
                source = base/split/folder/filename
                if not source.is_file():
                    raise FileNotFoundError(source)
                record[key] = relative(source,args.output)
            if "changeflag" in entry:
                record["changeflag"] = entry["changeflag"]
            data["splits"][split].append(record)
    else:
        for split in args.splits:
            indices = {key:image_index(base/split/folder) for key,folder in dirs.items()}
            stems = set(indices["pre"])
            if not stems:
                raise ValueError(f"No images found in {split}")
            for key,index in indices.items():
                if set(index) != stems:
                    raise ValueError(f"{split}/{dirs[key]} has unmatched stems")
            data["splits"][split] = [{"id":stem,**{key:relative(index[stem],args.output) for key,index in indices.items()}}
                                     for stem in sorted(stems)]
    if any(not rows for rows in data["splits"].values()):
        raise ValueError("One of the requested splits is empty")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"manifest":str(args.output),"samples":{key:len(rows) for key,rows in data["splits"].items()}}))


if __name__ == "__main__":
    main()
