#!/usr/bin/env python3
"""Generate small synthetic image pairs for testing the training/evaluation CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task",choices=["bcd","scd","bda","cc"],required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--size",type=int,default=64)
    parser.add_argument("--train-samples",type=int,default=8)
    args = parser.parse_args()
    if args.size < 16 or args.train_samples < 1:
        parser.error("size must be at least 16 and train-samples must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose an empty output directory")
    args.output.mkdir(parents=True,exist_ok=True)
    rng = np.random.default_rng(42)
    data = {"schema_version":1,"dataset":"SyntheticDemo","task":args.task,"splits":{}}
    if args.task == "scd":
        data.update(num_classes=3,class_names=["Unchanged","Source","Target"],
                    palette_rgb=[[255,255,255],[70,130,180],[190,110,80]])
    if args.task == "bda":
        data.update(num_classes=5,class_names=["Background","No damage","Minor","Major","Destroyed"])
    for split,count in [("train",args.train_samples),("val",2),("test",2)]:
        records = []
        for i in range(count):
            folder = args.output/split/f"pair_{i:03d}"
            folder.mkdir(parents=True)
            h = args.size
            background = rng.integers(35,65,size=(h,h,3),dtype=np.uint8)
            pre,post = background.copy(),background.copy()
            before,after = np.zeros((h,h),bool),np.zeros((h,h),bool)
            offset = int(rng.integers(0,max(1,h//8)))
            before[h//4:h//2,h//4+offset:h//2+offset] = True
            after[h//2:3*h//4,h//2-offset:3*h//4-offset] = True
            pre[before],post[after] = [200,205,210],[200,205,210]
            changed = before|after
            values = {"pre":pre,"post":post}
            record = {"id":f"{split}_{i:03d}"}
            if args.task == "bcd":
                values["label"] = changed.astype(np.uint8)*255
            elif args.task == "scd":
                values.update(label_pre=changed.astype(np.uint8),label_post=changed.astype(np.uint8)*2)
            elif args.task == "bda":
                loc = np.zeros((h,h),np.uint8)
                loc[h//4:3*h//4,h//4:3*h//4] = 1
                damage = np.zeros_like(loc)
                for row in range(2):
                    for col in range(2):
                        damage[h//4+row*h//4:h//2+row*h//4,h//4+col*h//4:h//2+col*h//4] = 1+2*row+col
                values.update(localization=loc,damage=damage)
            else:
                record["captions"] = [["a","white","building","appears","here"],
                                       ["a","white","building","appears","nearby"]]
            for key,value in values.items():
                path = folder/(key+".png")
                Image.fromarray(value).save(path)
                record[key] = path.relative_to(args.output).as_posix()
            records.append(record)
        data["splits"][split] = records
    output = args.output/"manifest.json"
    output.write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
