#!/usr/bin/env python3
"""Optionally download the official Kinetics X3D-L initialization weights."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

URL = "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_L.pyth"
SHA256 = "19c99d8ab8f2aceab4f8a59a85f6dbc3c7f3a262a798ca95b906709ed100866b"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("weights/X3D_L.pyth"))
    args = parser.parse_args()
    destination = args.output
    if destination.exists():
        if digest(destination) != SHA256:
            raise ValueError("Existing output has a different SHA-256; choose another output path")
        print(f"Verified: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with urlopen(URL, timeout=60) as source, temporary.open("wb") as target:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            target.write(chunk)
    if digest(temporary) != SHA256:
        temporary.unlink()
        raise ValueError("Downloaded file did not match the official SHA-256")
    temporary.replace(destination)
    print(f"Verified download: {destination}")


if __name__ == "__main__":
    main()
