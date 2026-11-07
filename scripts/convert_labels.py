#!/usr/bin/env python3
"""Convert directional change annotations, binary masks, and semantic palettes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}
# Semantic indices: 0 unchanged, 1 cropland, 2 road, 3 forest/grassland,
# 4 building, and 5 other. Transition indices are 1..8 in this order.
JL1H_TRANSITIONS = ((1,2),(1,3),(1,4),(1,5),(2,1),(3,1),(4,1),(5,1))


def _indexed(array):
    value = np.asarray(array)
    if value.ndim == 3 and value.shape[-1] in (3,4):
        if not np.all(value[:,:,:3] == value[:,:,:1]):
            raise ValueError("Colored masks need the remap --palette command first")
        value = value[:,:,0]
    if value.ndim != 2 or not (np.issubdtype(value.dtype,np.integer) or value.dtype == np.bool_):
        raise ValueError("An indexed mask must be a two-dimensional integer image")
    if value.size and (value.min() < 0 or value.max() > 65535):
        raise ValueError("Indexed input values must lie in 0..65535")
    return value


def _label_dtype(maximum):
    if not 0 <= maximum <= 65535:
        raise ValueError("PNG output label values must lie in 0..65535")
    return np.uint8 if maximum <= 255 else np.uint16


def split_jl1h_transition(array, ignore_index=255):
    """Decompose labels 1..8 into the original pre/post semantic indices."""
    value = _indexed(array)
    if ignore_index is not None and (type(ignore_index) is not int or 0 <= ignore_index <= 8):
        raise ValueError("ignore_index must be outside JL1H's valid 0..8 labels")
    if ignore_index is not None:
        _label_dtype(ignore_index)
    valid = np.ones(value.shape,bool) if ignore_index is None else value != ignore_index
    if not np.isin(value[valid],range(9)).all():
        raise ValueError("JL1H transition masks must contain 0..8 and the configured ignored value")
    dtype = _label_dtype(max(5,ignore_index or 0))
    pre,post = np.zeros(value.shape,dtype=dtype),np.zeros(value.shape,dtype=dtype)
    for category,(source,target) in enumerate(JL1H_TRANSITIONS,1):
        selected = value == category
        pre[selected],post[selected] = source,target
    if ignore_index is not None:
        pre[~valid],post[~valid] = ignore_index,ignore_index
    return pre,post


def binary_union(first, second):
    """Combine two 0/1 or 0/255 masks and return a 0/255 change mask."""
    a,b = _indexed(first),_indexed(second)
    if a.shape != b.shape:
        raise ValueError("The two binary masks must have matching spatial dimensions")
    if not np.isin(a,[0,1,255]).all() or not np.isin(b,[0,1,255]).all():
        raise ValueError("Binary input masks may contain only 0, 1, or 255")
    return ((a > 0) | (b > 0)).astype(np.uint8)*255


def _mapping_dict(mapping):
    if isinstance(mapping,list):
        result = dict(enumerate(mapping))
    elif isinstance(mapping,dict):
        result = {}
        for source,target in mapping.items():
            if isinstance(source,str):
                if not source.isdecimal():
                    raise ValueError("Index-mapping keys must be nonnegative integers")
                source = int(source)
            if type(source) is not int:
                raise ValueError("Index-mapping keys must be integer indices")
            result[source] = target
    else:
        raise ValueError("An index mapping must be a list or an old-index:new-index object")
    if not result or any(type(k) is not int or type(v) is not int or not 0 <= k <= 65535 or not 0 <= v <= 65535
                         for k,v in result.items()):
        raise ValueError("Index mappings must contain integer indices in 0..65535")
    return result


def remap_indices(array, mapping, *, ignore_index=255, keep_unmapped=False):
    """Apply an explicit index mapping while preserving the ignored value."""
    value = _indexed(array)
    mapping = _mapping_dict(mapping)
    if ignore_index is not None:
        _label_dtype(ignore_index)
        if ignore_index in mapping and mapping[ignore_index] != ignore_index:
            raise ValueError("The ignored value is preserved; use --ignore-index none to explicitly remap it")
    known = set(mapping)
    if ignore_index is not None:
        known.add(ignore_index)
    unknown = set(np.unique(value).tolist())-known
    if unknown and not keep_unmapped:
        raise ValueError(f"Input labels missing from the mapping: {sorted(unknown)}")
    maximum = max(max(mapping.values()),int(value.max()) if value.size and keep_unmapped else 0,ignore_index or 0)
    result = value.astype(_label_dtype(maximum),copy=True) if keep_unmapped else np.zeros(value.shape,dtype=_label_dtype(maximum))
    for source,target in mapping.items():
        result[value == source] = target
    if ignore_index is not None:
        result[value == ignore_index] = ignore_index
    return result


def decode_rgb_palette(array, palette, *, ignore_index=255, ignore_color=None):
    """Decode exact RGB colors to indices without nearest-color matching."""
    value = np.asarray(array)
    if value.ndim != 3 or value.shape[-1] not in (3,4):
        raise ValueError("Palette conversion requires an RGB or RGBA image")
    if isinstance(palette,dict) and "palette_rgb" in palette:
        palette = palette["palette_rgb"]
    if isinstance(palette,list):
        colors = dict(enumerate(palette))
    elif isinstance(palette,dict):
        colors = {int(index):color for index,color in palette.items()}
    else:
        raise ValueError("Palette must be an RGB list or index-to-RGB mapping")
    if not colors or any(not isinstance(color,(list,tuple)) or len(color) != 3 or
                         any(type(x) is not int or not 0 <= x <= 255 for x in color)
                         for color in colors.values()):
        raise ValueError("Every palette entry must contain three integer RGB values in 0..255")
    if len({tuple(x) for x in colors.values()}) != len(colors):
        raise ValueError("Palette colors must map to unique indices")
    if ignore_color is not None:
        if ignore_index is None:
            raise ValueError("An ignore color requires an ignored output index")
        if len(ignore_color) != 3 or any(type(x) is not int or not 0 <= x <= 255 for x in ignore_color):
            raise ValueError("ignore_color must contain three integer RGB values")
        if tuple(ignore_color) in {tuple(x) for x in colors.values()}:
            raise ValueError("The ignore color duplicates a semantic palette color")
    dtype = _label_dtype(max(max(colors),ignore_index or 0))
    result = np.zeros(value.shape[:2],dtype=dtype)
    matched = np.zeros(result.shape,bool)
    for index,color in colors.items():
        if index < 0:
            raise ValueError("Palette class indices must be nonnegative")
        selected = np.all(value[:,:,:3] == np.asarray(color),axis=-1)
        result[selected] = index
        matched |= selected
    ignored = np.zeros(result.shape,bool)
    if ignore_color is not None:
        ignored |= np.all(value[:,:,:3] == np.asarray(ignore_color),axis=-1)
    if value.shape[-1] == 4:
        ignored |= value[:,:,3] == 0
    if ignored.any():
        if ignore_index is None:
            raise ValueError("Transparent pixels require an ignore_index")
        result[ignored] = ignore_index
        matched |= ignored
    if not matched.all():
        unknown = np.unique(value[:,:,:3][~matched],axis=0)[:8].tolist()
        raise ValueError(f"Colors missing from the palette: {unknown}")
    return result


def _scan(path, pattern="*", recursive=False):
    if path.is_file():
        return {Path(path.stem):path}
    if not path.is_dir():
        raise FileNotFoundError(path)
    result = {}
    paths = path.rglob(pattern) if recursive else path.glob(pattern)
    for source in sorted(paths):
        if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
            key = source.relative_to(path).with_suffix("")
            if key in result:
                raise ValueError(f"Two input images share the relative stem {key}")
            result[key] = source
    if not result:
        raise ValueError(f"No matching label images in {path}")
    return result


def _read(path, rgb=False):
    with Image.open(path) as image:
        if rgb:
            image = image.convert("RGBA" if image.mode == "RGBA" or "transparency" in image.info else "RGB")
        return np.asarray(image).copy()


def _write(path, array, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}; use --overwrite to replace it")
    path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(array).save(path)


def _png_key(key):
    return key.parent/(key.name+".png")


def _ignore(value):
    if value.lower() == "none":
        return None
    number = int(value)
    _label_dtype(number)
    return number


def _color(value):
    color = tuple(int(x) for x in value.split(","))
    if len(color) != 3 or any(not 0 <= x <= 255 for x in color):
        raise argparse.ArgumentTypeError("Color must be three comma-separated values in 0..255")
    return color


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command",required=True)
    jl = commands.add_parser("jl1h",help="Split JL1H directional transition masks into two endpoint maps")
    jl.add_argument("--input",type=Path,required=True)
    jl.add_argument("--output-pre",type=Path,required=True)
    jl.add_argument("--output-post",type=Path,required=True)
    jl.add_argument("--ignore-index",type=_ignore,default=255)
    union = commands.add_parser("binary-union",help="Union S2Looking addition/removal masks")
    union.add_argument("--first",type=Path,required=True)
    union.add_argument("--second",type=Path,required=True)
    union.add_argument("--output",type=Path,required=True)
    remap = commands.add_parser("remap",help="Remap indexed labels or decode an RGB semantic palette")
    remap.add_argument("--input",type=Path,required=True)
    remap.add_argument("--output",type=Path,required=True)
    mode = remap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mapping",type=Path,help="JSON list or old-index:new-index mapping")
    mode.add_argument("--palette",type=Path,help="JSON RGB palette or metadata containing palette_rgb")
    remap.add_argument("--ignore-index",type=_ignore,default=255)
    remap.add_argument("--ignore-color",type=_color,help="Optional ignored RGB color, e.g. 255,0,255")
    remap.add_argument("--keep-unmapped",action="store_true",help="Keep unlisted indices in index-mapping mode")
    for command in (jl,union,remap):
        command.add_argument("--pattern",default="*",help="Input filename glob; supported image extensions only")
        command.add_argument("--recursive",action="store_true",help="Preserve relative subfolders")
        command.add_argument("--overwrite",action="store_true")
    args = parser.parse_args()
    if args.command == "binary-union":
        first = _scan(args.first,args.pattern,args.recursive)
        second = _scan(args.second,args.pattern,args.recursive)
        if args.first.is_file() and args.second.is_file():
            pairs = [(next(iter(first)),args.first,args.second)]
        else:
            if set(first) != set(second):
                raise ValueError("Binary mask folders must contain the same relative image stems")
            pairs = [(key,first[key],second[key]) for key in sorted(first)]
        for key,a,b in pairs:
            _write(args.output/_png_key(key),binary_union(_read(a),_read(b)),args.overwrite)
        count = len(pairs)
    else:
        inputs = _scan(args.input,args.pattern,args.recursive)
        settings = None
        if args.command == "remap":
            settings = json.loads((args.mapping or args.palette).read_text(encoding="utf-8"))
            if args.palette and args.keep_unmapped:
                parser.error("--keep-unmapped applies to indexed mappings, not RGB palettes")
            if args.mapping and args.ignore_color:
                parser.error("--ignore-color applies only to RGB palettes")
        for key,source in inputs.items():
            if args.command == "jl1h":
                pre,post = split_jl1h_transition(_read(source),args.ignore_index)
                _write(args.output_pre/_png_key(key),pre,args.overwrite)
                _write(args.output_post/_png_key(key),post,args.overwrite)
            else:
                converted = (remap_indices(_read(source),settings,ignore_index=args.ignore_index,keep_unmapped=args.keep_unmapped)
                             if args.mapping else decode_rgb_palette(_read(source,rgb=True),settings,ignore_index=args.ignore_index,ignore_color=args.ignore_color))
                _write(args.output/_png_key(key),converted,args.overwrite)
        count = len(inputs)
    print(json.dumps({"operation":args.command,"images":count}))


if __name__ == "__main__":
    main()
