"""Raw-annotation conversion tests use temporary synthetic labels only."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

SCRIPT = Path(__file__).resolve().parents[1]/"scripts/convert_labels.py"
spec = importlib.util.spec_from_file_location("label_conversion",SCRIPT)
convert = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convert)


def test_all_eight_jl1h_transitions_and_ignored_pixels():
    source = np.array([[0,1,2,3,4],[5,6,7,8,255]],dtype=np.uint8)
    pre,post = convert.split_jl1h_transition(source)
    assert pre.tolist() == [[0,1,1,1,1],[2,3,4,5,255]]
    assert post.tolist() == [[0,2,3,4,5],[1,1,1,1,255]]
    assert np.array_equal((pre > 0) & (pre != 255),(post > 0) & (post != 255))
    with pytest.raises(ValueError):
        convert.split_jl1h_transition(np.array([[9]],np.uint8))


def test_binary_union_accepts_both_binary_encodings():
    first = np.array([[0,1,0],[255,0,0]],np.uint8)
    second = np.array([[0,0,255],[0,1,0]],np.uint8)
    result = convert.binary_union(first,second)
    assert result.dtype == np.uint8
    assert result.tolist() == [[0,255,255],[255,255,0]]
    with pytest.raises(ValueError):
        convert.binary_union(np.array([[2]],np.uint8),np.array([[0]],np.uint8))


def test_index_mapping_preserves_ignore_and_requires_known_labels():
    source = np.array([[0,1,2,3],[4,5,6,255]],np.uint8)
    result = convert.remap_indices(source,[0,3,2,4,1,5,6])
    assert result.tolist() == [[0,3,2,4],[1,5,6,255]]
    assert convert.remap_indices(np.array([[7,255]],np.uint8),{"0":0},keep_unmapped=True).tolist() == [[7,255]]
    with pytest.raises(ValueError):
        convert.remap_indices(np.array([[7]],np.uint8),[0,1])
    with pytest.raises(ValueError):
        convert.remap_indices(source,{"0":0,"255":0},keep_unmapped=True)


def test_rgb_palette_exact_values_and_explicit_ignore_color():
    palette = [[255,255,255],[0,0,255],[0,128,0]]
    image = np.array([[palette[0],palette[1]],[palette[2],[255,0,255]]],np.uint8)
    result = convert.decode_rgb_palette(image,{"palette_rgb":palette},ignore_color=(255,0,255))
    assert result.tolist() == [[0,1],[2,255]]
    with pytest.raises(ValueError):
        convert.decode_rgb_palette(image,palette)
    with pytest.raises(ValueError):
        convert.decode_rgb_palette(image,palette,ignore_color=tuple(palette[0]))


def test_conversion_cli_preserves_subfolders_and_outputs_png(tmp_path):
    source = tmp_path/"transitions"/"nested"
    source.mkdir(parents=True)
    values = np.arange(9,dtype=np.uint8).reshape(3,3)
    Image.fromarray(values).save(source/"labels.v1.tif")
    first,second = tmp_path/"before",tmp_path/"after"
    subprocess.run([sys.executable,str(SCRIPT),"jl1h","--input",str(source.parent),
                    "--output-pre",str(first),"--output-post",str(second),"--recursive"],check=True,capture_output=True)
    assert (first/"nested/labels.v1.png").is_file() and (second/"nested/labels.v1.png").is_file()
    a,b = convert.split_jl1h_transition(values)
    assert np.array_equal(np.asarray(Image.open(first/"nested/labels.v1.png")),a)
    assert np.array_equal(np.asarray(Image.open(second/"nested/labels.v1.png")),b)


def test_union_and_remap_cli_use_json_and_matching_relative_stems(tmp_path):
    first,second = tmp_path/"a",tmp_path/"b"
    first.mkdir();second.mkdir()
    Image.fromarray(np.array([[0,1]],np.uint8)).save(first/"mask.png")
    Image.fromarray(np.array([[255,0]],np.uint8)).save(second/"mask.tif")
    output = tmp_path/"union"
    subprocess.run([sys.executable,str(SCRIPT),"binary-union","--first",str(first),
                    "--second",str(second),"--output",str(output)],check=True,capture_output=True)
    assert np.asarray(Image.open(output/"mask.png")).tolist() == [[255,255]]
    mapping = tmp_path/"mapping.json";mapping.write_text(json.dumps([0,3]))
    remapped = tmp_path/"remapped"
    subprocess.run([sys.executable,str(SCRIPT),"remap","--input",str(first),
                    "--output",str(remapped),"--mapping",str(mapping)],check=True,capture_output=True)
    assert np.asarray(Image.open(remapped/"mask.png")).tolist() == [[0,3]]


def test_palette_cli_decodes_palette_png_colors_before_remapping(tmp_path):
    source = tmp_path/"colored";source.mkdir()
    image = Image.fromarray(np.array([[0,1]],np.uint8)).convert("P")
    image.putpalette([255,0,0,0,0,255]+[0]*762)
    image.save(source/"mask.png")
    palette = tmp_path/"palette.json"
    palette.write_text(json.dumps([[0,0,255],[255,0,0]]))
    output = tmp_path/"decoded"
    subprocess.run([sys.executable,str(SCRIPT),"remap","--input",str(source),
                    "--output",str(output),"--palette",str(palette)],check=True,capture_output=True)
    assert np.asarray(Image.open(output/"mask.png")).tolist() == [[1,0]]
