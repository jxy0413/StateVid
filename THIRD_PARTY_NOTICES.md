# Third-party notices

The project code and documentation are distributed under **CC BY-NC-SA 4.0**, except where a component carries its own license. The full project license is in [LICENSE](LICENSE). Upstream components retain their copyright notices and license terms.

## Change3D

The vendored `x3d.py` and `change_decoder.py` implementations are from [Change3D](https://github.com/zhuduowang/Change3D), by Duowang Zhu and collaborators. The upstream repository specifies [CC BY-NC-SA 4.0](https://github.com/zhuduowang/Change3D/blob/master/README.md#-license).

The source snapshot is commit `2f51deae1b23f444b4ed5e5262d8884769c0d46f`. The original `Copyright (c) Duowang Zhu` headers are retained. The project-specific perception grids, state updates, modulation, task readouts, data adapters, and training interfaces are implemented in the surrounding project modules.

| Upstream file | SHA-256 |
| --- | --- |
| `model/x3d.py` | `12837930849d7d4749faa7d4007f44645f0c7a03f13f4c74fa9e1cd9b43b64b2` |
| `model/change_decoder.py` | `4d541d579ef274c9b13d5cd40fdc69a21e7217550dc9046d3c618a62587bee00` |

License copy: [CC-BY-NC-SA-4.0.txt](third_party/licenses/CC-BY-NC-SA-4.0.txt).

```bibtex
@InProceedings{Zhu_2025_CVPR,
  author = {Zhu, Duowang and Huang, Xiaohu and Huang, Haiyan and Zhou, Hao and Shao, Zhenfeng},
  title = {Change3D: Revisiting Change Detection and Captioning from A Video Modeling Perspective},
  booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference},
  year = {2025},
  pages = {24011--24022}
}
```

## PyTorchVideo and fvcore

The X3D implementation credits [PyTorchVideo](https://github.com/facebookresearch/pytorchvideo), copyright Facebook, Inc. and its affiliates. PyTorchVideo and [fvcore](https://github.com/facebookresearch/fvcore) are external runtime dependencies released under **Apache License 2.0**. Apache-licensed upstream material retains that license.

License copies:

- [PyTorchVideo-APACHE-2.0.txt](third_party/licenses/PyTorchVideo-APACHE-2.0.txt)
- [fvcore-APACHE-2.0.txt](third_party/licenses/fvcore-APACHE-2.0.txt)

The optional X3D-L initialization file is downloaded separately from the [official PyTorchVideo model zoo](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_L.pyth). It is not included in this source distribution.

## Optional caption evaluation

Standard caption evaluation can use the external [pycocoevalcap](https://github.com/salaniz/pycocoevalcap) package, derived from [COCO-caption](https://github.com/tylin/coco-caption). Its root license retains the copyright of Xinlei Chen, Hao Fang, Tsung-Yi Lin, and Ramakrishna Vedantam and the two redistribution conditions.

A copy is included for reference: [pycocoevalcap-BSD-2-Clause.txt](third_party/licenses/pycocoevalcap-BSD-2-Clause.txt). The evaluator source, METEOR Java archive, and paraphrase data are obtained through that optional dependency and are not bundled in this project.

## Datasets and trained weights

Datasets and trained checkpoints are separate from this source release. Use each dataset's official access route and license. The project provides data schemas and configuration files for locally obtained datasets.
