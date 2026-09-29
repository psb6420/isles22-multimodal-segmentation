# Third-party notices and data policy

The Apache-2.0 license in this repository applies only to the original source code and documentation committed here. It does not relicense datasets, pretrained weights, third-party software, papers, trademarks, or externally linked material.

## ISLES'22 data

No ISLES'22 MRI volume, annotation, subject-level report, subject identifier, registered derivative, prediction volume, or model checkpoint is distributed in this repository.

ISLES'22 is published under CC BY 4.0 together with an additional Data Usage Policy. The copy supplied with the dataset states that redistribution of data or accompanying materials requires written agreement from the ISLES'22 team. Users must obtain the dataset through its authorized distribution channel and comply with its current terms.

Required dataset citation stated by the distributed terms:

> Petzsche, M. R. H., de la Rosa, E., Hanning, U., Wiest, R., Pinilla, W. E. V., Reyes, M., et al. (2022). ISLES 2022: A multi-center magnetic resonance imaging stroke lesion segmentation dataset. arXiv:2206.06694.

- Official repository: <https://github.com/ezequieldlrosa/isles22>
- Creative Commons Attribution 4.0: <https://creativecommons.org/licenses/by/4.0/>

The aggregate metrics committed in `reports/model_comparison.csv` contain no images or subject identifiers.

## Software dependencies

Third-party packages are installed separately and remain governed by their own licenses. Their source code is not vendored in this repository.

| Project | Use | License/source |
|---|---|---|
| nnU-Net | 3D segmentation training and inference | [Apache-2.0](https://github.com/MIC-DKFZ/nnUNet/blob/master/LICENSE) |
| PyTorch | Deep-learning runtime | [Upstream license](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| NumPy | Array processing | [Upstream license](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| SciPy | Connected components and scientific routines | [BSD-3-Clause](https://github.com/scipy/scipy/blob/main/LICENSE.txt) |
| scikit-learn | Patient-level stratified split | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) |
| NiBabel | NIfTI input/output and geometry | [Upstream license](https://github.com/nipy/nibabel/blob/master/COPYING) |
| SimpleITK | FLAIR-to-DWI registration | [Apache-2.0](https://github.com/SimpleITK/SimpleITK/blob/main/LICENSE) |
| openpyxl | Reading the dataset's center spreadsheet | [Upstream project](https://foss.heptapod.net/openpyxl/openpyxl) |

## Referenced research implementations

The README links to SEALS, DeepISLES, Factorizer, MoME, BrainSegFounder, and MedSAM3 for scholarly comparison. Their source code and weights are not copied into this repository. A local SEALS checkout used during research is excluded by `.gitignore`; SEALS is distributed by its authors under [Apache-2.0](https://github.com/Tabrisrei/ISLES22_SEALS/blob/master/LICENCE).

Anyone adding third-party source code or weights later must preserve the upstream copyright and license notices and verify redistribution rights before committing them.
