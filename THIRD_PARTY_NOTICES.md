# Third-Party Notices

The [MIT License](LICENSE) in this repository covers the original code written for
this project. It does **not** cover, and was never intended to cover, third-party
components that remain the property of their respective owners. This file lists
those components explicitly.

## NVIDIA DeepStream SDK sample code (excluded from this repository)

`vision-pipeline`'s primary detector (`PeopleNet`) uses a custom output parser
built from the **unmodified sample code shipped with the NVIDIA DeepStream SDK**
(`nvdsinfer_custombboxparser.cpp`, `nvdsinfer_customclassifierparser.cpp`,
`nvdsinfer_customsegmentationparser.cpp`, and their `Makefile`/`README`), plus the
INT8 calibration cache generated alongside the PeopleNet engine
(`resnet34_peoplenet_int8.txt`).

These files carry NVIDIA's own header:

```
SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material... Any use, reproduction,
disclosure or distribution of this material... without an express license
agreement from NVIDIA CORPORATION or its affiliates is strictly prohibited.
```

Because of that notice, these files are **intentionally not tracked in this repo**
(see `.gitignore`) and are **not** covered by the project's MIT license. They are
not this project's intellectual property to redistribute.

### How to obtain them locally

If you are building `vision-pipeline` yourself on a machine with the DeepStream
SDK installed, the sample parser sources are already present on disk under:

```
/opt/nvidia/deepstream/deepstream/sources/objectDetector_Yolo/nvdsinfer_custom_impl_Yolo/
```

(or the equivalent `sources/` sample directory for your installed DeepStream
version — consult the DeepStream SDK documentation shipped with your JetPack/
DeepStream install). Copy the relevant files into
`models/peoplenet/custom_parser/`, adjust `parse-bbox-func-name` in
[`models/peoplenet/config_infer_peoplenet.txt`](models/peoplenet/config_infer_peoplenet.txt)
to match, and build with `make` inside that directory per NVIDIA's own
instructions. The INT8 calibration file (`resnet34_peoplenet_int8.txt`) is
regenerated automatically by TensorRT the first time the PeopleNet ONNX model is
built into an engine with `network-mode=2` (INT8) — you do not need to source it
separately.

## NVIDIA PeopleNet model weights

`resnet34_peoplenet_int8.onnx` (and the `.engine` file built from it) are **not**
included in this repository either. PeopleNet is a pretrained model distributed
by NVIDIA via NGC under NVIDIA's own model license, not the MIT license of this
project. Download it yourself from NGC and place it under `models/peoplenet/` —
see [`models/download_weights.sh`](models/download_weights.sh) and the main
[README](README.md#configuration) for details.

## Academic algorithm reference

The head-pose estimation core in
[`services/analytics_api/modules/head_pose_algo.py`](services/analytics_api/modules/head_pose_algo.py)
is an original implementation of the geometric method described in:

> Hui Yuan, Mengyu Li, Junhui Hou, Jimin Xiao, *"Single Image based Head Pose
> Estimation with Spherical Parameterization and 3D Morphing"*,
> [arXiv:1907.09217](https://arxiv.org/abs/1907.09217).

No source code from the paper's authors was used or is publicly available; this
is an independent re-implementation from the published method, credited here and
in the code comments.
