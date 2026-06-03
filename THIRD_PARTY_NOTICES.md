# Third-Party Notices

This project depends on third-party open-source software and can optionally use
third-party machine-learning model weights. This file is informational and does
not replace the license texts or terms provided by the original authors.

## Project license scope

The `LICENSE` file applies to this repository's original source code,
documentation, and configuration files only.

It does **not** grant rights to redistribute third-party dependencies, model
weights, pretrained checkpoints, datasets, trademarks, logos, or captured media.

## Python dependencies

Runtime dependencies are listed in `requirements.txt`. Before distributing a
built image, wheel cache, vendored dependency, or other binary artifact, review
the licenses and notice requirements for the direct and transitive dependencies,
including but not limited to:

| Dependency | Version used locally | License metadata observed locally |
| --- | ---: | --- |
| Streamlit | 1.50.0 | Apache License 2.0 |
| Ultralytics | 8.4.60 | AGPL-3.0 |
| OpenCV / `opencv-python-headless` | 4.13.0.92 | Apache 2.0 |
| NumPy | 2.0.2 | BSD-style license plus bundled notices |
| EasyOCR | 1.7.2 | Apache License 2.0 |
| InsightFace | 1.0.1 | Review upstream package/model terms |
| ONNX Runtime | 1.19.2 | MIT License |
| DeepFace | 0.0.100 | MIT |
| TensorFlow | 2.20.0 | Apache 2.0 |
| TF-Keras | 2.20.1 | Apache 2.0 |
| PyTorch | 2.8.0 | BSD-3-Clause |
| TorchVision | 0.23.0 | BSD |
| Pandas | 2.3.3 | BSD 3-Clause License |
| Pillow | 11.3.0 | Review upstream package terms |

You can generate a dependency license report with tools such as `pip-licenses`
or your organization's software-composition-analysis tooling.

**Important:** Ultralytics is reported by local package metadata as AGPL-3.0.
If you distribute this application, host it as a network service, or distribute
a container containing Ultralytics, review AGPL obligations and whether a
commercial Ultralytics license is required for your intended use.

## Model weights and checkpoints

This public repository should not include pretrained model files by default.
Users are responsible for obtaining any model weights they use and for complying
with the original license terms.

Known local model paths used by the application/container workflow include:

- `models/*.pt` for YOLO detector checkpoints
- `insightface_models/models/buffalo_l/*.onnx` for InsightFace Buffalo_L assets
- `.deepface_home/.deepface/weights/*.h5` for DeepFace fallback models such as
  VGG-Face

These model files may have license terms that differ from the loader libraries.
Some model licenses may restrict redistribution, commercial use, biometric/face
recognition use, or require attribution. Do not publish a container image, tar
archive, model directory, or release asset containing these files unless you
have verified that redistribution is permitted.

## Biometric/privacy notice

This project performs face and vehicle analysis. Deployments involving face
recognition, biometric identifiers, surveillance, or personally identifiable
information may be regulated by local law. You are responsible for obtaining
required consent, providing notices, limiting retention, securing data, and
complying with applicable privacy and biometric laws.

## Container images and release artifacts

For safer public GitHub publishing, this repository ignores generated container
archives, runtime databases, datasets, captured media, local caches, and model
weights. Publishing source code plus a container recipe is generally safer than
publishing prebuilt images that contain third-party weights.

If you distribute a prebuilt image, include all required third-party license
texts, copyright notices, source offers if applicable, attribution statements,
and model license notices.
