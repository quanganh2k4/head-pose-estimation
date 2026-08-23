#!/usr/bin/env bash
# ==============================================================================
# Model Weights Download & Setup Script
# Fetches PeopleNet and FaceMesh models for NVIDIA DeepStream / TensorRT
# ==============================================================================
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [1/2] Verifying PeopleNet Detector ==="
if [ ! -f "$MODEL_DIR/peoplenet/resnet34_peoplenet_int8.etlt" ]; then
    echo "PeopleNet model not found locally."
    echo "Please download NVIDIA PeopleNet model from NGC or ensure the files are in $MODEL_DIR/peoplenet/"
else
    echo "Found PeopleNet files."
fi

echo "=== [2/3] Verifying FaceMesh ONNX Model ==="
if [ ! -f "$MODEL_DIR/mediapipe_pose/20_new_onnx_postprocess_N-batch/face_mesh_192x192_post.onnx" ]; then
    echo "FaceMesh ONNX model not found."
    echo "Please place the face_mesh_192x192_post.onnx file under $MODEL_DIR/mediapipe_pose/..."
else
    echo "Found FaceMesh ONNX model."
fi

echo "=== [3/3] Verifying PeopleNet custom output parser ==="
if [ ! -f "$MODEL_DIR/peoplenet/custom_parser/nvdsinfer_custombboxparser.cpp" ]; then
    echo "PeopleNet custom_parser sources not found (this is expected on a fresh clone)."
    echo "These are NVIDIA DeepStream SDK proprietary sample files and are NOT"
    echo "distributed with this repo — see ../THIRD_PARTY_NOTICES.md for where to"
    echo "copy them from your local DeepStream SDK install and how to build them."
else
    echo "Found custom_parser sources."
fi

echo "=== Model check completed! ==="
