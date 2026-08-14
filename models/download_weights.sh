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

echo "=== [2/2] Verifying FaceMesh ONNX Model ==="
if [ ! -f "$MODEL_DIR/mediapipe_pose/20_new_onnx_postprocess_N-batch/face_mesh_192x192_post.onnx" ]; then
    echo "FaceMesh ONNX model not found."
    echo "Please place the face_mesh_192x192_post.onnx file under $MODEL_DIR/mediapipe_pose/..."
else
    echo "Found FaceMesh ONNX model."
fi

echo "=== Model check completed! ==="
