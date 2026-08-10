#!/bin/bash
# 生成 Python gRPC 桩代码,供 cloud_client.py 使用
#
# 用法:
#   cd tools && bash generate_proto.sh
#
# 依赖:
#   pip install grpcio-tools

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROTO_DIR="$SCRIPT_DIR/../protos"
PROTO_FILE="$PROTO_DIR/robot_brain.proto"

if [ ! -f "$PROTO_FILE" ]; then
    echo "Error: $PROTO_FILE not found"
    exit 1
fi

echo "Generating Python gRPC stubs from $PROTO_FILE..."

python3 -m grpc_tools.protoc \
    --proto_path="$PROTO_DIR" \
    --python_out="$PROTO_DIR" \
    --grpc_python_out="$PROTO_DIR" \
    "$PROTO_FILE"

echo "Done: generated robot_brain_pb2.py and robot_brain_pb2_grpc.py in $PROTO_DIR"
