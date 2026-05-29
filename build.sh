#!/bin/bash

# Piper TTS Docker 构建脚本
# 支持 linux/amd64 平台

set -e

IMAGE_NAME="piper-tts"
TAG="latest"
PLATFORM="linux/amd64"

echo "🚀 开始构建 Piper TTS Docker 镜像..."
echo "平台: $PLATFORM"
echo "镜像名: $IMAGE_NAME:$TAG"

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ Docker 未安装，请先安装 Docker"
    exit 1
fi

# 检查 models 目录是否存在
if [ ! -d "./models" ]; then
    echo "⚠️  警告: models 目录不存在，构建的镜像将不包含模型文件"
    echo "   请确保 models 目录中包含所需的 .onnx 和 .json 文件"
fi

# 构建镜像
echo "📦 正在构建 Docker 镜像..."
docker buildx build \
    --platform $PLATFORM \
    --load \
    --tag $IMAGE_NAME:$TAG \
    .

if [ $? -eq 0 ]; then
    echo "✅ 镜像构建成功！"
    echo "📋 镜像信息:"
    docker images | grep $IMAGE_NAME
    
    echo ""
    echo "🎯 使用方法:"
    echo "1. 直接运行: docker run -p 19527:5001 --shm-size=512mb $IMAGE_NAME:$TAG"
    echo "2. 使用 docker-compose: docker-compose up -d"
    echo ""
    echo "🔍 健康检查: curl http://localhost:19527/health"
else
    echo "❌ 镜像构建失败！"
    exit 1
fi