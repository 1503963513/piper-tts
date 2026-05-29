# Piper TTS Service

纯 CPU、零 GPU、高并发的 TTS 语音合成服务。基于 [Piper](https://github.com/rhasspy/piper) 引擎，支持中英混合朗读、WebSocket 流式输出、LRU 缓存，Docker 一键部署。

> 适用场景：政务云内网、无 GPU 环境、不想付费、需要高并发的 TTS 需求

## 特性

- **纯 CPU 运行** — 基于 ONNX Runtime，无需 CUDA，4 核 4G 即可跑
- **中英混合** — 自动识别中英文，分模型合成后拼接，音色过渡自然
- **流式输出** — WebSocket 逐片段推送，首字延迟 < 500ms
- **智能缓存** — LRU 磁盘缓存，相同文本秒返回（< 10ms）
- **动态并发** — 根据 CPU 核心数自动调整，防 OOM 防雪崩
- **Docker 部署** — 一行命令启动，模型内置于镜像

## 模型

| 模型 | 语言 | 说明 |
|------|------|------|
| `zh_CN-huayan-x_low` | 中文 | 华研女声，清晰自然 |
| `en_US-amy-low` | 英文 | Amy 女声，音色柔和 |

两个模型音色接近，中英混合朗读时过渡自然。

## 快速开始

### Docker（推荐）

```bash
docker run -d \
  --name piper-tts \
  -p 19527:5001 \
  --shm-size=1g \
  your-dockerhub-user/piper-tts:latest
```

### docker-compose

```bash
docker-compose up -d
```

### 手动运行

```bash
pip install -r requirements.txt
python app.py
```

## API

### 健康检查

```
GET /health
```

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models": ["zh", "en"],
  "cache_count": 12
}
```

### 语音合成

```
POST /synthesize
```

```json
{
  "text": "欢迎使用piper语音合成",
  "length_scale": 1.0,
  "noise_scale": 0.667
}
```

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `text` | string | ✅ | - | 合成文本，支持中英混合 |
| `length_scale` | float | ❌ | 1.0 | 语速，越小越快 |
| `noise_scale` | float | ❌ | 0.667 | 随机性 |
| `speaker_id` | int | ❌ | 0 | 说话人 ID（多人模型） |

返回：`audio/wav` 音频文件

### 流式合成（WebSocket）

```
WS /ws/synthesize
```

1. 建立 WebSocket 连接
2. 发送 `{"text": "要朗读的内容"}`
3. 流式接收 PCM 音频片段
4. 收到 `"END"` 表示合成完毕

### 清除缓存

```
POST /cache/clear
```

## 调用示例

```bash
# 中文
curl -X POST http://localhost:19527/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "政务云内网纯CPU语音合成"}' \
  --output zh.wav

# 中英混合
curl -X POST http://localhost:19527/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "基于piper引擎，支持中英混合"}' \
  --output mix.wav

# 加速
curl -X POST http://localhost:19527/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "语速变快了", "length_scale": 0.8}' \
  --output fast.wav
```

## 性能参考

> 4 核 8G 政务云服务器测试

| 指标 | 数值 |
|------|------|
| 单句合成（20 字内） | < 1s |
| 首字延迟（WebSocket） | < 500ms |
| 缓存命中 | < 10ms |
| 并发 20 请求 | 稳定响应 |
| 空闲内存占用 | ~300MB |

## 项目结构

```
piper-tts/
├── app.py              # 主服务
├── dockerfile          # Docker 构建
├── docker-compose.yml  # 编排配置
├── requirements.txt    # Python 依赖
├── models/             # ONNX 模型
│   ├── zh_CN-huayan-x_low.onnx
│   └── en_US-amy-low.onnx
└── build.sh            # 构建脚本
```

## 环境

- Python 3.10
- Flask + flask-sock + flask-cors
- ONNX Runtime（CPU）
- Piper TTS
- espeak-ng（音素转换后端）

## License

MIT
