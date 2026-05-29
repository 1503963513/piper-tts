#!/usr/bin/env python3
"""
Piper TTS Service - Optimized
保留原功能：流式输出、缓存LRU、中英混合、多线程并发
优化点：OOP封装、类型提示、资源管理、代码结构
"""

import os
import logging
import threading
import hashlib
import shutil
import tempfile
import re
import wave
import json
import time
import concurrent.futures
from collections import OrderedDict
from pathlib import Path
from typing import Optional, List, Dict, Generator, Tuple, Union

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_sock import Sock
import cn2an
import onnxruntime as ort

try:
    from piper import PiperVoice
except ImportError:
    print("Error: pip install piper-tts onnxruntime")
    exit(1)

# --- 配置管理 ---


class Config:
    VERSION = "1.0.0"
    # 基础路径
    BASE_TEMP_DIR = "/dev/shm" if os.path.exists(
        "/dev/shm") else tempfile.gettempdir()
    OUTPUT_DIR = os.path.join(BASE_TEMP_DIR, "piper_output")
    CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
    MODEL_DIR = os.getenv("MODEL_DIR", "/app/models" if os.path.exists("/app/models") else
                          "/models" if os.path.exists("/models") else
                          os.path.join(os.path.dirname(__file__), "models"))

    # 并发控制
    # 小机器（≤4核）：内存通常也小（如 8GB），严格限制并发，避免同时加载多个模型导致 OOM。
    # 大机器（>4核）：内存通常较大（如 32GB+），可以适当放宽，但始终保留 3 个核心的余量：
    #   1~2 个给系统/OS
    #   1 个给主进程/监控/其他服务
    #   防止 CPU 和内存同时打满导致系统无响应
    # “动态保守并发”策略 —— 核心越多，允许的并发越多，但永远不占满。
    TOTAL_CORES = os.cpu_count() or 2
    MAX_WORKERS = TOTAL_CORES + 2
    # 限制并发推理数，防止内存爆炸
    SAFE_LIMIT = max(1, int(TOTAL_CORES / 2)
                     ) if TOTAL_CORES <= 4 else max(1, TOTAL_CORES - 3)

    # 缓存策略
    CACHE_MAX_SIZE = 1000
    CACHE_MAX_LENGTH = 50
    MAX_STORAGE_BYTES = 512 * 1024 * 1024  # 512MB
    TEMP_FILE_TTL = 300  # 秒

    # 模型配置
    MODELS = {
        "zh": {
            "path": f"{MODEL_DIR}/zh_CN-huayan-x_low.onnx",
            "config": f"{MODEL_DIR}/zh_CN-huayan-x_low.onnx.json"
        },
        "en": {
            "path": f"{MODEL_DIR}/en_US-amy-low.onnx",
            "config": f"{MODEL_DIR}/en_US-amy-low.onnx.json"
        }
    }

    # 日志
    LOG_LEVEL = logging.ERROR


# 初始化目录
Path(Config.CACHE_DIR).mkdir(parents=True, exist_ok=True)

# 日志配置
logging.basicConfig(level=Config.LOG_LEVEL,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 文本处理工具 ---


class TextProcessor:
    SPLIT_PATTERN = re.compile(
        r'([\u4e00-\u9fff]+|[a-zA-Z\s\d.,!?;:\'-]+|[^\u4e00-\u9fff\w\s]+)')
    ZH_CHECK = re.compile(r'[\u4e00-\u9fff]')
    EN_CHECK = re.compile(r'[a-zA-Z]')
    SENTENCE_SPLIT = re.compile(r'([^。！？；.!?;]+[。！？；.!?;]?)')

    @staticmethod
    def normalize_zh(text: str) -> str:
        try:
            return cn2an.transform(text, "cn2an", "smart")
        except Exception:
            return text

    @classmethod
    def split_mixed_text(cls, text: str) -> List[Dict[str, str]]:
        """将混合文本分割为带有语言标记的片段"""
        segments = []
        matches = cls.SPLIT_PATTERN.findall(text)
        for match in matches:
            if not match.strip():
                continue

            if cls.ZH_CHECK.search(match):
                lang = "zh"
            elif cls.EN_CHECK.search(match):
                lang = "en"
            else:
                # 标点符号跟随上一段语言
                lang = segments[-1]["lang"] if segments else "zh"

            if segments and segments[-1]["lang"] == lang:
                segments[-1]["text"] += match
            else:
                segments.append({"text": match, "lang": lang})
        return segments

    @classmethod
    def smart_split_stream(cls, text: str) -> List[str]:
        """
        流式优化的分句策略：
        首句极短(6字符)以实现极速首字，次句稍长，后续稳定。
        """
        raw_sentences = cls.SENTENCE_SPLIT.findall(text)
        final_chunks = []
        current_chunk = ""
        chunk_index = 0

        for sent in raw_sentences:
            if chunk_index == 0:
                target_len = 6
            elif chunk_index == 1:
                target_len = 20
            else:
                target_len = 60

            if len(current_chunk) + len(sent) < target_len:
                current_chunk += sent
            else:
                if current_chunk:
                    final_chunks.append(current_chunk)
                    chunk_index += 1
                current_chunk = sent

        if current_chunk:
            final_chunks.append(current_chunk)
        return final_chunks

# --- 缓存管理 ---


class CacheManager:
    def __init__(self):
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self._restore_from_disk()

    def _restore_from_disk(self):
        """服务重启时恢复缓存索引"""
        try:
            if os.path.exists(Config.CACHE_DIR):
                files = sorted([
                    (os.path.getmtime(os.path.join(Config.CACHE_DIR, f)), f)
                    for f in os.listdir(Config.CACHE_DIR) if f.endswith('.wav')
                ])
                with self.lock:
                    for _, f in files:
                        self.cache[f[:-4]] = os.path.join(Config.CACHE_DIR, f)
        except Exception as e:
            logger.error(f"Failed to restore cache: {e}")

    def get_hash(self, text: str, model_name: str, sid: str, ls: float, ns: float) -> str:
        content = f"{text}|{model_name}|{sid}|{ls}|{ns}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def get(self, key: str) -> Optional[str]:
        with self.lock:
            if key in self.cache:
                wav_path = self.cache[key]
                if os.path.exists(wav_path):
                    self.cache.move_to_end(key)
                    # 返回副本以防止文件被占用或修改
                    temp_copy = tempfile.NamedTemporaryFile(
                        suffix='.wav', dir=Config.OUTPUT_DIR, delete=False)
                    temp_copy.close()
                    shutil.copy2(wav_path, temp_copy.name)
                    return temp_copy.name
                else:
                    del self.cache[key]
        return None

    def put(self, key: str, temp_path: str) -> str:
        """将临时文件移动到缓存目录并记录"""
        with self.lock:
            # LRU 淘汰
            while len(self.cache) >= Config.CACHE_MAX_SIZE:
                _, oldest_path = self.cache.popitem(last=False)
                self._safe_remove(oldest_path)

            target_path = os.path.join(Config.CACHE_DIR, f"{key}.wav")
            shutil.move(temp_path, target_path)
            self.cache[key] = target_path
            return target_path

    def clear(self):
        with self.lock:
            for _, p in list(self.cache.items()):
                self._safe_remove(p)
            self.cache.clear()

    def evict_by_size(self):
        """基于磁盘大小的清理策略"""
        total_size = 0
        file_list = []

        # 计算大小
        if os.path.exists(Config.CACHE_DIR):
            for filename in os.listdir(Config.CACHE_DIR):
                filepath = os.path.join(Config.CACHE_DIR, filename)
                try:
                    size = os.path.getsize(filepath)
                    total_size += size
                    file_list.append(filepath)
                except OSError:
                    pass

        if total_size > Config.MAX_STORAGE_BYTES:
            logger.warning(
                f"⚠️ Cache size {total_size/1024/1024:.1f}MB > Limit. Evicting...")
            with self.lock:
                while total_size > Config.MAX_STORAGE_BYTES and self.cache:
                    key, wav_path = self.cache.popitem(last=False)
                    try:
                        if os.path.exists(wav_path):
                            size = os.path.getsize(wav_path)
                            os.remove(wav_path)
                            total_size -= size
                    except Exception:
                        pass

    @staticmethod
    def _safe_remove(path: str):
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass

# --- 核心引擎 ---


class PiperEngine:
    def __init__(self):
        self.models: Dict[str, PiperVoice] = {}
        self.semaphore = threading.Semaphore(Config.SAFE_LIMIT)
        self.cache_manager = CacheManager()
        self._load_models()

    def _load_models(self):
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        for lang, cfg in Config.MODELS.items():
            try:
                self.models[lang] = PiperVoice.load(
                    cfg["path"], config_path=cfg["config"], use_cuda=False
                )
                logger.info(f"Loaded model: {lang}")
            except Exception as e:
                logger.error(f"Failed to load {lang} model: {e}")

    def synthesize(self, text: str, lang_code: str,
                   speaker_id: Optional[int] = None,  # 恢复默认值为 None
                   length_scale: float = 1.0, noise_scale: float = 0.667,
                   stop_event: threading.Event = None) -> Optional[str]:

        if stop_event and stop_event.is_set():
            return None

        voice = self.models.get(lang_code)
        if not voice:
            return None

        # 【关键修复】根据模型类型处理 speaker_id
        # 如果模型只有一个说话人，必须传入 None，否则会报 sid 错误
        if voice.config.num_speakers <= 1:
            speaker_id = None
        else:
            # 如果是多人模型且用户没传 id，默认给 0
            speaker_id = 0 if speaker_id is None else int(speaker_id)

        # 参数归一化（仅用于生成缓存 Key）
        sid_str = str(speaker_id) if speaker_id is not None else "single"
        ls_val = length_scale if length_scale is not None else 1.0
        ns_val = noise_scale if noise_scale is not None else 0.667

        # 1. 检查缓存
        model_name = "zh_CN" if lang_code == "zh" else "en_US"
        is_cacheable = len(text) <= Config.CACHE_MAX_LENGTH

        if is_cacheable:
            ls_str = f"{ls_val:.1f}"
            ns_str = f"{ns_val:.3f}"
            cache_key = self.cache_manager.get_hash(
                text, model_name, sid_str, ls_str, ns_str)
            cached_file = self.cache_manager.get(cache_key)
            if cached_file:
                return cached_file

        # 2. 推理
        temp_wav = tempfile.NamedTemporaryFile(
            suffix='.wav', dir=Config.OUTPUT_DIR, delete=False)
        temp_wav.close()

        try:
            with self.semaphore:
                if stop_event and stop_event.is_set():
                    os.remove(temp_wav.name)
                    return None

                # 清洗非法字符
                clean_text = re.sub(
                    r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？,!.?]', '', text)

                with wave.open(temp_wav.name, "wb") as wav_file:

                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(voice.config.sample_rate)

                    voice.synthesize(clean_text, wav_file,
                                     speaker_id=speaker_id,  # 此时如果是单人模型，这里是 None
                                     length_scale=ls_val,
                                     noise_scale=ns_val,
                                     sentence_silence=0.05)

            # 3. 存入缓存或返回
            if is_cacheable:
                return self.cache_manager.put(cache_key, temp_wav.name)
            return temp_wav.name

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            if os.path.exists(temp_wav.name):
                os.remove(temp_wav.name)
            return None

    # def synthesize(self, text: str, lang_code: str,
    #                speaker_id: Optional[int] = None,
    #                length_scale: float = 1.0, noise_scale: float = 0.667,
    #                stop_event: threading.Event = None) -> Optional[str]:

    #     if stop_event and stop_event.is_set():
    #         return None

    #     voice = self.models.get(lang_code)
    #     if not voice:
    #         return None

    #     # --- FIX: 安全处理 None 值 ---
    #     # 如果前端没传参数，这里会接收到 None，必须先转为默认浮点数
    #     ls_val = length_scale if length_scale is not None else 1.0
    #     ns_val = noise_scale if noise_scale is not None else 0.667

    #     # 逻辑：确定 speaker_id
    #     if voice.config.num_speakers <= 1:
    #         sid_val = None
    #     else:
    #         sid_val = 0 if speaker_id is None else int(speaker_id)

    #     # 1. 检查缓存
    #     # 使用处理后的 ls_val 和 ns_val 生成 Key，确保不会出现 formatting NoneType 错误
    #     sid_str = str(sid_val) if sid_val is not None else "single"
    #     ls_str = f"{ls_val:.1f}"
    #     ns_str = f"{ns_val:.3f}"

    #     model_name = "zh_CN" if lang_code == "zh" else "en_US"
    #     is_cacheable = len(text) <= Config.CACHE_MAX_LENGTH

    #     if is_cacheable:
    #         cache_key = self.cache_manager.get_hash(
    #             text, model_name, sid_str, ls_str, ns_str)
    #         cached_file = self.cache_manager.get(cache_key)
    #         if cached_file:
    #             return cached_file

    #     # 2. 推理
    #     temp_wav = tempfile.NamedTemporaryFile(
    #         suffix='.wav', dir=Config.OUTPUT_DIR, delete=False)
    #     temp_wav.close()

    #     try:
    #         with self.semaphore:
    #             if stop_event and stop_event.is_set():
    #                 os.remove(temp_wav.name)
    #                 return None

    #             # 清洗文本
    #             clean_text = re.sub(
    #                 r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？,!.?]', '', text)

    #             with wave.open(temp_wav.name, "wb") as wav_file:
    #                 # 必须手动设置 WAV 格式，否则报错 channels not specified
    #                 wav_file.setnchannels(1)
    #                 wav_file.setsampwidth(2)
    #                 wav_file.setframerate(voice.config.sample_rate)

    #                 # --- 兼容性调用 ---
    #                 # 仅保留最基础参数，适配 piper-tts 官方 PyPI 简版
    #                 # 如果你安装的是完整版 wheel，这里可以恢复 length_scale 等参数
    #                 voice.synthesize(clean_text, wav_file)

    #         # 3. 存入缓存或返回
    #         if is_cacheable:
    #             return self.cache_manager.put(cache_key, temp_wav.name)
    #         return temp_wav.name

    #     except Exception as e:
    #         logger.error(f"Synthesis failed: {e}")
    #         if os.path.exists(temp_wav.name):
    #             os.remove(temp_wav.name)
    #         return None


# --- 全局实例 ---
app = Flask(__name__)
CORS(app)
sock = Sock(app)
engine = PiperEngine()

# --- 辅助功能 ---


def merge_audio_files(file_paths: List[str]) -> Optional[str]:
    """合并多个WAV文件"""
    if not file_paths:
        return None
    if len(file_paths) == 1:
        return file_paths[0]

    merged_output = tempfile.NamedTemporaryFile(
        suffix='.wav', dir=Config.OUTPUT_DIR, delete=False)
    merged_output.close()

    try:
        data = []
        params = None
        for f in file_paths:
            with wave.open(f, 'rb') as w:
                if not params:
                    params = w.getparams()
                data.append(w.readframes(w.getnframes()))

        with wave.open(merged_output.name, 'wb') as w:
            if params:
                w.setparams(params)
            for d in data:
                w.writeframes(d)

        # 清理临时片段 (不清理缓存内的文件)
        for f in file_paths:
            if Config.CACHE_DIR not in os.path.abspath(f):
                try:
                    os.remove(f)
                except OSError:
                    pass
        return merged_output.name
    except Exception as e:
        logger.error(f"Merge error: {e}")
        return None


def stream_generator(text: str, speaker_id, length_scale, noise_scale, stop_event: threading.Event):
    """流式生成器：管理并发任务和有序输出"""
    text = TextProcessor.normalize_zh(text)
    sentences = TextProcessor.smart_split_stream(text)

    # 构建所有细分片段的任务列表
    tasks = []
    for sent in sentences:
        tasks.extend(TextProcessor.split_mixed_text(sent))

    if not tasks:
        return

    result_queue = __import__('queue').Queue()

    # 提交任务
    def producer():
        with concurrent.futures.ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            futures = []
            for idx, task in enumerate(tasks):
                if stop_event.is_set():
                    break

                future = executor.submit(
                    engine.synthesize,
                    task["text"], task["lang"],
                    speaker_id, length_scale, noise_scale, stop_event
                )

                # 闭包捕获索引
                def on_done(f, i=idx):
                    if stop_event.is_set():
                        return
                    try:
                        path = f.result()
                        status = 'ready' if path else 'error'
                        result_queue.put((status, i, path))
                    except:
                        result_queue.put(('error', i, None))

                future.add_done_callback(on_done)
                futures.append(future)

    threading.Thread(target=producer, daemon=True).start()

    # 消费结果 (保证顺序)
    buffer = {}
    next_idx = 0
    total = len(tasks)

    while next_idx < total:
        if stop_event.is_set():
            break

        # 检查缓冲区是否已有当前需要的片段
        if next_idx in buffer:
            wav_path = buffer.pop(next_idx)
            if wav_path:
                yield from pcm_chunk_reader(wav_path, stop_event)
            next_idx += 1
            continue

        try:
            status, idx, path = result_queue.get(timeout=0.1)
            if idx == next_idx:
                if status == 'ready' and path:
                    yield from pcm_chunk_reader(path, stop_event)
                next_idx += 1
            else:
                if status == 'ready' and path:
                    buffer[idx] = path
        except:
            # Queue Empty or Timeout
            continue


def pcm_chunk_reader(wav_path: str, stop_event: threading.Event) -> Generator[bytes, None, None]:
    """读取WAV PCM数据，首包极速策略"""
    try:
        with wave.open(wav_path, 'rb') as wf:
            # 简单的格式校验
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                return

            first_chunk = True
            while not stop_event.is_set():
                # 首包 1024 (64ms)，后续 4096 (256ms)
                size = 1024 if first_chunk else 4096
                data = wf.readframes(size)
                if not data:
                    break
                yield data
                first_chunk = False
    except Exception:
        pass
    finally:
        # 如果不是缓存文件，读取完即删除
        if Config.CACHE_DIR not in os.path.abspath(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass

# --- 后台维护 ---


def maintenance_worker():
    """定期清理僵尸文件和执行LRU淘汰"""
    while True:
        time.sleep(30)
        try:
            current_time = time.time()

            # 1. 清理 Config.OUTPUT_DIR 根目录下的超时文件
            if os.path.exists(Config.OUTPUT_DIR):
                for fname in os.listdir(Config.OUTPUT_DIR):
                    fpath = os.path.join(Config.OUTPUT_DIR, fname)
                    if os.path.isdir(fpath):
                        continue

                    try:
                        # 只清理 wav 且超时的
                        if fname.endswith('.wav') and (current_time - os.path.getmtime(fpath) > Config.TEMP_FILE_TTL):
                            os.remove(fpath)
                            logger.info(f"🧹 Zombie cleaned: {fname}")
                    except OSError:
                        pass

            # 2. 触发缓存大小检查
            engine.cache_manager.evict_by_size()

        except Exception as e:
            logger.error(f"Maintenance error: {e}")

# --- API 路由 ---


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "version": Config.VERSION,  # 新增此字段供前端展示
        "models": list(engine.models.keys()),
        "cache_count": len(engine.cache_manager.cache)
    })


@app.route('/synthesize', methods=['POST'])
def api_synthesize():
    try:
        data = request.get_json() or {}
        text = data.get('text', '')
        if not text:
            return jsonify({"error": "Empty text"}), 400

        segments = TextProcessor.split_mixed_text(text)
        results = [None] * len(segments)

        # 同样使用线程池并发合成
        with concurrent.futures.ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            future_map = {
                executor.submit(
                    engine.synthesize,
                    seg["text"], seg["lang"],
                    data.get('speaker_id'), data.get(
                        'length_scale'), data.get('noise_scale')
                ): i for i, seg in enumerate(segments)
            }

            for future in concurrent.futures.as_completed(future_map):
                idx = future_map[future]
                results[idx] = future.result()

        # 过滤失败的片段
        valid_paths = [p for p in results if p]
        final_wav = merge_audio_files(valid_paths)

        if final_wav:
            return send_file(final_wav, mimetype="audio/wav")
        return jsonify({"error": "Synthesis failed"}), 500

    except Exception as e:
        logger.error(f"API Error: {e}")
        return jsonify({"error": str(e)}), 500


@sock.route('/ws/synthesize')
def ws_synthesize(ws):
    stop_event = threading.Event()
    try:
        while True:
            data = ws.receive()
            if not data:
                break

            try:
                params = json.loads(data)
            except json.JSONDecodeError:
                continue

            iterator = stream_generator(
                params.get('text', ''),
                params.get('speaker_id'),
                params.get('length_scale'),
                params.get('noise_scale'),
                stop_event
            )

            for chunk in iterator:
                if stop_event.is_set():
                    break
                try:
                    ws.send(chunk)
                except Exception:
                    # WebSocket 发送失败通常意味着连接断开
                    stop_event.set()
                    break

            if not stop_event.is_set():
                try:
                    ws.send("END")
                except Exception:
                    pass

            time.sleep(3.0)
            break

    except Exception as e:
        logger.error(f"WS Error: {e}")
    finally:
        stop_event.set()


@app.route('/cache/clear', methods=['POST'])
def api_clear_cache():
    engine.cache_manager.clear()
    return jsonify({"status": "cleared"})


if __name__ == '__main__':
    # 启动后台维护线程
    t = threading.Thread(target=maintenance_worker, daemon=True)
    t.start()
    logger.info("🚀 Piper Service Started with Optimized Architecture")

    port = int(os.environ.get('PIPER_PORT', 5001))
    app.run(host='0.0.0.0', port=port, threaded=True)
