"""
API工具模块
提供FastAPI应用初始化、路由处理和工具函数
"""

# 应用初始化
from .app import create_app

# 队列工作器
from .queue_worker import queue_worker

# 请求处理器
from .request_processor import (
    _process_request_refactored,  # pyright: ignore[reportPrivateUsage]
)

# 路由处理器（改为从 routers 聚合导入）
from .routers import (
    audio_speech,
    cancel_request,
    chat_completions,
    get_api_info,
    get_queue_status,
    health_check,
    list_models,
    list_tts_models,
    list_tts_voices,
    read_index,
    websocket_log_endpoint,
)
from .sse import (
    generate_sse_chunk,
    generate_sse_error_chunk,
    generate_sse_stop_chunk,
)

# TTS处理器
from .tts_processor import process_tts_request

# 工具函数
from .utils import prepare_combined_prompt
from .utils_ext.helper import use_helper_get_response
from .utils_ext.stream import (
    clear_stream_queue,
    use_stream_response,
)
from .utils_ext.tokens import (
    calculate_usage_stats,
    estimate_tokens,
)
from .utils_ext.validation import validate_chat_request

__all__ = [
    # 应用初始化
    "create_app",
    # 路由処理器
    "read_index",
    "get_api_info",
    "health_check",
    "list_models",
    "chat_completions",
    "cancel_request",
    "get_queue_status",
    "websocket_log_endpoint",
    # TTS路由
    "audio_speech",
    "list_tts_voices",
    "list_tts_models",
    # 工具函数
    "generate_sse_chunk",
    "generate_sse_stop_chunk",
    "generate_sse_error_chunk",
    "use_stream_response",
    "clear_stream_queue",
    "use_helper_get_response",
    "validate_chat_request",
    "prepare_combined_prompt",
    "estimate_tokens",
    "calculate_usage_stats",
    # 请求处理器
    "_process_request_refactored",
    # TTS处理器
    "process_tts_request",
    # 队列工作器
    "queue_worker",
]
