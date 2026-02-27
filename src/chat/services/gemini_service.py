# -*- coding: utf-8 -*-

import os
import logging
from typing import Optional, Dict, List, Callable, Any, Tuple, get_origin, get_args, Union
import asyncio
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import re
import base64

from PIL import Image
import io
import httpx

# 导入新库
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

# 导入数据库管理器和提示词配置
from src.chat.utils.database import chat_db_manager
from src.chat.config import chat_config as app_config
from src.chat.utils.prompt_utils import replace_emojis
from src.chat.services.prompt_service import prompt_service
from src.chat.services.key_rotation_service import (
    KeyRotationService,
    NoAvailableKeyError,
)
from src.chat.features.tools.services.tool_service import ToolService
from src.chat.features.tools.tool_loader import load_tools_from_directory
from src.chat.features.chat_settings.services.chat_settings_service import (
    chat_settings_service,
)
from src.chat.utils.image_utils import sanitize_image
from src.database.services.token_usage_service import token_usage_service
from src.database.database import AsyncSessionLocal


log = logging.getLogger(__name__)

# --- 设置专门用于记录无效 API 密钥的 logger ---
# 确保 data 目录存在
if not os.path.exists("data"):
    os.makedirs("data")

# 创建一个新的 logger 实例
invalid_key_logger = logging.getLogger("invalid_api_keys")
invalid_key_logger.setLevel(logging.ERROR)

# 创建文件处理器，将日志写入到 data/invalid_api_keys.log
# 使用 a 模式表示追加写入
fh = logging.FileHandler("data/invalid_api_keys.log", mode="a", encoding="utf-8")
fh.setLevel(logging.ERROR)

# 创建格式化器并设置
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)

# 为 logger 添加处理器
# 防止重复添加处理器
if not invalid_key_logger.handlers:
    invalid_key_logger.addHandler(fh)


def _api_key_handler(func: Callable) -> Callable:
    """
    双轨制装饰器：
    1. generate_embedding -> 强制使用 GOOGLE_API_KEYS_LIST (官方直连)
    2. 其他方法 (聊天) -> 优先使用 CUSTOM_GEMINI_URL (自定义通道)
    """
    @wraps(func)
    async def wrapper(self: "GeminiService", *args, **kwargs):
        # [逻辑分流] 判断当前任务类型
        is_embedding_task = func.__name__ == "generate_embedding"

        # 一次性调试 URL（仅供本轮调用使用）
        debug_base_url = kwargs.pop("debug_base_url", None)

        # [逻辑分流] 决定是否使用自定义通道 (非向量化 且 配置了自定义参数)
        use_custom_channel = (not is_embedding_task) and bool(self.custom_chat_key) and bool(
            debug_base_url or self.custom_chat_url
        )

        # 定义一个简单的类来模拟 key_obj，用于自定义通道场景
        class MockKeyObj:
            def __init__(self, k): 
                self.key = k
                self.consecutive_failures = 0

        while True:
            key_obj = None
            client = None
            
            try:
                # --- [核心修改点] 根据通道类型创建不同的 Client ---
                if use_custom_channel:
                    # 【通道 A：聊天】走自定义中转（支持一次性调试 URL 覆盖）
                    effective_custom_url = debug_base_url or self.custom_chat_url
                    key_obj = MockKeyObj(self.custom_chat_key)
                    http_options = types.HttpOptions(base_url=effective_custom_url)
                    client = genai.Client(
                        api_key=self.custom_chat_key, http_options=http_options
                    )
                    if debug_base_url:
                        log.info(
                            f"🧪 一次性调试已生效：{func.__name__} 临时改用 URL: {debug_base_url}"
                        )

                else:
                    # 【通道 B：向量】走官方直连 (或者是聊天回退)
                    # 从轮换服务获取官方 Key
                    key_obj = await self.key_rotation_service.acquire_key()
                    if debug_base_url:
                        http_options = types.HttpOptions(base_url=debug_base_url)
                        client = genai.Client(api_key=key_obj.key, http_options=http_options)
                        log.info(
                            f"🧪 一次性调试已生效：{func.__name__} 临时改用官方通道 URL: {debug_base_url}"
                        )
                    else:
                        # 强制直连 (不传 http_options，默认就是 google 官方)
                        client = genai.Client(api_key=key_obj.key)
                    # log.info(f"使用官方通道调用 {func.__name__}")

                # --- 下面的逻辑保留原有的重试和错误处理框架 ---
                
                failure_penalty = 25
                key_should_be_cooled_down = False
                key_is_invalid = False

                max_attempts = app_config.API_RETRY_CONFIG["MAX_ATTEMPTS_PER_KEY"]
                # 如果是自定义通道，其实不需要循环尝试不同 Key，但为了复用代码结构，还是保留循环
                for attempt in range(max_attempts):
                    try:
                        # 只有官方通道才需要打印详细的 Key 轮换日志
                        if not use_custom_channel:
                            log.info(
                                f"使用密钥 ...{key_obj.key[-4:]} (尝试 {attempt + 1}/{max_attempts}) 调用 {func.__name__}"
                            )

                        kwargs["client"] = client
                        result = await func(self, *args, **kwargs)

                        safety_penalty = 0
                        is_blocked_by_safety = False
                        if isinstance(result, types.GenerateContentResponse):
                            safety_penalty = self._handle_safety_ratings(
                                result, key_obj.key
                            )
                            if (
                                not result.parts
                                and result.prompt_feedback
                                and result.prompt_feedback.block_reason
                            ):
                                is_blocked_by_safety = True

                        if is_blocked_by_safety:
                            log.warning(
                                f"安全策略拦截: {result.prompt_feedback.block_reason if result.prompt_feedback else '未知'}。"
                            )
                            failure_penalty = 0
                            key_should_be_cooled_down = True
                            break

                        # [重要] 只有官方 Key 才需要释放回池子
                        if not use_custom_channel:
                            await self.key_rotation_service.release_key(
                                key_obj.key, success=True, safety_penalty=safety_penalty
                            )
                        return result

                    except (genai_errors.ClientError, genai_errors.ServerError) as e:
                        error_str = str(e)
                        match = re.match(r"(\d{3})", error_str)
                        status_code = int(match.group(1)) if match else None
                        
                        # 简单的可重试判断
                        is_retryable = status_code in [429, 503]
                        if isinstance(e, genai_errors.ServerError) and "503" in error_str:
                             is_retryable = True

                        if is_retryable:
                            log.warning(f"可重试错误 ({status_code})")
                            if attempt < max_attempts - 1:
                                await asyncio.sleep(1) # 简单等待
                            else:
                                key_should_be_cooled_down = True
                        
                        elif status_code in [400, 403] and "API_KEY" in str(e).upper():
                             log.error(f"密钥无效 ({status_code})")
                             key_is_invalid = True # 标记为无效，触发外层循环换 Key (仅限官方)
                             failure_penalty = 101
                             key_should_be_cooled_down = True
                             break
                        else:
                            # 其他错误
                            log.error(f"API调用发生错误: {e}", exc_info=True)
                            # 如果是自定义通道报错，直接返回错误提示，不再尝试轮换
                            if use_custom_channel:
                                return "服务连接失败，请稍后再试。"
                                
                            await self.key_rotation_service.release_key(key_obj.key, success=True)
                            if func.__name__ == "generate_embedding":
                                return None
                            return "呜哇，有点晕嘞，等我休息一会儿 <伤心>"

                if key_is_invalid:
                    if use_custom_channel:
                         return "配置的自定义 API Key 无效。"
                    continue # 官方通道：换下一个 Key 重试

                if key_should_be_cooled_down:
                    if not use_custom_channel:
                        await self.key_rotation_service.release_key(
                            key_obj.key, success=False, failure_penalty=failure_penalty
                        )
                    else:
                        # 自定义通道冷却：简单休眠一下
                        await asyncio.sleep(2)

            except NoAvailableKeyError:
                log.error("所有官方 API 密钥均不可用。")
                if func.__name__ == "generate_embedding":
                    return None
                return "啊啊啊服务器要爆炸啦！现在有点忙不过来，你过一会儿再来找我玩吧！<生气>"
            except Exception as outer_e:
                log.error(f"严重未知错误: {outer_e}", exc_info=True)
                if func.__name__ == "generate_embedding":
                    return None
                return "系统发生严重错误。"

    return wrapper


class GeminiService:
    """Gemini AI 服务类，使用数据库存储用户对话上下文"""

    SAFETY_PENALTY_MAP: Dict[str, int] = {
        "HARM_PROBABILITY_UNSPECIFIED": 0,
        "NEGLIGIBLE": 0,
        "LOW": 5,
        "MEDIUM": 15,
        "HIGH": 30,
    }

    def __init__(self):
        self.bot = None  # 用于存储 Discord Bot 实例
        self.custom_chat_url = os.getenv("CUSTOM_GEMINI_URL")
        self.custom_chat_key = os.getenv("CUSTOM_GEMINI_API_KEY")
        if self.custom_chat_url and self.custom_chat_key:
            log.info(f"✅ 已启用自定义聊天通道: {self.custom_chat_url}")
        else:
            log.info("ℹ️ 未检测到完整的自定义聊天配置，将回退到官方 API。")

        # --- [新增] DeepSeek 独立配置 ---
        self.deepseek_url = os.getenv("DEEPSEEK_URL")
        self.deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        if self.deepseek_url and self.deepseek_key:
            log.info(f"✅ 已加载 DeepSeek 配置。URL: {self.deepseek_url}")

        # 一次性调试 URL（下一次发送生效一次后自动清空）
        self._one_time_debug_base_url: Optional[str] = None

        # --- (新) SDK 底层调试日志 ---
        # 根据最新指南 (2025)，开启此选项可查看详细的 HTTP 请求/响应
        if app_config.DEBUG_CONFIG.get("LOG_SDK_HTTP_REQUESTS", False):
            log.info("已开启 google-genai SDK 底层 DEBUG 日志。")
            # 设置基础日志记录器以捕获 httpx 的调试信息
            logging.basicConfig(level=logging.DEBUG)
        # --- 密钥轮换服务（用于RAG功能）---
        google_api_keys_str = os.getenv("GOOGLE_API_KEYS_LIST", "")
        processed_keys_str = google_api_keys_str.strip().strip('"')
        api_keys = [key.strip() for key in processed_keys_str.split(",") if key.strip()]

        if api_keys:
            self.key_rotation_service = KeyRotationService(api_keys)
            log.info(
                f"GeminiService 初始化并由 KeyRotationService 管理 {len(api_keys)} 个密钥。"
            )
        else:
            self.key_rotation_service = None
            log.warning("未配置 GOOGLE_API_KEYS_LIST，RAG检索功能将被禁用。")

        self.default_model_name = app_config.GEMINI_MODEL
        self.executor = ThreadPoolExecutor(
            max_workers=app_config.MAX_CONCURRENT_REQUESTS
        )
        self.user_request_timestamps: Dict[int, List[datetime]] = {}
        self.safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]

        # --- 工具配置 (模块化标准) ---
        # 1. 使用加载器动态发现所有工具
        self.available_tools, self.tool_map = load_tools_from_directory(
            "src/chat/features/tools/functions"
        )

        # 2. 实例化工具服务，并传入工具映射
        self.tool_service = ToolService(
            bot=self.bot, tool_map=self.tool_map, tool_declarations=self.available_tools
        )

        log.info("--- 工具加载完成 (模块化) ---")
        log.info(
            f"已加载 {len(self.available_tools)} 个工具: {list(self.tool_map.keys())}"
        )
        log.info("------------------------------------")

    def set_bot(self, bot):
        """注入 Discord Bot 实例。"""
        self.bot = bot
        log.info("Discord Bot 实例已成功注入 GeminiService。")
        # 关键：同时将 bot 实例注入到 ToolService 中
        self.tool_service.bot = bot
        self.last_called_tools: List[str] = []
        log.info("Discord Bot 实例已成功注入 ToolService。")

    def arm_one_time_debug_base_url(self, base_url: str) -> None:
        """设置一次性调试 URL：仅下一次 generate_response 生效一次。"""
        self._one_time_debug_base_url = base_url.rstrip("/")
        log.info(f"🧪 已激活一次性调试 URL: {self._one_time_debug_base_url}")

    def consume_one_time_debug_base_url(self) -> Optional[str]:
        """消费一次性调试 URL，并立即清空。"""
        current = self._one_time_debug_base_url
        self._one_time_debug_base_url = None
        if current:
            log.info(f"🧪 已消费一次性调试 URL: {current}")
        return current

    def _create_client_with_key(self, api_key: str):
        """使用给定的 API 密钥动态创建一个 Gemini 客户端实例。"""
        base_url = os.getenv("GEMINI_API_BASE_URL")
        if base_url:
            log.info(f"使用自定义 Gemini API 端点: {base_url}")
            # 根据用户提供的文档，正确的方法是使用 types.HttpOptions
            # Cloudflare Worker 需要 /gemini 后缀，所以我们不移除它
            http_options = types.HttpOptions(base_url=base_url)
            return genai.Client(api_key=api_key, http_options=http_options)
        else:
            log.info("使用默认 Gemini API 端点。")
            return genai.Client(api_key=api_key)

    async def get_user_conversation_history(
        self, user_id: int, guild_id: int
    ) -> List[Dict]:
        """从数据库获取用户的对话历史"""
        context = await chat_db_manager.get_ai_conversation_context(user_id, guild_id)
        if context and context.get("conversation_history"):
            return context["conversation_history"]
        return []

    # --- Refactored Cooldown Logic ---

    # --- Static Helper Methods for Serialization ---
    @staticmethod
    def _serialize_for_logging(obj):
        """自定义序列化函数，用于截断长文本以进行日志记录。"""
        if isinstance(obj, dict):
            return {
                key: GeminiService._serialize_for_logging(value)
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [GeminiService._serialize_for_logging(item) for item in obj]
        elif isinstance(obj, str) and len(obj) > 200:
            return obj[:200] + "..."
        elif isinstance(obj, Image.Image):
            return f"<PIL.Image object: mode={obj.mode}, size={obj.size}>"
        else:
            try:
                json.JSONEncoder().default(obj)
                return obj
            except TypeError:
                return str(obj)

    @staticmethod
    def _serialize_parts_for_error_logging(obj):
        """自定义序列化函数，用于在出现问题时记录请求体。"""
        if isinstance(obj, types.Part):
            if obj.text:
                return {"type": "text", "content": obj.text}
            elif obj.inline_data:
                return {
                    "type": "image",
                    "mime_type": obj.inline_data.mime_type,
                    "data_size": len(obj.inline_data.data)
                    if obj.inline_data and obj.inline_data.data
                    else 0,
                }
        elif isinstance(obj, Image.Image):
            return f"<PIL.Image object: mode={obj.mode}, size={obj.size}>"
        try:
            return json.JSONEncoder().default(obj)
        except TypeError:
            return str(obj)

    @staticmethod
    def _serialize_parts_for_logging_full(content: types.Content):
        """自定义序列化函数，用于完整记录 Content 对象。"""
        serialized_parts = []
        if content.parts:
            for part in content.parts:
                if part.text:
                    serialized_parts.append({"type": "text", "content": part.text})
                elif part.inline_data and part.inline_data.data:
                    serialized_parts.append(
                        {
                            "type": "image",
                            "mime_type": part.inline_data.mime_type,
                            "data_size": len(part.inline_data.data),
                            "data_preview": part.inline_data.data[:50].hex()
                            + "...",  # 记录数据前50字节的十六进制预览
                        }
                    )
                elif part.file_data:
                    serialized_parts.append(
                        {
                            "type": "file",
                            "mime_type": part.file_data.mime_type,
                            "file_uri": part.file_data.file_uri,
                        }
                    )
                else:
                    serialized_parts.append(
                        {"type": "unknown_part", "content": str(part)}
                    )
        return {"role": content.role, "parts": serialized_parts}

    # --- Refactored generate_response and its helpers ---
    def _prepare_api_contents(self, conversation: List[Dict]) -> List[types.Content]:
        """将对话历史转换为 API 所需的 Content 对象列表。"""
        processed_contents = []
        for turn in conversation:
            role = turn.get("role")
            parts_data = turn.get("parts", [])
            if not (role and parts_data):
                continue

            processed_parts = []
            for part_item in parts_data:
                if isinstance(part_item, str):
                    processed_parts.append(types.Part(text=part_item))
                elif isinstance(part_item, Image.Image):
                    buffered = io.BytesIO()
                    part_item.save(buffered, format="PNG")
                    img_bytes = buffered.getvalue()
                    try:
                        img_bytes, mime_type = sanitize_image(img_bytes)
                    except Exception as e:
                        log.error(f"图片处理失败: {e}")
                        mime_type = "image/png"
                    processed_parts.append(
                        types.Part(
                            inline_data=types.Blob(
                                mime_type=mime_type, data=img_bytes
                            )
                        )
                    )

            if processed_parts:
                processed_contents.append(
                    types.Content(role=role, parts=processed_parts)
                )
        return processed_contents

    async def _post_process_response(
        self, raw_response: str, user_id: int, guild_id: int
    ) -> str:
        """对 AI 的原始回复进行清理和处理。"""
        # 1. Clean various reply prefixes and tags
        reply_prefix_pattern = re.compile(
            r"^\s*([\[［]【回复|回复}\s*@.*?[\)）\]］])\s*", re.IGNORECASE
        )
        formatted = reply_prefix_pattern.sub("", raw_response)
        formatted = re.sub(
            r"<CURRENT_USER_MESSAGE_TO_REPLY.*?>", "", formatted, flags=re.IGNORECASE
        )
        # formatted = regex_service.clean_ai_output(formatted)

        # 2. Remove old Discord emoji codes
        discord_emoji_pattern = re.compile(r":\w+:")
        formatted = discord_emoji_pattern.sub("", formatted)

        # 3. Replace custom emoji placeholders using the centralized function
        formatted = replace_emojis(formatted)

        return formatted

    def _handle_safety_ratings(
        self, response: types.GenerateContentResponse, key: str
    ) -> int:
        """检查响应的安全评分并返回相应的惩罚值。"""
        total_penalty = 0
        if not response.candidates:
            return 0

        candidate = response.candidates[0]
        if candidate.safety_ratings:
            for rating in candidate.safety_ratings:
                # 将枚举值转换为字符串键
                category_name = (
                    rating.category.name.replace("HARM_CATEGORY_", "")
                    if rating.category
                    else "UNKNOWN"
                )
                severity_name = (
                    rating.probability.name if rating.probability else "UNKNOWN"
                )

                penalty = self.SAFETY_PENALTY_MAP.get(severity_name, 0)
                if penalty > 0:
                    log.warning(
                        f"密钥 ...{key[-4:]} 收到安全警告。类别: {category_name}, 严重性: {severity_name}, 惩罚: {penalty}"
                    )
                    total_penalty += penalty
        return total_penalty

    async def generate_response(
        self,
        user_id: int,
        guild_id: int,
        message: str,
        channel: Optional[Any] = None,
        replied_message: Optional[str] = None,
        images: Optional[List[Dict]] = None,
        user_name: str = "用户",
        channel_context: Optional[List[Dict]] = None,
        world_book_entries: Optional[List[Dict]] = None,
        personal_summary: Optional[str] = None,
        affection_status: Optional[Dict[str, Any]] = None,
        user_profile_data: Optional[Dict[str, Any]] = None,
        guild_name: str = "未知服务器",
        location_name: str = "未知位置",
        model_name: Optional[str] = None,
        user_id_for_settings: Optional[str] = None,
    ) -> str:
        """
        AI 回复生成的分发器。
        如果选择了自定义模型，则优先尝试自定义端点；如果失败，则自动回退到官方 API。
        """
        # 一次性调试 URL：在入口处消费，确保只生效 1 次
        one_time_debug_base_url = self.consume_one_time_debug_base_url()
        if one_time_debug_base_url:
            log.info(
                f"🧪 generate_response 已启用一次性调试 URL: {one_time_debug_base_url}，模型: {model_name or self.default_model_name}"
            )

        # --- [新增] DeepSeek 专用路由 ---
        if model_name in ["deepseek-chat", "deepseek-reasoner"]:
            if self.deepseek_url and self.deepseek_key:
                log.info(f"检测到 {model_name} 模型，切换至 DeepSeek 专用通道。")
                return await self._generate_with_deepseek(
                    user_id=user_id,
                    guild_id=guild_id,
                    message=message,
                    channel=channel,
                    replied_message=replied_message,
                    images=images,
                    user_name=user_name,
                    channel_context=channel_context,
                    world_book_entries=world_book_entries,
                    personal_summary=personal_summary,
                    affection_status=affection_status,
                    user_profile_data=user_profile_data,
                    guild_name=guild_name,
                    location_name=location_name,
                    model_name=model_name,
                    user_id_for_settings=user_id_for_settings,
                    override_base_url=one_time_debug_base_url,
                )
            else:
                log.warning("请求使用 deepseek-chat 但未配置 DEEPSEEK_URL 或 DEEPSEEK_API_KEY。")
                return "DeepSeek 配置缺失，请检查环境变量。"

        # 如果选择了自定义模型，则尝试使用它，并准备好回退
        if model_name and model_name in app_config.CUSTOM_GEMINI_ENDPOINTS:
            log.info(f"检测到自定义模型 '{model_name}'，将优先尝试使用自定义端点。")
            max_attempts = 2  # 1次主尝试 + 1次重试
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    log.info(
                        f"尝试使用自定义端点 '{model_name}' (尝试 {attempt + 1}/{max_attempts})"
                    )
                    return await self._generate_with_custom_endpoint(
                        user_id=user_id,
                        guild_id=guild_id,
                        message=message,
                        channel=channel,
                        replied_message=replied_message,
                        images=images,
                        user_name=user_name,
                        channel_context=channel_context,
                        world_book_entries=world_book_entries,
                        personal_summary=personal_summary,
                        affection_status=affection_status,
                        user_profile_data=user_profile_data,
                        guild_name=guild_name,
                        location_name=location_name,
                        model_name=model_name,
                        user_id_for_settings=user_id_for_settings,
                        override_base_url=one_time_debug_base_url,
                    )
                except Exception as e:
                    last_exception = e
                    log.warning(
                        f"使用自定义端点 '{model_name}' (尝试 {attempt + 1}/{max_attempts}) 失败: {e}"
                    )
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(1)  # 在重试前稍作等待

            # 如果所有尝试都失败了，则执行回退逻辑
            # 检查是否配置了RAG API密钥
            if self.key_rotation_service is None:
                log.error(
                    f"自定义端点 '{model_name}' 的所有 {max_attempts} 次尝试均失败。最终错误: {last_exception}. "
                    f"未配置 GOOGLE_API_KEYS_LIST，无法回退到官方 API。"
                )
                return "呜哇，有点晕嘞，自定义端点连接失败了，还没有配置备用API密钥呢，等配置好了再来找我玩吧！"

            log.warning(
                f"自定义端点 '{model_name}' 的所有 {max_attempts} 次尝试均失败。最终错误: {last_exception}. "
                f"将回退到官方 API。"
            )
            # --- [新逻辑] 回退时固定使用 gemini-2.5-flash 模型 ---
            fallback_model_name = "gemini-2.5-flash"
            log.info(f"回退到官方 API，固定使用模型 '{fallback_model_name}'。")

            return await self._generate_with_official_api(
                user_id=user_id,
                guild_id=guild_id,
                message=message,
                channel=channel,
                replied_message=replied_message,
                images=images,
                user_name=user_name,
                channel_context=channel_context,
                world_book_entries=world_book_entries,
                personal_summary=personal_summary,
                affection_status=affection_status,
                user_profile_data=user_profile_data,
                guild_name=guild_name,
                location_name=location_name,
                model_name=fallback_model_name,  # 关键：使用固定的回退模型
                user_id_for_settings=user_id_for_settings,
                debug_base_url=one_time_debug_base_url,
            )

        # 对于非自定义模型或回退失败后的默认路径
        # 检查是否配置了RAG API密钥
        if self.key_rotation_service is None:
            log.error(
                "未配置 GOOGLE_API_KEYS_LIST，无法使用官方 API。请选择自定义端点模型。"
            )
            return "呜哇，现在有点晕嘞，还没有配置好API密钥呢，请选择自定义端点模型或配置 GOOGLE_API_KEYS_LIST 哦～"

        log.info(
            f"使用模型 '{model_name or self.default_model_name}'，将使用官方 API 逻辑。"
        )
        return await self._generate_with_official_api(
            user_id=user_id,
            guild_id=guild_id,
            message=message,
            channel=channel,
            replied_message=replied_message,
            images=images,
            user_name=user_name,
            channel_context=channel_context,
            world_book_entries=world_book_entries,
            personal_summary=personal_summary,
            affection_status=affection_status,
            user_profile_data=user_profile_data,
            guild_name=guild_name,
            location_name=location_name,
            model_name=model_name,
            user_id_for_settings=user_id_for_settings,
            debug_base_url=one_time_debug_base_url,
        )

    async def _generate_with_custom_endpoint(
        self,
        user_id: int,
        guild_id: int,
        message: str,
        channel: Optional[Any] = None,
        replied_message: Optional[str] = None,
        images: Optional[List[Dict]] = None,
        user_name: str = "用户",
        channel_context: Optional[List[Dict]] = None,
        world_book_entries: Optional[List[Dict]] = None,
        personal_summary: Optional[str] = None,
        affection_status: Optional[Dict[str, Any]] = None,
        user_profile_data: Optional[Dict[str, Any]] = None,
        guild_name: str = "未知服务器",
        location_name: str = "未知位置",
        model_name: Optional[str] = None,
        user_id_for_settings: Optional[str] = None,
        override_base_url: Optional[str] = None,
    ) -> str:
        """
        [新增] 使用自定义端点 (例如公益站) 生成 AI 回复。
        此方法不使用密钥轮换，直接根据配置创建客户端。
        失败时会抛出异常，由调用方 (generate_response) 处理回退逻辑。
        """
        if not model_name:
            raise ValueError("调用自定义端点时需要提供 model_name。")
        endpoint_config = app_config.CUSTOM_GEMINI_ENDPOINTS.get(model_name)
        if not endpoint_config:
            error_msg = (
                f"模型 '{model_name}' 的自定义端点配置未找到。"
                "请检查 CUSTOM_GEMINI_ENDPOINTS 配置。"
            )
            log.error(error_msg)
            raise ValueError(error_msg)

        effective_base_url = override_base_url or endpoint_config.get("base_url")
        effective_api_key = endpoint_config.get("api_key")
        if not effective_base_url or not effective_api_key:
            error_msg = (
                f"模型 '{model_name}' 的自定义端点配置不完整。"
                "请检查 CUSTOM_GEMINI_URL_* 和 CUSTOM_GEMINI_API_KEY_* 环境变量。"
            )
            log.error(error_msg)
            raise ValueError(error_msg)

        # --- 关键：为自定义端点单独创建客户端，不使用全局的 _create_client_with_key ---
        if override_base_url:
            log.info(
                f"🧪 一次性调试已生效：自定义模型 '{model_name}' 临时改用 URL: {effective_base_url}"
            )
        else:
            log.info(f"正在为自定义端点创建客户端: {effective_base_url}")
        http_options = types.HttpOptions(base_url=effective_base_url)
        client = genai.Client(api_key=effective_api_key, http_options=http_options)

        # --- [重构] 针对自定义端点的图片净化 ---
        # 只有在调用自定义端点时才执行此操作，因为官方API可以处理这些图片。
        # 使用顺序处理策略：一张一张处理，处理完一张释放内存，再处理下一张
        sanitized_images_for_endpoint = []
        if images:
            total_images = len(images)
            max_images = app_config.IMAGE_PROCESSING_CONFIG.get(
                "MAX_IMAGES_PER_MESSAGE", 9
            )
            sequential_processing = app_config.IMAGE_PROCESSING_CONFIG.get(
                "SEQUENTIAL_PROCESSING", True
            )

            log.info(f"检测到 {total_images} 张图片，将为自定义端点进行净化处理。")

            # 限制处理的图片数量
            images_to_process = images[:max_images]
            if total_images > max_images:
                log.warning(
                    f"图片数量 ({total_images}) 超过最大限制 ({max_images})，将只处理前 {max_images} 张。"
                )

            for idx, img_data in enumerate(images_to_process, 1):
                # --- [优化] 仅当图片来源是用户附件时才进行净化 ---
                if img_data.get("source") == "attachment":
                    try:
                        # [修复] 增加健壮性，同时检查 'data' 和 'bytes' 键
                        image_bytes = img_data.get("data") or img_data.get("bytes")
                        if not image_bytes:
                            log.warning(
                                f"附件图片数据字典中缺少 'data' 或 'bytes' 键，已跳过。Keys: {list(img_data.keys())}"
                            )
                            continue

                        log.info(f"正在处理第 {idx}/{len(images_to_process)} 张图片...")
                        sanitized_bytes, new_mime_type = sanitize_image(image_bytes)

                        # [内存优化] 处理完成后立即删除原始图片数据引用
                        # 这有助于垃圾回收器及时释放内存
                        if "data" in img_data:
                            del img_data["data"]
                        if "bytes" in img_data:
                            del img_data["bytes"]

                        # [修复] 保持键名一致性，使用 'data' 存储净化后的字节
                        sanitized_images_for_endpoint.append(
                            {
                                "data": sanitized_bytes,
                                "mime_type": new_mime_type,
                                "source": "attachment",  # 来源是确定的
                            }
                        )

                        log.info(f"第 {idx}/{len(images_to_process)} 张图片处理完成。")

                        # [内存优化] 如果启用了顺序处理，在每张图片处理后强制垃圾回收
                        if sequential_processing:
                            import gc

                            gc.collect()

                    except Exception as e:
                        # 如果净化失败，记录错误并通知用户
                        log.error(
                            f"为自定义端点净化第 {idx} 张图片时失败: {e}", exc_info=True
                        )
                        return "呜哇，这张图好像有点问题，我处理不了…可以换一张试试吗？<伤心>"
                else:
                    # 对于非附件图片（如表情），直接使用原始数据
                    sanitized_images_for_endpoint.append(img_data)

        # 复用核心生成逻辑。从此方法抛出的任何异常都将由 generate_response 捕获。
        return await self._execute_generation_cycle(
            user_id=user_id,
            guild_id=guild_id,
            message=message,
            channel=channel,
            replied_message=replied_message,
            images=sanitized_images_for_endpoint
            if images
            else None,  # 如果有图片，则使用净化后的版本
            user_name=user_name,
            channel_context=channel_context,
            world_book_entries=world_book_entries,
            personal_summary=personal_summary,
            affection_status=affection_status,
            user_profile_data=user_profile_data,
            guild_name=guild_name,
            location_name=location_name,
            prompt_model_name=model_name,  # 传递用于选择 Prompt 的原始模型名称
            api_model_name=endpoint_config.get("model_name")
            or self.default_model_name,  # 传递用于调用 API 的真实模型名称
            client=client,
            user_id_for_settings=user_id_for_settings,
        )

    @_api_key_handler
    async def _generate_with_official_api(
        self,
        user_id: int,
        guild_id: int,
        message: str,
        channel: Optional[Any] = None,
        replied_message: Optional[str] = None,
        images: Optional[List[Dict]] = None,
        user_name: str = "用户",
        channel_context: Optional[List[Dict]] = None,
        world_book_entries: Optional[List[Dict]] = None,
        personal_summary: Optional[str] = None,
        affection_status: Optional[Dict[str, Any]] = None,
        user_profile_data: Optional[Dict[str, Any]] = None,
        guild_name: str = "未知服务器",
        location_name: str = "未知位置",
        model_name: Optional[str] = None,
        client: Any = None,
        user_id_for_settings: Optional[str] = None,
    ) -> str:
        """
        [重构] 使用官方 API 密钥池生成 AI 回复。
        此方法由 _api_key_handler 装饰器管理，负责密钥轮换和重试。
        """
        if not client:
            raise ValueError("装饰器未能提供客户端实例。")

        # 将所有生成逻辑委托给核心方法
        return await self._execute_generation_cycle(
            user_id=user_id,
            guild_id=guild_id,
            message=message,
            channel=channel,
            replied_message=replied_message,
            images=images,
            user_name=user_name,
            channel_context=channel_context,
            world_book_entries=world_book_entries,
            personal_summary=personal_summary,
            affection_status=affection_status,
            user_profile_data=user_profile_data,
            guild_name=guild_name,
            location_name=location_name,
            prompt_model_name=model_name,  # 对于官方 API，prompt 和 api 模型名称相同
            api_model_name=model_name,
            client=client,
            user_id_for_settings=user_id_for_settings,
        )

    async def _generate_with_deepseek(
        self,
        user_id: int,
        guild_id: int,
        message: str,
        channel: Optional[Any],
        replied_message: Optional[str],
        images: Optional[List[Dict]],
        user_name: str,
        channel_context: Optional[List[Dict]],
        world_book_entries: Optional[List[Dict]],
        personal_summary: Optional[str],
        affection_status: Optional[Dict[str, Any]],
        user_profile_data: Optional[Dict[str, Any]],
        guild_name: str,
        location_name: str,
        model_name: Optional[str],
        user_id_for_settings: Optional[str] = None,
        override_base_url: Optional[str] = None,
    ) -> str:
        """
        [重构] 包含完整工具调用 (Tool Calling) 循环的 DeepSeek 专用通道。
        自动完成 Gemini 协议到 OpenAI 协议的双向转换。
        已根据最新官方文档更新：移除 Strict Mode，支持 reasoning_content 回传。
        """
        effective_model_name = model_name or "deepseek-chat"
        await chat_settings_service.increment_model_usage(effective_model_name)

        # --- 1. 自动 RAG 检索逻辑 ---
        if not world_book_entries and message:
            try:
                from src.chat.features.world_book.database.world_book_db_manager import world_book_db_manager
                found_entries = await world_book_db_manager.search_entries_in_message(message)
                if found_entries:
                    world_book_entries = found_entries
                    titles = [e.get('title', '未知') for e in found_entries]
                    log.info(f"📚 [DeepSeek] 触发世界书/成员设定，已注入 Prompt: {titles}")
            except Exception as e:
                log.warning(f"[DeepSeek] 世界书自动检索失败: {e}")

        # --- 2. 获取并转换工具 (Python函数对象 -> OpenAI 格式) ---
        dynamic_tools = await self.tool_service.get_dynamic_tools_for_context(
            user_id_for_settings=user_id_for_settings
        )
        openai_tools = []

        # Python 类型注解 -> JSON Schema 类型映射
        _PY_TYPE_MAP = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }
        _STR_TYPE_MAP = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
            "List": "array",
            "Dict": "object",
            "Any": "string",
            "Tuple": "array",
        }
        # 由系统注入、不暴露给模型的参数
        _INTERNAL_PARAMS = {
            "bot", "guild", "channel", "guild_id", "thread_id",
            "log_detailed", "kwargs",
        }

        def _schema_from_annotation(annotation: Any) -> Dict[str, Any]:
            """将 Python 注解转换为 JSON Schema。"""
            if annotation is Any:
                return {"type": "string"}

            origin = get_origin(annotation)
            args = get_args(annotation)

            # Optional[T] / T | None
            if args and any(arg is type(None) for arg in args):
                non_none_args = [arg for arg in args if arg is not type(None)]
                if len(non_none_args) == 1:
                    return _schema_from_annotation(non_none_args[0])

            # List[T] / list[T]
            if origin in (list, List, tuple, Tuple):
                item_annotation = args[0] if args else str
                item_schema = _schema_from_annotation(item_annotation)
                if "type" not in item_schema:
                    item_schema = {"type": "string"}
                return {"type": "array", "items": item_schema}

            # Dict[K, V] / dict
            if origin in (dict, Dict):
                return {"type": "object"}

            # Union[A, B] (非 Optional 的 Union，回退 string)
            if origin is Union:
                return {"type": "string"}

            # 直接类型（int/str/...）
            if isinstance(annotation, type) and annotation in _PY_TYPE_MAP:
                schema_type = _PY_TYPE_MAP[annotation]
                if schema_type == "array":
                    return {"type": "array", "items": {"type": "string"}}
                return {"type": schema_type}

            # 字符串注解兜底
            if isinstance(annotation, str):
                normalized = annotation.replace("typing.", "").strip()
                normalized = normalized.replace("<class '", "").replace("'>", "")
                if normalized.startswith("Optional[") and normalized.endswith("]"):
                    return _schema_from_annotation(normalized[9:-1].strip())
                if normalized.startswith("List[") and normalized.endswith("]"):
                    inner = normalized[5:-1].strip()
                    item_schema = _schema_from_annotation(inner)
                    if "type" not in item_schema:
                        item_schema = {"type": "string"}
                    return {"type": "array", "items": item_schema}
                if normalized.startswith("Dict["):
                    return {"type": "object"}
                mapped = _STR_TYPE_MAP.get(normalized)
                if mapped == "array":
                    return {"type": "array", "items": {"type": "string"}}
                if mapped:
                    return {"type": mapped}

            return {"type": "string"}

        def _is_default_type_compatible(schema_type: str, value: Any) -> bool:
            if schema_type == "string":
                return isinstance(value, str)
            if schema_type == "integer":
                return isinstance(value, int) and not isinstance(value, bool)
            if schema_type == "number":
                return isinstance(value, (int, float)) and not isinstance(value, bool)
            if schema_type == "boolean":
                return isinstance(value, bool)
            if schema_type == "array":
                return isinstance(value, list)
            if schema_type == "object":
                return isinstance(value, dict)
            return True

        def _attach_default_if_compatible(
            prop_schema: Dict[str, Any], default_value: Any, field_path: str
        ) -> None:
            schema_type = prop_schema.get("type")
            if not schema_type:
                return

            if _is_default_type_compatible(schema_type, default_value):
                prop_schema["default"] = default_value
                return

            converted = None
            try:
                if schema_type == "string":
                    converted = str(default_value)
                elif schema_type == "integer" and isinstance(default_value, str):
                    raw = default_value.strip()
                    if re.fullmatch(r"[+-]?\d+", raw):
                        converted = int(raw)
                elif schema_type == "number" and isinstance(default_value, str):
                    converted = float(default_value.strip())
                elif schema_type == "boolean" and isinstance(default_value, str):
                    lowered = default_value.strip().lower()
                    if lowered in {"true", "false"}:
                        converted = lowered == "true"
            except Exception:
                converted = None

            if converted is not None and _is_default_type_compatible(schema_type, converted):
                prop_schema["default"] = converted
                log.warning(
                    f"[DeepSeek Schema] 字段 '{field_path}' 的默认值类型不匹配，已自动修正为 {converted!r}。"
                )
                return

            log.warning(
                f"[DeepSeek Schema] 字段 '{field_path}' 的默认值 {default_value!r} "
                f"与类型 '{schema_type}' 不匹配，已移除 default。"
            )

        if dynamic_tools:
            import inspect as _inspect
            try:
                from pydantic import BaseModel as _BaseModel
            except ImportError:
                _BaseModel = None

            def _pydantic_to_schema(model_cls):
                """将 Pydantic BaseModel 展开为 JSON Schema object。"""
                props = {}
                reqs = []
                for field_name, field_info in model_cls.model_fields.items():
                    ann = field_info.annotation
                    desc = field_info.description or ""
                    prop = _schema_from_annotation(ann)

                    if not field_info.is_required() and field_info.default is not None:
                        _attach_default_if_compatible(
                            prop, field_info.default, f"{model_cls.__name__}.{field_name}"
                        )

                    if desc:
                        prop["description"] = desc

                    props[field_name] = prop

                    if field_info.is_required():
                        reqs.append(field_name)

                schema = {"type": "object", "properties": props}
                if reqs:
                    schema["required"] = reqs
                return schema

            for tool in dynamic_tools:
                try:
                    func_name = tool.__name__
                    func_desc = (tool.__doc__ or "").strip()

                    sig = _inspect.signature(tool)
                    properties = {}
                    required = []

                    for param_name, param in sig.parameters.items():
                        if param_name in _INTERNAL_PARAMS:
                            continue
                        if param.kind in (
                            _inspect.Parameter.VAR_KEYWORD,
                            _inspect.Parameter.VAR_POSITIONAL,
                        ):
                            continue

                        # 检查是否是 Pydantic BaseModel 子类 → 作为嵌套 object 保留键名
                        ann = param.annotation

                        if (
                            _BaseModel is not None
                            and ann != _inspect.Parameter.empty
                            and isinstance(ann, type)
                            and issubclass(ann, _BaseModel)
                        ):
                            # 保留原参数名（如 params），schema 展开为 object
                            sub_schema = _pydantic_to_schema(ann)
                            properties[param_name] = sub_schema

                            if param.default is _inspect.Parameter.empty:
                                required.append(param_name)
                            continue

                        prop_schema = _schema_from_annotation(
                            ann if ann != _inspect.Parameter.empty else Any
                        )

                        # 处理默认值（必须与类型一致，或可安全转换）
                        if (
                            param.default is not _inspect.Parameter.empty
                            and param.default is not None
                            and isinstance(param.default, (str, int, float, bool, list, dict))
                        ):
                            _attach_default_if_compatible(
                                prop_schema, param.default, f"{func_name}.{param_name}"
                            )

                        properties[param_name] = prop_schema

                        if param.default is _inspect.Parameter.empty:
                            required.append(param_name)

                    func_dict = {
                        "name": func_name,
                        "description": func_desc,
                    }

                    # 无参数工具：省略 parameters，避免严格校验器误报
                    if properties:
                        final_params = {"type": "object", "properties": properties}
                        if required:
                            final_params["required"] = required
                        func_dict["parameters"] = final_params

                    openai_tools.append({
                        "type": "function",
                        "function": func_dict
                    })
                    log.debug(f"[DeepSeek] 成功转换工具: {func_name}")

                except Exception as e:
                    log.error(f"[DeepSeek 工具转换失败] 跳过工具 '{getattr(tool, '__name__', tool)}'，错误: {e}", exc_info=True)

            if openai_tools:
                log.info(f"[DeepSeek] 成功转换 {len(openai_tools)} 个工具发往 API。")
            else:
                log.warning("[DeepSeek] 获取到了工具，但转换结果为空！")

        # --- 3. 构建 Prompt 并转换为 OpenAI 消息格式 ---
        final_conversation = await prompt_service.build_chat_prompt(
            user_name=user_name,
            message=message,
            replied_message=replied_message,
            images=images,
            channel_context=channel_context,
            world_book_entries=world_book_entries,
            affection_status=affection_status,
            personal_summary=personal_summary,
            user_profile_data=user_profile_data,
            guild_name=guild_name,
            location_name=location_name,
            model_name=effective_model_name,
            channel=channel,
            user_id=user_id,
        )

        openai_messages = []
        is_first_user = True  # 第一条 user 消息是 system prompt，转为 system role
        for turn in final_conversation:
            gemini_role = turn.get("role")
            content = "".join(
                [part["text"] if isinstance(part, dict) and "text" in part else str(part)
                 for part in turn.get("parts", [])
                 if not (hasattr(part, "thought") and getattr(part, "thought", False))]
            )
            if not content:
                continue

            if gemini_role == "model":
                openai_messages.append({"role": "assistant", "content": content})
            else:
                if is_first_user:
                    # 第一条 user 消息（system prompt）转为 system role
                    openai_messages.append({"role": "system", "content": content})
                    is_first_user = False
                else:
                    openai_messages.append({"role": "user", "content": content})

        # --- 4. 输出完整上下文日志 ---
        log_detailed = app_config.DEBUG_CONFIG.get("LOG_DETAILED_GEMINI_PROCESS", False)
        if log_detailed:
            log.info(f"--- [DeepSeek] 完整发送上下文 (用户 {user_id}) ---")
            log.info(
                json.dumps(
                    openai_messages,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )
            if openai_tools:
                log.info("--- [DeepSeek] 工具列表 ---")
                log.info(
                    json.dumps(openai_tools, ensure_ascii=False, indent=2, default=str)
                )
            log.info("------------------------------------")

        # --- 5. 核心：带工具调用的请求循环 ---
        api_url = (override_base_url or self.deepseek_url or "").rstrip("/")
        if not api_url:
            return "DeepSeek URL 配置缺失，请检查配置。"
        if override_base_url:
            log.info(f"🧪 一次性调试已生效：DeepSeek 临时改用 URL: {api_url}")
        # 不再需要自动修正 beta URL，因为用户已确认使用标准/新版 URL
        if not api_url.endswith("/chat/completions"):
             api_url += "/chat/completions"
             
        gen_config = app_config.MODEL_GENERATION_CONFIG.get(
            effective_model_name, app_config.MODEL_GENERATION_CONFIG["default"]
        )

        max_calls = 5
        called_tool_names = []
        bad_format_retries = 0

        try:
            async with httpx.AsyncClient(timeout=120.0) as http_client:
                for i in range(max_calls):
                    payload = {
                        "model": effective_model_name,
                        "messages": openai_messages,
                        "stream": False,
                        "temperature": gen_config.get("temperature", 1.3),
                        "top_p": gen_config.get("top_p", 0.95),
                        "max_tokens": gen_config.get("max_output_tokens", 8192),
                    }
                    if openai_tools:
                        payload["tools"] = openai_tools

                    response = await http_client.post(
                        api_url,
                        headers={"Authorization": f"Bearer {self.deepseek_key}", "Content-Type": "application/json"},
                        json=payload
                    )
                    response.raise_for_status()
                    result = response.json()

                    response_message = result["choices"][0]["message"]
                    
                    # --- [DeepSeek] 提取并记录思考过程 ---
                    reasoning_content = response_message.get("reasoning_content")
                    content = response_message.get("content") or ""
                    tool_calls = response_message.get("tool_calls")

                    if reasoning_content:
                        log.info(f"--- [DeepSeek] 思考过程 ---\n{reasoning_content}\n-----------------------------")

                    # [关键更新] 将回复加入上下文，必须包含 reasoning_content 以维持思考连续性
                    # 根据最新文档，assistant 消息应包含 content, reasoning_content, tool_calls
                    msg_to_append = {
                        "role": "assistant",
                        "content": content,
                    }
                    if reasoning_content is not None:
                        msg_to_append["reasoning_content"] = reasoning_content
                    if tool_calls is not None:
                        msg_to_append["tool_calls"] = tool_calls
                    
                    openai_messages.append(msg_to_append)

                    # 如果没有工具调用，说明已生成最终回复
                    if not tool_calls:
                        # --- 违禁词检测 ---
                        if content and re.search(r"不过话说回来|话说回来|另外|话又说回来|不过话又说回来", content):
                            if bad_format_retries < 3:
                                log.warning(
                                    f"[DeepSeek] 检测到违禁词 '不过话说回来' (尝试 {bad_format_retries + 1}/3)。正在重试..."
                                )
                                # 这里我们需要把刚才 append 的 assistant 消息暂时移除，或者添加一条 system prompt 要求重试
                                # 为了简单起见，我们添加 user 消息要求重写
                                openai_messages.append({
                                    "role": "user",
                                    "content": "[系统提示] 检测到你使用了“不过话说回来|话说回来|另外|话又说回来”。这是被禁止的。请重新生成回复，去掉这个短语，保持语气自然。"
                                })
                                bad_format_retries += 1
                                continue
                            else:
                                return "抱歉，我的说话格式一直达不到要求，我是杂鱼"
                        # ----------------
                        
                        if log_detailed:
                            log.info("--- [DeepSeek] 模型决策：直接生成文本回复 (未调用工具) ---")
                        self.last_called_tools = called_tool_names
                        log.info("--- [DeepSeek] 文本生成完成 ---")

                        # --- [DeepSeek] 记录 Token 使用情况 ---
                        if "usage" in result:
                            try:
                                usage = result["usage"]
                                input_tokens = usage.get("prompt_tokens", 0)
                                output_tokens = usage.get("completion_tokens", 0)
                                total_tokens = usage.get("total_tokens", 0)
                                
                                usage_date = datetime.now(ZoneInfo("Asia/Shanghai")).date()
                                async with AsyncSessionLocal() as session:
                                    usage_record = await token_usage_service.get_token_usage(session, usage_date)
                                    if usage_record:
                                        await token_usage_service.update_token_usage(
                                            session, usage_record, input_tokens, output_tokens, total_tokens
                                        )
                                    else:
                                        await token_usage_service.create_token_usage(
                                            session, usage_date, input_tokens, output_tokens, total_tokens
                                        )
                                log.info(f"[DeepSeek] Token 记录: In={input_tokens}, Out={output_tokens}, Total={total_tokens}")
                            except Exception as e:
                                log.error(f"[DeepSeek] Token 记录失败: {e}")

                        return await self._post_process_response(content, user_id, guild_id)

                    # 有工具调用
                    if log_detailed:
                        log.info(f"--- [DeepSeek] 模型决策：建议进行工具调用 (第 {i + 1}/{max_calls} 次) ---")
                        for call in tool_calls:
                            try:
                                args_preview = json.loads(call["function"]["arguments"])
                            except Exception:
                                args_preview = {}
                            log.info(f"  - 工具名称: {call['function']['name']}")
                            args_str_preview = json.dumps(args_preview, ensure_ascii=False, indent=2)
                            log.info("  - 调用参数:\n" + args_str_preview)
                        log.info("------------------------------------")

                    for call in tool_calls:
                        tool_name = call["function"]["name"]
                        called_tool_names.append(tool_name)

                        try:
                            args = json.loads(call["function"]["arguments"])
                        except Exception:
                            args = {}

                        log.info(f"  - 准备执行工具: {tool_name}, 参数: {args}")

                        # 【关键桥接】伪造 Gemini 的 FunctionCall 对象，交给 ToolService 执行
                        mock_gemini_call = types.FunctionCall(name=tool_name, args=args)

                        tool_res = await self.tool_service.execute_tool_call(
                            tool_call=mock_gemini_call,
                            channel=channel,
                            user_id=user_id,
                            log_detailed=log_detailed,
                            user_id_for_settings=user_id_for_settings,
                        )

                        # 解析工具执行的结果并转换为 OpenAI 格式
                        if isinstance(tool_res, types.Part) and tool_res.function_response:
                            raw_response = tool_res.function_response.response
                            # DeepSeek 是纯文本 API，无法处理图片数据
                            # 深拷贝后剔除所有 base64 图片字段，避免请求体过大触发 400
                            import copy as _copy
                            clean_response = _copy.deepcopy(raw_response)
                            result_data = clean_response.get("result", {})
                            if isinstance(result_data, dict):
                                profile = result_data.get("profile", {})
                                if isinstance(profile, dict) and "avatar_image_base64" in profile:
                                    del profile["avatar_image_base64"]
                                    profile["avatar_note"] = "（头像图片数据已省略，仅文字通道可用）"
                            content_str = json.dumps(clean_response, ensure_ascii=False)
                        else:
                            content_str = str(tool_res)

                        openai_messages.append({
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": tool_name,
                            "content": content_str,
                        })

                    # 工具执行完毕，继续下一轮让 DeepSeek 总结结果
                    continue

                # 如果超出了最大调用次数
                self.last_called_tools = called_tool_names
                return "哎呀，我好像陷入了一个复杂的思考循环里，换个话题聊聊吧！"

        except Exception as e:
            error_info = f"{type(e).__name__}: {str(e)}"
            log.error(f"DeepSeek API 调用失败: {error_info}", exc_info=True)
            # 返回具体错误信息给用户，方便调试
            return f"DeepSeek 连接失败: {error_info}。请检查日志或配置。"

    async def _execute_generation_cycle(
        self,
        user_id: int,
        guild_id: int,
        message: str,
        channel: Optional[Any],
        replied_message: Optional[str],
        images: Optional[List[Dict]],
        user_name: str,
        channel_context: Optional[List[Dict]],
        world_book_entries: Optional[List[Dict]],
        personal_summary: Optional[str],
        affection_status: Optional[Dict[str, Any]],
        user_profile_data: Optional[Dict[str, Any]],
        guild_name: str,
        location_name: str,
        prompt_model_name: Optional[str],
        api_model_name: Optional[str],
        client: Any,
        user_id_for_settings: Optional[str] = None,
    ) -> str:
        """
        [新增] 核心的 AI 生成周期，包含上下文构建、工具调用循环和响应处理。
        此方法被 _generate_with_official_api 和 _generate_with_custom_endpoint 复用。
        """
        # 自动 RAG 检索逻辑：如果调用方没传条目，且有用户消息，我们就自己去数据库搜！
        if not world_book_entries and message:
            try:
                # 延迟导入，防止循环引用
                from src.chat.features.world_book.database.world_book_db_manager import world_book_db_manager
                
                #拿着用户的消息去数据库里搜关键词
                found_entries = await world_book_db_manager.search_entries_in_message(message)
                
                if found_entries:
                    # 如果搜到了，就强制替换掉原本为空的 entries
                    world_book_entries = found_entries
                    titles = [e.get('title', '未知') for e in found_entries]
                    log.info(f"📚 触发世界书/成员设定，已注入 Prompt: {titles}")
            except Exception as e:
                log.warning(f"世界书自动检索失败 (不影响正常对话): {e}")
        # --- 模型使用计数 ---
        # 使用 prompt_model_name (表面模型名) 进行计数，而不是 api_model_name (真实模型名)
        model_to_count = prompt_model_name or self.default_model_name
        await chat_settings_service.increment_model_usage(model_to_count)

        # 1. 构建完整的对话提示（现在是异步方法）
        final_conversation = await prompt_service.build_chat_prompt(
            user_name=user_name,
            message=message,
            replied_message=replied_message,
            images=images,
            channel_context=channel_context,
            world_book_entries=world_book_entries,
            affection_status=affection_status,
            personal_summary=personal_summary,
            user_profile_data=user_profile_data,
            guild_name=guild_name,
            location_name=location_name,
            model_name=prompt_model_name,
            channel=channel,  # 传递 channel 对象
            user_id=user_id,
        )

        # 3. 准备 API 调用参数 (重构)
        model_key = prompt_model_name or "default"
        gen_config_data = app_config.MODEL_GENERATION_CONFIG.get(
            model_key, app_config.MODEL_GENERATION_CONFIG["default"]
        ).copy()

        log.info(f"正在为模型 '{model_key}' 加载生成配置。")

        # 从配置中提取 thinking_config，剩下的作为 generation_config 的参数
        thinking_config_data = gen_config_data.pop("thinking_config", None)
        gen_config_params = {**gen_config_data, "safety_settings": self.safety_settings}

        # --- [新增] 动态开启 Google 搜索和 URL 上下文工具 ---
        # 1. 初始化一个工具配置列表
        enabled_tools = []

        # 2. 添加 Google 搜索和 URL 阅读工具 (暂时禁用以恢复功能)
        # enabled_tools.append(types.Tool(google_search=types.GoogleSearch()))
        # enabled_tools.append(types.Tool(url_context=types.UrlContext()))
        # log.info("已为本次调用启用 Google 搜索工具。")

        # 3. 根据上下文动态获取函数工具
        dynamic_tools = await self.tool_service.get_dynamic_tools_for_context(
            user_id_for_settings=user_id_for_settings
        )
        if dynamic_tools:
            enabled_tools.extend(dynamic_tools)
            log.info(f"已根据上下文合并 {len(dynamic_tools)} 个动态函数工具。")

        # 4. 如果最终有工具被启用，则配置到生成参数中
        if enabled_tools:
            gen_config_params["tools"] = enabled_tools
            # 保持手动调用模式，让我们可以控制工具的执行流程
            gen_config_params["automatic_function_calling"] = (
                types.AutomaticFunctionCallingConfig(disable=True)
            )
            log.info("已启用手动工具调用模式，并集成了原生搜索及自定义函数。")

        gen_config = types.GenerateContentConfig(**gen_config_params)

        # 根据提取的 thinking_config_data 动态构建 ThinkingConfig
        if thinking_config_data:
            gen_config.thinking_config = types.ThinkingConfig(**thinking_config_data)
            log.info(
                f"已为模型 '{model_key}' 启用思维链 (Thinking)，配置: {thinking_config_data}"
            )

        # 4. 准备初始对话历史
        conversation_history = self._prepare_api_contents(final_conversation)

        if app_config.DEBUG_CONFIG["LOG_AI_FULL_CONTEXT"]:
            log.info(f"--- 初始 AI 上下文 (用户 {user_id}) ---")
            log.info(
                json.dumps(
                    [
                        self._serialize_parts_for_logging_full(c)
                        for c in conversation_history
                    ],
                    ensure_ascii=False,
                    indent=2,
                )
            )
            log.info("------------------------------------")

        # 5. 实现手动、顺序工具调用循环
        called_tool_names = []
        thinking_was_used = False
        max_calls = 5
        for i in range(max_calls):
            log_detailed = app_config.DEBUG_CONFIG.get(
                "LOG_DETAILED_GEMINI_PROCESS", False
            )
            if log_detailed:
                log.info(f"--- [工具调用循环: 第 {i + 1}/{max_calls} 次] ---")

            response = None
            for attempt in range(2):
                response = await client.aio.models.generate_content(
                    model=(api_model_name or self.default_model_name),
                    contents=conversation_history,
                    config=gen_config,
                )
                if response and (
                    (response.candidates and response.candidates[0].content)
                    or (hasattr(response, "function_calls") and response.function_calls)
                ):
                    break
                log.warning(f"模型返回空响应 (尝试 {attempt + 1}/2)。将在1秒后重试...")
                if attempt < 1:
                    await asyncio.sleep(1)

            if log_detailed:
                if response and response.candidates:
                    candidate = response.candidates[0]
                    if candidate and candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, "thought") and part.thought:
                                thinking_was_used = True
                                log.info("--- 模型思考过程 (Thinking) ---")
                                log.info(part.text)
                                log.info("---------------------------------")

            function_calls = (
                response.function_calls
                if response and hasattr(response, "function_calls")
                else None
            )

            if not function_calls:
                if log_detailed:
                    log.info("--- 模型决策：直接生成文本回复 (未调用工具) ---")
                    log.info("模型返回了最终文本响应，工具调用流程结束。")
                break

            if log_detailed:
                log.info("--- 模型决策：建议进行工具调用 ---")
                for call in function_calls:
                    args_str = json.dumps(dict(call.args), ensure_ascii=False, indent=2)
                    log.info(f"  - 工具名称: {call.name}")
                    log.info(f"  - 调用参数:\n{args_str}")
                log.info("------------------------------------")

            for call in function_calls:
                called_tool_names.append(call.name)

            if (
                response
                and response.candidates
                and response.candidates[0].content
                and response.candidates[0].content.parts
            ):
                conversation_history.append(response.candidates[0].content)

            if log_detailed:
                log.info(f"准备执行 {len(function_calls)} 个工具调用...")

            tool_result_parts = []
            tasks = [
                self.tool_service.execute_tool_call(
                    tool_call=call,
                    channel=channel,
                    user_id=user_id,
                    log_detailed=log_detailed,
                    user_id_for_settings=user_id_for_settings,
                )
                for call in function_calls
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    log.error(f"执行工具时发生异常: {result}", exc_info=result)
                    tool_result_parts.append(
                        types.Part.from_function_response(
                            name="unknown_tool",
                            response={
                                "error": f"An exception occurred during tool execution: {str(result)}"
                            },
                        )
                    )
                # 确保 result 是 Part 类型，并且其 function_response 和 response 属性都存在
                elif (
                    isinstance(result, types.Part)
                    and result.function_response
                    and result.function_response.response
                ):
                    tool_name = result.function_response.name
                    # 现在这里是绝对安全的
                    # 首先检查是否有错误信息
                    error_message = result.function_response.response.get("error")
                    if error_message:
                        # 如果有错误信息，直接使用错误信息作为结果
                        original_result = error_message
                        log.info(f"工具返回错误信息: {error_message}")
                    else:
                        original_result = result.function_response.response.get(
                            "result", {}
                        )

                    # --- 新增：处理工具返回的头像图片 ---
                    if isinstance(original_result, dict):
                        profile = original_result.get("profile", {})
                        if "avatar_image_base64" in profile:
                            log.info(
                                "检测到工具返回的 avatar_image_base64，正在处理为图片 Part。"
                            )
                            try:
                                image_bytes = base64.b64decode(
                                    profile["avatar_image_base64"]
                                )
                                # 创建一个新的图片 Part
                                image_part = types.Part(
                                    inline_data=types.Blob(
                                        mime_type="image/png", data=image_bytes
                                    )
                                )
                                tool_result_parts.append(image_part)
                                # 从原始结果中移除，避免冗余
                                del profile["avatar_image_base64"]
                            except Exception as e:
                                log.error(
                                    f"处理 avatar_image_base64 时出错: {e}",
                                    exc_info=True,
                                )
                    # --- 图片处理结束 ---

                    response_content: Dict[str, Any]

                    if isinstance(original_result, (dict, list)):
                        response_content = {"result": original_result}
                    else:
                        safe_tool_name = tool_name or "unknown_tool"
                        safe_result_str = str(original_result or "")

                        wrapped_result_str = (
                            prompt_service.build_tool_result_wrapper_prompt(
                                safe_tool_name,
                                safe_result_str,
                            )
                        )
                        response_content = {"result": wrapped_result_str}

                    # 只有在 response_content['result'] 真正有内容时才创建 FunctionResponse Part
                    # 这样可以避免在只有图片的情况下，发送一个空的、无意义的文本结果 Part
                    if response_content.get("result"):
                        new_response_part = types.Part.from_function_response(
                            name=tool_name or "unknown_tool",
                            response=response_content,
                        )
                        tool_result_parts.append(new_response_part)

                else:
                    log.warning(f"接收到未知的工具执行结果类型: {type(result)}")

            if log_detailed:
                log.info(
                    f"已收集 {len(tool_result_parts)} 个工具执行结果，准备将其返回给模型。"
                )
                try:
                    # --- [新增调试日志] ---
                    # 序列化并打印工具返回的详细内容
                    results_for_log = []
                    for part in tool_result_parts:
                        if part and part.function_response:
                            results_for_log.append(
                                {
                                    "name": part.function_response.name,
                                    "response": self._serialize_for_logging(
                                        part.function_response.response
                                    ),
                                }
                            )
                    log.info(
                        f"--- [工具结果详细内容] ---\n{json.dumps(results_for_log, ensure_ascii=False, indent=2)}"
                    )
                    # --- [日志结束] ---
                except Exception as e:
                    log.error(f"序列化工具结果用于日志记录时出错: {e}")

            conversation_history.append(
                types.Content(role="tool", parts=tool_result_parts)
            )

            if i == max_calls - 1:
                log.warning("已达到最大工具调用限制，流程终止。")
                self.last_called_tools = called_tool_names
                return "哎呀，我好像陷入了一个复杂的思考循环里，我们换个话题聊聊吧！"

        if response and response.parts:
            final_thought = ""
            final_text = ""
            for part in response.parts or []:
                if hasattr(part, "thought") and part.thought:
                    thinking_was_used = True
                    final_thought += part.text
                elif hasattr(part, "text"):
                    final_text += part.text

            if log_detailed:
                if final_thought:
                    log.info("--- 模型最终回复的思考过程 ---")
                    log.info(final_thought.strip())
                    log.info("-----------------------------")
                else:
                    log.info("--- 模型最终回复未提供明确的思考过程。---")

            raw_ai_response = final_text.strip()

            if raw_ai_response:
                from src.chat.services.context_service import context_service

                await context_service.update_user_conversation_history(
                    user_id, guild_id, message if message else "", raw_ai_response
                )
                formatted_response = await self._post_process_response(
                    raw_ai_response, user_id, guild_id
                )
                # --- 新增：记录 Token 使用情况 ---
                await self._record_token_usage(
                    client=client,
                    model_name=api_model_name or self.default_model_name,
                    input_contents=conversation_history,
                    output_text=raw_ai_response,
                )
                total_tokens = 0
                if response and response.usage_metadata:
                    total_tokens = response.usage_metadata.total_token_count

                log.info("--- Gemini API 请求摘要 ---")
                log.info(
                    f"  - 本次调用是否使用思考功能: {'是' if thinking_was_used else '否'}"
                )
                if thinking_was_used:
                    log.info(f"  - 思考过程Token消耗: {total_tokens}")

                if called_tool_names:
                    unique_tools = sorted(list(set(called_tool_names)))
                    log.info(
                        f"  - 调用了 {len(unique_tools)} 个工具: {', '.join(unique_tools)}"
                    )
                else:
                    log.info("  - 未调用任何工具。")
                log.info("--------------------------")

                self.last_called_tools = called_tool_names
                return formatted_response

        self.last_called_tools = called_tool_names
        if (
            response
            and response.prompt_feedback
            and response.prompt_feedback.block_reason
        ):
            try:
                conversation_for_log = json.dumps(
                    GeminiService._serialize_for_logging(final_conversation),
                    ensure_ascii=False,
                    indent=2,
                )
                full_response_for_log = str(response)
                log.warning(
                    f"用户 {user_id} 的请求被安全策略阻止，原因: {response.prompt_feedback.block_reason if response.prompt_feedback else '未知'}\n"
                    f"--- 完整的对话历史 ---\n{conversation_for_log}\n"
                    f"--- 完整的 API 响应 ---\n{full_response_for_log}"
                )
            except Exception as log_e:
                log.error(f"序列化被阻止的请求用于日志记录时出错: {log_e}")
                log.warning(
                    f"用户 {user_id} 的请求被安全策略阻止，原因: {response.prompt_feedback.block_reason if response.prompt_feedback else '未知'} (详细内容记录失败)"
                )
            return "呜啊! 这个太色情啦,我不看我不看"
        else:
            log.warning(f"未能为用户 {user_id} 生成有效回复。")
            return "哎呀，我好像没太明白你的意思呢～可以再说清楚一点吗？✨"
        return "哎呀，我好像没太明白你的意思呢～可以再说清楚一点吗？✨"

    @_api_key_handler
    async def generate_embedding(
        self,
        text: str,
        task_type: str = "retrieval_document",
        title: Optional[str] = None,
        client: Any = None,
    ) -> Optional[List[float]]:
        """
        为给定文本生成嵌入向量。
        """
        if not client:
            raise ValueError("装饰器未能提供客户端实例。")

        if not text or not text.strip():
            log.warning(
                f"generate_embedding 接收到空文本！text: '{text}', task_type: '{task_type}'"
            )
            return None

        loop = asyncio.get_event_loop()
        embed_config = types.EmbedContentConfig(task_type=task_type)
        if title and task_type == "retrieval_document":
            embed_config.title = title

        embedding_result = await loop.run_in_executor(
            self.executor,
            lambda: client.models.embed_content(
                model="gemini-embedding-001",
                contents=[types.Part(text=text)],
                config=embed_config,
            ),
        )

        if embedding_result and embedding_result.embeddings:
            return embedding_result.embeddings[0].values
        return None

    @_api_key_handler
    async def generate_text(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        model_name: Optional[str] = None,
        client: Any = None,
    ) -> Optional[str]:
        """
        一个用于简单文本生成的精简方法。
        不涉及对话历史或上下文，仅根据输入提示生成文本。
        非常适合用于如“查询重写”等内部任务。

        Args:
            prompt: 提供给模型的输入提示。
            temperature: 控制生成文本的随机性。如果为 None，则使用 config 中的默认值。
            model_name: 指定要使用的模型。如果为 None，则使用默认的聊天模型。

        Returns:
            生成的文本字符串，如果失败则返回 None。
        """
        if not client:
            raise ValueError("装饰器未能提供客户端实例。")

        loop = asyncio.get_event_loop()
        gen_config_params = app_config.GEMINI_TEXT_GEN_CONFIG.copy()
        if temperature is not None:
            gen_config_params["temperature"] = temperature
        gen_config = types.GenerateContentConfig(
            **gen_config_params, safety_settings=self.safety_settings
        )
        final_model_name = model_name or self.default_model_name

        response = await loop.run_in_executor(
            self.executor,
            lambda: client.models.generate_content(
                model=final_model_name, contents=[prompt], config=gen_config
            ),
        )

        if response.parts:
            return response.text.strip()
        return None

    @_api_key_handler
    async def generate_simple_response(
        self,
        prompt: str,
        generation_config: Dict,
        model_name: Optional[str] = None,
        client: Any = None,
    ) -> Optional[str]:
        """
        一个用于单次、非对话式文本生成的方法，允许传入完整的生成配置和可选的模型名称。
        非常适合用于如“礼物回应”、“投喂”等需要自定义生成参数的一次性任务。

        Args:
            prompt: 提供给模型的完整输入提示。
            generation_config: 一个包含生成参数的字典 (e.g., temperature, max_output_tokens).
            model_name: (可选) 指定要使用的模型。如果为 None，则使用默认的聊天模型。

        Returns:
            生成的文本字符串，如果失败则返回 None。
        """
        if not client:
            raise ValueError("装饰器未能提供客户端实例。")

        loop = asyncio.get_event_loop()
        gen_config = types.GenerateContentConfig(
            **generation_config, safety_settings=self.safety_settings
        )
        final_model_name = model_name or self.default_model_name

        response = await loop.run_in_executor(
            self.executor,
            lambda: client.models.generate_content(
                model=final_model_name, contents=[prompt], config=gen_config
            ),
        )

        if response.parts:
            return response.text.strip()

        log.warning(f"generate_simple_response 未能生成有效内容。API 响应: {response}")
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            log.warning(
                f"请求可能被安全策略阻止，原因: {response.prompt_feedback.block_reason}"
            )

        return None

    @_api_key_handler
    async def generate_thread_praise(
        self, conversation_history: List[Dict[str, Any]], client: Any = None
    ) -> Optional[str]:
        """
        专用于生成帖子夸奖的方法。
        现在接收一个由 prompt_service 构建好的完整对话历史。

        Args:
            conversation_history: 完整的对话历史列表。
            client: (由装饰器注入) Gemini 客户端。

        Returns:
            生成的夸奖文本，如果失败则返回 None。
        """
        if not client:
            raise ValueError("装饰器未能提供客户端实例。")

        loop = asyncio.get_event_loop()
        # --- (新增) 为暖贴功能启用思考 ---
        praise_config = app_config.GEMINI_THREAD_PRAISE_CONFIG.copy()
        thinking_budget = praise_config.pop("thinking_budget", None)

        gen_config = types.GenerateContentConfig(
            **praise_config,
            safety_settings=self.safety_settings,
        )

        if thinking_budget is not None:
            gen_config.thinking_config = types.ThinkingConfig(
                include_thoughts=True, thinking_budget=thinking_budget
            )
            log.info(f"已为暖贴功能启用思维链 (Thinking)，预算: {thinking_budget}。")

        final_model_name = self.default_model_name

        final_contents = self._prepare_api_contents(conversation_history)

        # 如果开启了 AI 完整上下文日志，则打印到终端
        if app_config.DEBUG_CONFIG["LOG_AI_FULL_CONTEXT"]:
            log.info("--- 暖贴功能 · 完整 AI 上下文 ---")
            log.info(
                json.dumps(
                    [
                        self._serialize_parts_for_logging_full(content)
                        for content in final_contents
                    ],
                    ensure_ascii=False,
                    indent=2,
                )
            )
            log.info("------------------------------------")

        response = await loop.run_in_executor(
            self.executor,
            lambda: client.models.generate_content(
                model=final_model_name, contents=final_contents, config=gen_config
            ),
        )

        if response.parts:
            # --- (修正) 采用与主对话相同的逻辑，正确分离思考过程和最终回复 ---
            final_text = ""
            for part in response.parts:
                # 关键：只有当 part 不是思考过程时，才将其文本内容计入最终回复
                if hasattr(part, "thought") and part.thought:
                    # 这是思考过程，忽略它
                    pass
                elif hasattr(part, "text"):
                    # 这是最终回复
                    final_text += part.text
            return final_text.strip()

        log.warning(f"generate_thread_praise 未能生成有效内容。API 响应: {response}")
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            log.warning(
                f"请求可能被安全策略阻止，原因: {response.prompt_feedback.block_reason}"
            )

        return None

    async def summarize_for_rag(
        self,
        latest_query: str,
        user_name: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        根据用户的最新发言和可选的对话历史，生成一个用于RAG搜索的独立查询。

        Args:
            latest_query: 用户当前发送的最新消息。
            user_name: 提问用户的名字。
            conversation_history: (可选) 包含多轮对话的列表。

        Returns:
            一个精炼后的、适合向量检索的查询字符串。
        """
        if not latest_query:
            log.info("RAG summarization called with no latest_query.")
            return ""

        prompt = prompt_service.build_rag_summary_prompt(
            latest_query, user_name, conversation_history
        )
        summarized_query = await self.generate_text(
            prompt, temperature=0.0, model_name=app_config.QUERY_REWRITING_MODEL
        )

        if not summarized_query:
            log.info("RAG查询总结失败，将直接使用用户的原始查询。")
            return latest_query.strip()

        return summarized_query.strip().strip('"')

    async def clear_user_context(self, user_id: int, guild_id: int):
        """清除指定用户的对话上下文"""
        await chat_db_manager.clear_ai_conversation_context(user_id, guild_id)
        log.info(f"已清除用户 {user_id} 在服务器 {guild_id} 的对话上下文")

    def is_available(self) -> bool:
        """检查AI服务是否可用"""
        return self.key_rotation_service is not None

    @_api_key_handler
    async def generate_text_with_image(
        self, prompt: str, image_bytes: bytes, mime_type: str, client: Any = None
    ) -> Optional[str]:
        """
        一个用于简单图文生成的精简方法。
        不涉及对话历史或上下文，仅根据输入提示和图片生成文本。
        非常适合用于如“投喂”等一次性功能。

        Args:
            prompt: 提供给模型的输入提示。
            image_bytes: 图片的字节数据。
            mime_type: 图片的 MIME 类型 (e.g., 'image/jpeg', 'image/png').

        Returns:
            生成的文本字符串，如果失败则返回 None。
        """
        if not client:
            raise ValueError("装饰器未能提供客户端实例。")

        # --- 新增：处理 GIF 图片 ---
        if mime_type == "image/gif":
            try:
                log.info("检测到 GIF 图片，尝试提取第一帧...")
                with Image.open(io.BytesIO(image_bytes)) as img:
                    # 寻求第一帧并转换为 RGBA 以确保兼容性
                    img.seek(0)
                    # 创建一个新的 BytesIO 对象来保存转换后的图片
                    output_buffer = io.BytesIO()
                    # 将图片保存为 PNG 格式
                    img.save(output_buffer, format="PNG")
                    # 获取转换后的字节数据
                    image_bytes = output_buffer.getvalue()
                    # 更新 MIME 类型
                    mime_type = "image/png"
                    log.info("成功将 GIF 第一帧转换为 PNG。")
            except Exception as e:
                log.error(f"处理 GIF 图片时出错: {e}", exc_info=True)
                return "呜哇，我的眼睛跟不上啦！有点看花眼了"
        # --- GIF 处理结束 ---

        # --- 新增：检查并压缩图片 ---
        try:
            image_bytes, mime_type = sanitize_image(image_bytes)
        except Exception as e:
            log.error(f"压缩图片时出错: {e}")

        request_contents = [
            prompt,
            types.Part(inline_data=types.Blob(mime_type=mime_type, data=image_bytes)),
        ]
        gen_config = types.GenerateContentConfig(
            **app_config.GEMINI_VISION_GEN_CONFIG, safety_settings=self.safety_settings
        )

        response = await client.aio.models.generate_content(
            model=self.default_model_name, contents=request_contents, config=gen_config
        )

        if response.parts:
            return response.text.strip()
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
            log.warning(
                f"图文生成请求被安全策略阻止: {response.prompt_feedback.block_reason}"
            )
            return "为啥要投喂色图啊喂"

        log.warning(f"未能为图文生成有效回复。Response: {response}")
        return "我好像没看懂这张图里是什么，可以换一张或者稍后再试试吗？"

    @_api_key_handler
    async def generate_confession_response(
        self, prompt: str, client: Any = None
    ) -> Optional[str]:
        """
        专用于生成忏悔回应的方法。
        """
        if not client:
            raise ValueError("装饰器未能提供客户端实例。")

        gen_config = types.GenerateContentConfig(
            **app_config.GEMINI_CONFESSION_GEN_CONFIG,
            safety_settings=self.safety_settings,
        )
        final_model_name = self.default_model_name

        if app_config.DEBUG_CONFIG["LOG_AI_FULL_CONTEXT"]:
            log.info("--- 忏悔功能 · 完整 AI 上下文 ---")
            log.info(prompt)
            log.info("------------------------------------")

        response = await client.aio.models.generate_content(
            model=final_model_name, contents=[prompt], config=gen_config
        )

        if response.parts:
            return response.text.strip()

        log.warning(
            f"generate_confession_response 未能生成有效内容。API 响应: {response}"
        )
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            log.warning(
                f"请求可能被安全策略阻止，原因: {response.prompt_feedback.block_reason}"
            )

        return None

    async def _record_token_usage(
        self,
        client: Any,  # 使用 Any 来避免 Pylance 对动态 client 的错误
        model_name: str,
        input_contents: List[types.Content],
        output_text: str,
    ):
        """记录 API 调用的 Token 使用情况到数据库。"""
        try:
            # 分别计算输入和输出的 Token
            # 使用 type: ignore 来抑制 Pylance 无法推断 client.aio.models 的错误
            input_token_response = await client.aio.models.count_tokens(  # type: ignore
                model=model_name, contents=input_contents
            )
            output_token_response = await client.aio.models.count_tokens(  # type: ignore
                model=model_name, contents=[output_text]
            )

            input_tokens = input_token_response.total_tokens
            output_tokens = output_token_response.total_tokens
            total_tokens = input_tokens + output_tokens

            # 获取当前日期并更新数据库
            usage_date = datetime.now(ZoneInfo("Asia/Shanghai")).date()
            async with AsyncSessionLocal() as session:
                usage_record = await token_usage_service.get_token_usage(
                    session, usage_date
                )
                if usage_record:
                    await token_usage_service.update_token_usage(
                        session,
                        usage_record,
                        input_tokens,
                        output_tokens,
                        total_tokens,
                    )
                else:
                    await token_usage_service.create_token_usage(
                        session,
                        usage_date,
                        input_tokens,
                        output_tokens,
                        total_tokens,
                    )
            log.info(
                f"Token usage recorded: Input={input_tokens}, Output={output_tokens}, Total={total_tokens}"
            )
        except Exception as e:
            log.error(f"Failed to record token usage: {e}", exc_info=True)


# 全局实例
gemini_service = GeminiService()
