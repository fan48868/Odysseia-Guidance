# -*- coding: utf-8 -*-

import asyncio
import base64
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union, get_args, get_origin
from zoneinfo import ZoneInfo

import httpx
from google.genai import types

from src.chat.config import chat_config as app_config
from src.chat.features.chat_settings.services.chat_settings_service import chat_settings_service
from src.chat.features.tools.services.tool_service import ToolService
from src.chat.services.kimi_key_rotation import NoAvailableKimiKeyError
from src.chat.services.openai_models import (
    CustomModelClient,
    DeepSeekModelClient,
    KimiModelClient,
)
from src.chat.services.prompt_service import prompt_service
from src.chat.utils.image_utils import sanitize_image_to_size_limit
from src.database.database import AsyncSessionLocal
from src.database.services.token_usage_service import token_usage_service

log = logging.getLogger(__name__)


class OpenAIService:
    """
    OpenAI 兼容通道服务（DeepSeek / Kimi / Custom）。
    说明：
    - 本服务专门承接 OpenAI 协议调用链（messages/tools/tool_calls）。
    - 通过注入 ToolService 复用现有工具体系，避免重复加载工具。
    - 通过注入 post_process_response 回调复用统一后处理逻辑。
    """

    def __init__(
        self,
        tool_service: ToolService,
        post_process_response: Callable[[str, int, int], Awaitable[str]],
    ):
        self.tool_service = tool_service
        self.post_process_response = post_process_response
        self.last_called_tools: List[str] = []

        # OpenAI 模型总管理：模型细节下沉至独立客户端
        self.deepseek_model_client = DeepSeekModelClient()
        self.custom_model_client = CustomModelClient()
        self.kimi_model_client = KimiModelClient(
            bot_getter=lambda: getattr(self.tool_service, "bot", None),
            alert_user_id=1046310552365973524,
        )

    @staticmethod
    def _normalize_bool_flag(raw_value: Optional[str]) -> bool:
        return str(raw_value or "").strip().lower() in {"true", "1", "yes"}

    def _is_vps_mode_enabled(self) -> bool:
        return self._normalize_bool_flag(os.environ.get("VPS_MODE"))

    def _compress_images_for_vps_mode(
        self, images: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        if not images or not self._is_vps_mode_enabled():
            return images

        target_size_bytes = 1 * 1024 * 1024
        processed_images: List[Dict[str, Any]] = []

        for idx, image in enumerate(images, start=1):
            if not isinstance(image, dict):
                processed_images.append(image)
                continue

            image_bytes = image.get("data") or image.get("bytes")
            if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
                processed_images.append(dict(image))
                continue

            image_copy = dict(image)
            original_size = len(image_bytes)

            try:
                compressed_bytes, compressed_mime_type = sanitize_image_to_size_limit(
                    bytes(image_bytes),
                    target_size_bytes=target_size_bytes,
                    max_image_size_bytes=target_size_bytes,
                    min_quality=20,
                )
                image_copy["data"] = compressed_bytes
                if "bytes" in image_copy:
                    image_copy["bytes"] = compressed_bytes
                image_copy["mime_type"] = compressed_mime_type
                log.info(
                    "[OpenAIService] VPS mode image compression applied | index=%s | original_kb=%.2f | compressed_kb=%.2f | mime_type=%s",
                    idx,
                    original_size / 1024,
                    len(compressed_bytes) / 1024,
                    compressed_mime_type,
                )
            except Exception as e:
                if original_size > target_size_bytes:
                    log.warning(
                        "[OpenAIService] VPS mode image compression failed, dropping oversized image | index=%s | size_kb=%.2f | error=%s",
                        idx,
                        original_size / 1024,
                        e,
                    )
                    continue
                log.warning(
                    "[OpenAIService] VPS mode image compression failed, using original image | index=%s | size_kb=%.2f | error=%s",
                    idx,
                    original_size / 1024,
                    e,
                )

            processed_images.append(image_copy)

        return processed_images

    def _build_tool_image_context_list(
        self, images: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, str]]:
        """
        将当前轮可用图片构建为工具可注入的标准上下文列表。
        仅在服务层注入，不暴露给模型参数 schema。
        """
        if not images:
            return []

        max_images = app_config.IMAGE_PROCESSING_CONFIG.get("MAX_IMAGES_PER_MESSAGE", 9)
        image_context_list: List[Dict[str, str]] = []

        for idx, img in enumerate(images[:max_images], start=1):
            if not isinstance(img, dict):
                continue

            image_bytes = img.get("data") or img.get("bytes")
            if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
                continue

            mime_type = str(img.get("mime_type", "image/png"))
            source = str(img.get("source", "unknown"))

            try:
                image_b64 = base64.b64encode(bytes(image_bytes)).decode("utf-8")
            except Exception as e:
                log.warning(f"构建图片上下文时 Base64 编码失败，已跳过第 {idx} 张: {e}")
                continue

            image_context_list.append(
                {
                    "index": str(idx),
                    "mime_type": mime_type,
                    "source": source,
                    "image_base64": image_b64,
                }
            )

        return image_context_list

    @staticmethod
    def _build_request_error_log_fields(e: httpx.RequestError) -> Dict[str, str]:
        req = getattr(e, "request", None)
        cause = getattr(e, "__cause__", None)

        return {
            "exc_type": type(e).__name__,
            "exc_repr": repr(e),
            "exc_str": str(e) or "<empty>",
            "cause_type": type(cause).__name__ if cause else "<none>",
            "cause_repr": repr(cause) if cause else "<none>",
            "request_method": getattr(req, "method", "<none>") if req else "<none>",
            "request_url": str(getattr(req, "url", "<none>")) if req else "<none>",
        }

    def _log_custom_stream_parse_failure(
        self,
        *,
        log_prefix: str,
        response: httpx.Response,
        used_api_url: str,
        response_text: str,
    ) -> None:
        reason, diagnostics = (
            self.custom_model_client.explain_streaming_chat_completion_parse_failure(
                response_text
            )
        )
        content_type = response.headers.get("content-type", "<none>")
        server = response.headers.get("server", "<none>")
        response_url = used_api_url or str(getattr(response.request, "url", "<none>"))
        diagnostics_json = json.dumps(diagnostics, ensure_ascii=False)

        log.error(
            "[%s] Failed to parse streaming response | reason=%s | status=%s | content_type=%s | server=%s | url=%s | diagnostics=%s",
            log_prefix,
            reason,
            response.status_code,
            content_type,
            server,
            response_url,
            diagnostics_json,
        )
        log.error(
            "[%s] Full streaming response body:\n%s",
            log_prefix,
            response_text if response_text else "<empty>",
        )

    @staticmethod
    def _apply_blacklist_notice(
        response: str, blacklist_punishment_active: bool
    ) -> str:
        if not blacklist_punishment_active or not response:
            return response
        notice = "(当前在黑名单中，已将消息替换) \n"
        if response.startswith(notice):
            return response
        return f"{notice}{response}"

    async def _send_with_network_retry_once(
        self,
        *,
        channel_label: str,
        send_coro_factory: Callable[[], Awaitable[Dict[str, Any]]],
        on_retry: Optional[
            Callable[[int, httpx.RequestError], Awaitable[None]]
        ] = None,
    ) -> Dict[str, Any]:
        """
        对“网络层请求失败”(httpx.RequestError)自动重试 1 次。

        - 仅重试 httpx.RequestError（连接/超时/DNS/TLS 等）
        - 第 2 次仍失败则抛出，由上层统一 return 错误信息
        """
        max_attempts = 2

        for attempt in range(max_attempts):
            try:
                return await send_coro_factory()
            except httpx.RequestError as e:
                if attempt >= max_attempts - 1:
                    raise

                err_fields = self._build_request_error_log_fields(e)
                retry_delay_seconds = 2**attempt
                log.warning(
                        "[%s] 网络层请求失败，自动重试 1 次 | exc_type=%s | exc_str=%s | "
                        "cause_type=%s | request_method=%s | request_url=%s | retry_in=%ss",
                        channel_label,
                        err_fields["exc_type"],
                        err_fields["exc_str"],
                        err_fields["cause_type"],
                        err_fields["request_method"],
                        err_fields["request_url"],
                        retry_delay_seconds,
                    )
                if on_retry is not None:
                    await on_retry(attempt, e)

                await asyncio.sleep(retry_delay_seconds)

    async def generate_simple_text(
        self,
        prompt: str,
        model_name: str,
        generation_config: Optional[Dict[str, Any]] = None,
        user_id: int = 0,
        override_base_url: Optional[str] = None,
    ) -> Optional[str]:
        """
        单轮、无工具的 OpenAI 兼容文本生成。

        用于内部任务，例如个人记忆摘要、礼物感谢词等一次性 prompt。
        这里不构建完整聊天上下文，也不执行工具调用。
        """
        effective_model_name = model_name or "deepseek-chat"
        is_deepseek_model = effective_model_name in {
            "deepseek-chat",
            "deepseek-reasoner",
        }
        is_custom_model = effective_model_name == "custom"
        is_kimi_model = effective_model_name == "kimi-k2.5"

        if not (is_deepseek_model or is_custom_model or is_kimi_model):
            log.warning(
                "[OpenAI Simple] 非 OpenAI 兼容模型不应走此路径: %s",
                effective_model_name,
            )
            return None

        custom_runtime_config: Optional[Dict[str, Any]] = None
        if is_custom_model:
            custom_runtime_config = self.custom_model_client.refresh_from_env()

        if is_deepseek_model:
            channel_label = "DeepSeek"
            validation_error = self.deepseek_model_client.get_validation_error(
                effective_model_name
            )
        elif is_custom_model:
            channel_label = "Custom"
            validation_error = self.custom_model_client.get_validation_error(
                runtime_config=custom_runtime_config
            )
        else:
            channel_label = "Kimi"
            validation_error = self.kimi_model_client.get_validation_error()

        if validation_error:
            log.warning(
                "[%s Simple] 模型校验失败: %s",
                channel_label,
                validation_error,
            )
            return None

        gen_config = dict(generation_config or {})
        if not gen_config:
            gen_config = app_config.MODEL_GENERATION_CONFIG.get(
                effective_model_name,
                app_config.MODEL_GENERATION_CONFIG["default"],
            )

        if is_deepseek_model:
            request_model_name = effective_model_name
        elif is_custom_model:
            request_model_name = self.custom_model_client.get_request_model_name(
                effective_model_name,
                runtime_config=custom_runtime_config,
            )
        else:
            request_model_name = self.kimi_model_client.get_request_model_name()

        payload: Dict[str, Any] = {
            "model": request_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": is_custom_model,
            "temperature": gen_config.get("temperature", 1.3),
            "top_p": gen_config.get("top_p", 0.95),
            "max_tokens": gen_config.get("max_output_tokens", 8192),
        }
        if is_kimi_model:
            payload["thinking"] = {"type": "disabled"}

        http_client = httpx.AsyncClient(timeout=120.0)

        async def recreate_http_client(
            attempt: int, error: httpx.RequestError
        ) -> None:
            nonlocal http_client

            stale_http_client = http_client
            http_client = httpx.AsyncClient(timeout=120.0)

            try:
                await stale_http_client.aclose()
            except Exception as close_error:
                log.warning(
                    "[Custom Simple] Failed to close stale HTTP client before retry | attempt=%s | request_error=%s | close_error=%s",
                    attempt + 1,
                    type(error).__name__,
                    close_error,
                )

        try:
            if is_deepseek_model:
                request_result = await self._send_with_network_retry_once(
                    channel_label=channel_label,
                    send_coro_factory=lambda: self.deepseek_model_client.send(
                        http_client=http_client,
                        payload=payload,
                        override_base_url=override_base_url,
                    ),
                )
            elif is_custom_model:
                request_result = await self._send_with_network_retry_once(
                    channel_label=channel_label,
                    send_coro_factory=lambda: self.custom_model_client.send(
                        http_client=http_client,
                        payload=payload,
                        override_base_url=override_base_url,
                        runtime_config=custom_runtime_config,
                    ),
                    on_retry=recreate_http_client,
                )
            else:
                request_result = await self._send_with_network_retry_once(
                    channel_label=channel_label,
                    send_coro_factory=lambda: self.kimi_model_client.send(
                        http_client=http_client,
                        payload=payload,
                        user_id=user_id,
                        override_base_url=override_base_url,
                    ),
                )

            response: httpx.Response = request_result["response"]
            used_api_url = str(request_result.get("used_api_url", ""))
            response.raise_for_status()

            result: Optional[Dict[str, Any]] = None
            if is_custom_model and self.custom_model_client.is_sse_chat_completion_response(
                response
            ):
                response_text = response.text or ""
                result = self.custom_model_client.parse_streaming_chat_completion_body(
                    response_text
                )
                if not result:
                    self._log_custom_stream_parse_failure(
                        log_prefix="Custom Simple",
                        response=response,
                        used_api_url=used_api_url,
                        response_text=response_text,
                    )
                    return None

            try:
                if result is None:
                    result = response.json()
            except json.JSONDecodeError:
                response_text = (response.text or "").strip()
                body_preview = response_text[:1200] if response_text else "<empty>"
                content_type = response.headers.get("content-type", "<none>")
                server = response.headers.get("server", "<none>")
                log.error(
                    "[%s Simple] 返回非 JSON 响应 | status=%s | content_type=%s | server=%s | url=%s | body=%s",
                    channel_label,
                    response.status_code,
                    content_type,
                    server,
                    used_api_url,
                    body_preview,
                )
                return None

            choices = result.get("choices")
            if not isinstance(choices, list) or not choices:
                log.warning(
                    "[%s Simple] 响应缺少 choices 字段: %s",
                    channel_label,
                    result,
                )
                return None

            first_choice = choices[0]
            if not isinstance(first_choice, dict):
                log.warning(
                    "[%s Simple] choices[0] 不是对象: %s",
                    channel_label,
                    first_choice,
                )
                return None

            response_message = first_choice.get("message")
            if not isinstance(response_message, dict):
                response_message = (
                    first_choice.get("delta")
                    if isinstance(first_choice.get("delta"), dict)
                    else {}
                )

            content = response_message.get("content") or ""
            if isinstance(content, list):
                content = self.kimi_model_client.extract_text_from_openai_content(
                    content
                )
            elif not isinstance(content, str):
                content = str(content).strip() if content else ""

            final_text = content.strip()
            if final_text:
                return final_text

            log.warning(
                "[%s Simple] 未能生成有效文本内容。finish_reason=%s, response=%s",
                channel_label,
                first_choice.get("finish_reason"),
                result,
            )
            return None

        except NoAvailableKimiKeyError as e:
            log.error("[Kimi Simple] %s", str(e) or "当前没有可用的 Kimi Key。")
            return None
        except httpx.HTTPStatusError as e:
            response_text = ""
            try:
                response_text = e.response.text
            except Exception:
                response_text = "<无法读取响应体>"

            log.error(
                "[%s Simple] HTTP 错误: %s | detail=%s",
                channel_label,
                e,
                response_text[:1000],
                exc_info=True,
            )
            return None
        except httpx.RequestError as e:
            err_fields = self._build_request_error_log_fields(e)
            log.error(
                "[%s Simple] 网络请求失败 | exc_type=%s | exc_repr=%s | exc_str=%s | cause_type=%s | cause_repr=%s | request_method=%s | request_url=%s",
                channel_label,
                err_fields["exc_type"],
                err_fields["exc_repr"],
                err_fields["exc_str"],
                err_fields["cause_type"],
                err_fields["cause_repr"],
                err_fields["request_method"],
                err_fields["request_url"],
                exc_info=True,
            )
            return None
        except Exception as e:
            log.error(
                "[%s Simple] 生成失败: %s",
                channel_label,
                e,
                exc_info=True,
            )
            return None
        finally:
            await http_client.aclose()

    async def generate_response(
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
        blacklist_punishment_active: bool = False,
        override_base_url: Optional[str] = None,
        videos: Optional[List[Dict[str, Any]]] = None,
        retrieval_query_text: Optional[str] = None,
        retrieval_query_embedding: Optional[List[float]] = None,
        rag_timeout_fallback: bool = False,
    ) -> str:
        """
        OpenAI 兼容专用通道（DeepSeek / Kimi / Custom）。
        """
        effective_model_name = model_name or "deepseek-chat"
        is_deepseek_model = effective_model_name in {
            "deepseek-chat",
            "deepseek-reasoner",
        }
        is_custom_model = effective_model_name == "custom"
        is_kimi_model = effective_model_name == "kimi-k2.5"
        is_direct_openai_model = is_deepseek_model or is_custom_model
        custom_runtime_config: Optional[Dict[str, Any]] = None
        if is_custom_model:
            custom_runtime_config = self.custom_model_client.refresh_from_env()
        custom_vision_enabled = bool(
            custom_runtime_config["enable_vision"] if custom_runtime_config else False
        )
        custom_video_input_enabled = bool(
            custom_runtime_config["enable_video_input"]
            if custom_runtime_config
            else False
        )

        if is_deepseek_model:
            channel_label = "DeepSeek"
        elif is_custom_model:
            channel_label = "Custom"
        else:
            channel_label = "Kimi"

        # 选择目标网关配置（由独立模型客户端负责）
        if is_deepseek_model:
            validation_error = self.deepseek_model_client.get_validation_error(
                effective_model_name
            )
            if validation_error:
                return self._apply_blacklist_notice(
                    validation_error, blacklist_punishment_active
                )
        elif is_custom_model:
            validation_error = self.custom_model_client.get_validation_error(
                runtime_config=custom_runtime_config
            )
            if validation_error:
                return self._apply_blacklist_notice(
                    validation_error, blacklist_punishment_active
                )
        else:
            validation_error = self.kimi_model_client.get_validation_error()
            if validation_error:
                return self._apply_blacklist_notice(
                    validation_error, blacklist_punishment_active
                )

        should_enable_kimi_web_search = False
        if is_kimi_model:
            should_enable_kimi_web_search = self.kimi_model_client.resolve_web_search_enabled(
                message=message,
                replied_message=replied_message,
            )

        images = self._compress_images_for_vps_mode(images)

        await chat_settings_service.increment_model_usage(effective_model_name)

        # 自动 RAG 检索
        if not world_book_entries and message:
            try:
                from src.chat.features.world_book.database.world_book_db_manager import (
                    world_book_db_manager,
                )

                found_entries = await world_book_db_manager.search_entries_in_message(message)
                if found_entries:
                    world_book_entries = found_entries
                    titles = [e.get("title", "未知") for e in found_entries]
                    # log.info(f"📚 [{channel_label}] 触发世界书/成员设定，已注入 Prompt: {titles}")
            except Exception as e:
                log.warning(f"[{channel_label}] 世界书自动检索失败: {e}")

        # 获取并转换工具
        dynamic_tools = await self.tool_service.get_dynamic_tools_for_context(
            user_id_for_settings=user_id_for_settings
        )
        openai_tools: List[Dict[str, Any]] = []

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
        _INTERNAL_PARAMS = {"bot", "guild", "channel", "guild_id", "thread_id", "kwargs"}

        def _schema_from_annotation(annotation: Any) -> Dict[str, Any]:
            if annotation is Any:
                return {"type": "string"}

            origin = get_origin(annotation)
            args = get_args(annotation)

            if args and any(arg is type(None) for arg in args):
                non_none_args = [arg for arg in args if arg is not type(None)]
                if len(non_none_args) == 1:
                    return _schema_from_annotation(non_none_args[0])
                return {"type": "string"}

            if origin in (list, List, tuple, Tuple):
                item_annotation = args[0] if args else str
                item_schema = _schema_from_annotation(item_annotation)
                if "type" not in item_schema and "anyOf" not in item_schema:
                    item_schema = {"type": "string"}
                return {"type": "array", "items": item_schema}

            if origin in (dict, Dict):
                return {"type": "object"}

            if origin is Union and args:
                non_none_args = [arg for arg in args if arg is not type(None)]
                if non_none_args:
                    return _schema_from_annotation(non_none_args[0])
                return {"type": "string"}

            if isinstance(annotation, type) and annotation in _PY_TYPE_MAP:
                schema_type = _PY_TYPE_MAP[annotation]
                if schema_type == "array":
                    return {"type": "array", "items": {"type": "string"}}
                return {"type": schema_type}

            if isinstance(annotation, str):
                normalized = annotation.replace("typing.", "").strip()
                normalized = normalized.replace("<class '", "").replace("'>", "")
                if normalized.startswith("Optional[") and normalized.endswith("]"):
                    return _schema_from_annotation(normalized[9:-1].strip())
                if normalized.startswith("List[") and normalized.endswith("]"):
                    inner = normalized[5:-1].strip()
                    item_schema = _schema_from_annotation(inner)
                    if "type" not in item_schema and "anyOf" not in item_schema:
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

        def _is_default_type_compatible(schema_type: Any, value: Any) -> bool:
            if isinstance(schema_type, list):
                if value is None:
                    return "null" in schema_type
                return any(
                    st != "null" and _is_default_type_compatible(st, value)
                    for st in schema_type
                )

            if schema_type == "null":
                return value is None
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

        _NO_CONVERSION = object()

        def _coerce_default_to_schema_type(schema_type: Any, default_value: Any) -> Any:
            target_types = schema_type if isinstance(schema_type, list) else [schema_type]

            if default_value is None and "null" in target_types:
                return None

            for t in target_types:
                if t == "null":
                    continue
                try:
                    if t == "string":
                        return str(default_value)
                    if t == "integer" and isinstance(default_value, str):
                        raw = default_value.strip()
                        if re.fullmatch(r"[+-]?\d+", raw):
                            return int(raw)
                    if t == "number" and isinstance(default_value, str):
                        return float(default_value.strip())
                    if t == "boolean" and isinstance(default_value, str):
                        lowered = default_value.strip().lower()
                        if lowered in {"true", "false"}:
                            return lowered == "true"
                except Exception:
                    continue

            return _NO_CONVERSION

        def _attach_default_if_compatible(
            prop_schema: Dict[str, Any], default_value: Any, field_path: str
        ) -> None:
            schema_type = prop_schema.get("type")
            if not schema_type:
                return

            if _is_default_type_compatible(schema_type, default_value):
                prop_schema["default"] = default_value
                return

            converted = _coerce_default_to_schema_type(schema_type, default_value)
            if converted is not _NO_CONVERSION and _is_default_type_compatible(
                schema_type, converted
            ):
                prop_schema["default"] = converted
                log.warning(
                    f"[{channel_label} Schema] 字段 '{field_path}' 的默认值类型不匹配，已自动修正为 {converted!r}。"
                )
                return

            prop_schema.pop("default", None)
            log.warning(
                f"[{channel_label} Schema] 字段 '{field_path}' 的默认值 {default_value!r} "
                f"与类型 '{schema_type}' 不匹配，已移除 default。"
            )

        def _resolve_local_ref(
            root_schema: Dict[str, Any], ref: str
        ) -> Optional[Dict[str, Any]]:
            if not isinstance(ref, str) or not ref.startswith("#/"):
                return None

            node: Any = root_schema
            for part in ref[2:].split("/"):
                if not isinstance(node, dict) or part not in node:
                    return None
                node = node[part]

            return node if isinstance(node, dict) else None

        def _normalize_schema_dict(
            schema: Any,
            field_path: str = "root",
            root_schema: Optional[Dict[str, Any]] = None,
        ) -> Any:
            if root_schema is None and isinstance(schema, dict):
                root_schema = schema

            if isinstance(schema, list):
                return [
                    _normalize_schema_dict(
                        item,
                        field_path=f"{field_path}[{idx}]",
                        root_schema=root_schema,
                    )
                    for idx, item in enumerate(schema)
                ]

            if not isinstance(schema, dict):
                return schema

            if "$ref" in schema and isinstance(root_schema, dict):
                resolved = _resolve_local_ref(root_schema, schema["$ref"])
                if resolved:
                    merged = dict(resolved)
                    for key, value in schema.items():
                        if key != "$ref":
                            merged[key] = value
                    schema = merged

            normalized: Dict[str, Any] = {}
            for key, value in schema.items():
                if key in {"$defs", "definitions"}:
                    continue

                if key == "properties" and isinstance(value, dict):
                    normalized_props = {}
                    for prop_name, prop_schema in value.items():
                        normalized_props[prop_name] = _normalize_schema_dict(
                            prop_schema,
                            field_path=f"{field_path}.properties.{prop_name}",
                            root_schema=root_schema,
                        )
                    normalized[key] = normalized_props
                    continue

                if key in {"items", "additionalProperties"}:
                    normalized[key] = _normalize_schema_dict(
                        value,
                        field_path=f"{field_path}.{key}",
                        root_schema=root_schema,
                    )
                    continue

                if key in {"anyOf", "oneOf", "allOf"} and isinstance(value, list):
                    variants = [
                        _normalize_schema_dict(
                            item,
                            field_path=f"{field_path}.{key}[{idx}]",
                            root_schema=root_schema,
                        )
                        for idx, item in enumerate(value)
                    ]

                    chosen_type = None
                    for item in variants:
                        if isinstance(item, dict):
                            t = item.get("type")
                            if isinstance(t, str) and t != "null":
                                chosen_type = t
                                break

                    normalized["type"] = chosen_type or "string"
                    continue

                if key == "type":
                    if isinstance(value, str):
                        normalized[key] = value.lower()
                    elif isinstance(value, list):
                        normalized[key] = [
                            t.lower() if isinstance(t, str) else t for t in value
                        ]
                    else:
                        normalized[key] = value
                    continue

                normalized[key] = value

            normalized.pop("nullable", None)

            if normalized.get("default") is None:
                normalized.pop("default", None)

            if "default" in normalized and "type" in normalized:
                _attach_default_if_compatible(normalized, normalized["default"], field_path)

            return normalized

        if dynamic_tools:
            import inspect as _inspect

            try:
                from pydantic import BaseModel as _BaseModel
            except ImportError:
                _BaseModel = None

            def _pydantic_to_schema(model_cls):
                raw_schema = model_cls.model_json_schema()
                normalized = _normalize_schema_dict(raw_schema, field_path=model_cls.__name__)

                if not isinstance(normalized, dict):
                    return {"type": "object", "properties": {}}

                normalized.pop("$defs", None)
                normalized.pop("definitions", None)

                if normalized.get("type") != "object":
                    fallback_schema: Dict[str, Any] = {
                        "type": "object",
                        "properties": normalized.get("properties", {}),
                    }
                    if isinstance(raw_schema, dict) and raw_schema.get("required"):
                        fallback_schema["required"] = raw_schema["required"]
                    return fallback_schema

                return normalized

            # 全局 TTS 模式：仅启用其中一个（所有模型生效）
            tts_mode_raw = await chat_settings_service.db_manager.get_global_setting(
                "global_tts_mode"
            )
            tts_mode = (tts_mode_raw or "legacy").strip().lower()
            if tts_mode not in {"legacy", "new"}:
                tts_mode = "legacy"
            disabled_tts_tool_name = "new_tts_tool" if tts_mode != "new" else "tts_tool"

            allow_deep_vision_tool = is_deepseek_model or (
                is_custom_model and custom_vision_enabled
            )

            for tool in dynamic_tools:
                func_name = getattr(tool, "__name__", "")
                if func_name == disabled_tts_tool_name:
                    log.info(
                        f"[{effective_model_name}] 已禁用工具: {func_name}（全局 TTS 模式: {tts_mode}）"
                    )
                    continue
                if func_name == "analyze_image_with_gemini_pro" and not allow_deep_vision_tool:
                    log.info(
                        f"[{effective_model_name}] 已禁用工具: {func_name}（仅 DeepSeek / custom(开启识图) 可用）"
                    )
                    continue

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

                        ann = param.annotation
                        if (
                            _BaseModel is not None
                            and ann != _inspect.Parameter.empty
                            and isinstance(ann, type)
                            and issubclass(ann, _BaseModel)
                        ):
                            sub_schema = _pydantic_to_schema(ann)
                            properties[param_name] = sub_schema
                            if param.default is _inspect.Parameter.empty:
                                required.append(param_name)
                            continue

                        prop_schema = _schema_from_annotation(
                            ann if ann != _inspect.Parameter.empty else Any
                        )
                        prop_schema = _normalize_schema_dict(
                            prop_schema, field_path=f"{func_name}.{param_name}"
                        )

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

                    final_params = {"type": "object", "properties": properties}
                    if required:
                        final_params["required"] = required
                    func_dict["parameters"] = final_params

                    openai_tools.append({"type": "function", "function": func_dict})
                    log.debug(f"[{channel_label}] 成功转换工具: {func_name}")

                except Exception as e:
                    log.error(
                        f"[{channel_label} 工具转换失败] 跳过工具 '{getattr(tool, '__name__', tool)}'，错误: {e}",
                        exc_info=True,
                    )

            if openai_tools:
                log.info(f"[{channel_label}] 成功转换 {len(openai_tools)} 个工具发往 API。")
            else:
                log.warning(f"[{channel_label}] 获取到了工具，但转换结果为空！")

        kimi_builtin_tools: List[Dict[str, Any]] = []
        if is_kimi_model and should_enable_kimi_web_search:
            kimi_builtin_tools = [
                {
                    "type": "builtin_function",
                    "function": {"name": "$web_search"},
                }
            ]

        # 构建 Prompt 并转 OpenAI 消息格式
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
            retrieval_query_text=retrieval_query_text,
            retrieval_query_embedding=retrieval_query_embedding,
            rag_timeout_fallback=rag_timeout_fallback,
        )

        openai_messages: List[Dict[str, Any]] = []
        is_first_user = True
        for turn in final_conversation:
            gemini_role = turn.get("role")

            # DeepSeek / Custom(开启识图)：沿用 OCR 拼接为纯文本
            if is_deepseek_model or (
                is_custom_model and custom_vision_enabled
            ):
                if is_deepseek_model:
                    content = await self.deepseek_model_client.build_turn_content(
                        turn.get("parts", []) or []
                    )
                else:
                    content = await self.custom_model_client.build_turn_content(
                        turn.get("parts", []) or [],
                        enable_vision=custom_vision_enabled,
                    )

                if not content:
                    continue

                if gemini_role == "model":
                    openai_messages.append({"role": "assistant", "content": content})
                else:
                    if is_first_user:
                        openai_messages.append({"role": "system", "content": content})
                        is_first_user = False
                    else:
                        openai_messages.append({"role": "user", "content": content})
                continue

            # Kimi / Custom：直接发送多模态 content block（图片直传，不走 Moonshot OCR）
            content_blocks = self.kimi_model_client.build_turn_content(
                turn.get("parts", []) or []
            )
            if not content_blocks:
                continue

            if gemini_role == "model":
                assistant_text = self.kimi_model_client.extract_text_from_openai_content(
                    content_blocks
                )
                if assistant_text:
                    openai_messages.append({"role": "assistant", "content": assistant_text})
            else:
                if is_first_user:
                    system_text = self.kimi_model_client.extract_text_from_openai_content(
                        content_blocks
                    )
                    if system_text:
                        openai_messages.append({"role": "system", "content": system_text})
                    is_first_user = False
                else:
                    openai_messages.append({"role": "user", "content": content_blocks})

        if is_deepseek_model and videos:
            log.info(f"[{channel_label}] 收到视频输入，但当前仅 Kimi 分支支持视频解析，已忽略。")

        if is_custom_model and videos:
            if custom_video_input_enabled and not custom_vision_enabled:
                video_blocks = self.kimi_model_client.build_video_content_blocks(videos)
                if video_blocks:
                    appended_to_existing_user = False
                    for msg in reversed(openai_messages):
                        if msg.get("role") != "user":
                            continue

                        msg_content = msg.get("content")
                        if isinstance(msg_content, list):
                            msg_content.extend(video_blocks)
                        elif isinstance(msg_content, str):
                            msg["content"] = [
                                {"type": "text", "text": msg_content},
                                *video_blocks,
                            ]
                        else:
                            msg["content"] = list(video_blocks)

                        appended_to_existing_user = True
                        break

                    if not appended_to_existing_user:
                        openai_messages.append(
                            {"role": "user", "content": list(video_blocks)}
                        )

                    log.info(
                        "[%s] 已注入 %s 个视频 block 到本轮请求。",
                        channel_label,
                        len(video_blocks),
                    )
                else:
                    log.info(
                        "[%s] 收到视频输入，但无可用视频 block（可能被大小或格式限制过滤）。",
                        channel_label,
                    )
            elif custom_video_input_enabled and custom_vision_enabled:
                log.warning(
                    "[%s] 检测到 enable_vision=true 且 enable_video_input=true，已按限制忽略视频输入。",
                    channel_label,
                )
            else:
                log.info(
                    "[%s] 收到视频输入，但“开启视频输入”未启用，已忽略。",
                    channel_label,
                )

        if is_kimi_model and videos:
            video_blocks = self.kimi_model_client.build_video_content_blocks(videos)
            if video_blocks:
                appended_to_existing_user = False
                for msg in reversed(openai_messages):
                    if msg.get("role") != "user":
                        continue

                    msg_content = msg.get("content")
                    if isinstance(msg_content, list):
                        msg_content.extend(video_blocks)
                    elif isinstance(msg_content, str):
                        msg["content"] = [{"type": "text", "text": msg_content}, *video_blocks]
                    else:
                        msg["content"] = list(video_blocks)

                    appended_to_existing_user = True
                    break

                if not appended_to_existing_user:
                    openai_messages.append({"role": "user", "content": list(video_blocks)})

                log.info("[Kimi] 已注入 %s 个视频 block 到本轮请求。", len(video_blocks))
            else:
                log.info("[Kimi] 收到视频输入，但无可用视频 block（可能被限制策略过滤）。")

        def _truncate_data_uri_for_log(url: str) -> str:
            if not isinstance(url, str):
                return str(url)

            if url.startswith("data:") and ";base64," in url:
                prefix, b64_data = url.split(";base64,", 1)
                preview = b64_data[:80]
                return f"{prefix};base64,{preview}...(truncated, total_base64_chars={len(b64_data)})"

            if len(url) > 500:
                return url[:500] + "...(truncated)"

            return url

        def _sanitize_openai_payload_for_log(obj: Any) -> Any:
            if isinstance(obj, dict):
                sanitized = {}
                for k, v in obj.items():
                    if k == "url" and isinstance(v, str):
                        sanitized[k] = _truncate_data_uri_for_log(v)
                    else:
                        sanitized[k] = _sanitize_openai_payload_for_log(v)
                return sanitized

            if isinstance(obj, list):
                return [_sanitize_openai_payload_for_log(item) for item in obj]

            if isinstance(obj, str) and len(obj) > 1500:
                return obj[:1500] + "...(truncated)"

            return obj

        log_detailed = app_config.DEBUG_CONFIG.get("LOG_DETAILED_GEMINI_PROCESS", False)
        if log_detailed:
            current_log_model_name = effective_model_name
            if is_custom_model:
                current_log_model_name = self.custom_model_client.get_request_model_name(
                    effective_model_name,
                    runtime_config=custom_runtime_config,
                )
            elif is_kimi_model:
                current_log_model_name = self.kimi_model_client.get_request_model_name()

            safe_openai_messages = _sanitize_openai_payload_for_log(openai_messages)
            # log.info(f"--- [{channel_label}] 完整发送上下文 (用户 {user_id}) ---")
            # log.info(json.dumps(safe_openai_messages, ensure_ascii=False, indent=2, default=str))

            if is_deepseek_model or is_custom_model:
                request_tools_for_log = openai_tools
            elif is_kimi_model:
                request_tools_for_log = openai_tools + kimi_builtin_tools
            else:
                request_tools_for_log = []
            '''
            if request_tools_for_log:
                log.info(f"--- [{channel_label}] 工具列表 ---")
                log.info(
                    json.dumps(
                        request_tools_for_log, ensure_ascii=False, indent=2, default=str
                    )
                )
            '''
            log.info("------------------------------------")
            log.info("[%s] 当前模型: %s", channel_label, current_log_model_name)

        # 核心请求循环
        if override_base_url:
            log.info(f"🧪 一次性调试已生效：{channel_label} 通道临时改用 URL: {override_base_url}")

        gen_config = app_config.MODEL_GENERATION_CONFIG.get(
            effective_model_name, app_config.MODEL_GENERATION_CONFIG["default"]
        )

        max_calls = 5
        called_tool_names: List[str] = []
        bad_format_retries = 0
        deep_vision_used = False
        tool_image_context_list = self._build_tool_image_context_list(images)
        kimi_token_too_long_retry_used = False

        # 本轮对话是否跳过公益站（如果本轮中公益站曾报错，则后续因违禁词等引起的重试直接走官方站）
        skip_custom_site_for_this_turn = False

        http_client = httpx.AsyncClient(timeout=120.0)

        async def recreate_http_client(
            attempt: int, error: httpx.RequestError
        ) -> None:
            nonlocal http_client

            stale_http_client = http_client
            http_client = httpx.AsyncClient(timeout=120.0)

            try:
                await stale_http_client.aclose()
            except Exception as close_error:
                log.warning(
                    "[Custom] Failed to close stale HTTP client before retry | attempt=%s | request_error=%s | close_error=%s",
                    attempt + 1,
                    type(error).__name__,
                    close_error,
                )

            log.warning(
                "[Custom] Recreated HTTP client before retry | attempt=%s | request_error=%s",
                attempt + 1,
                type(error).__name__,
            )

        try:
            for i in range(max_calls):
                if is_deepseek_model:
                    request_model_name = effective_model_name
                elif is_custom_model:
                    request_model_name = self.custom_model_client.get_request_model_name(
                        effective_model_name,
                        runtime_config=custom_runtime_config,
                    )
                else:
                    request_model_name = self.kimi_model_client.get_request_model_name()

                payload = {
                    "model": request_model_name,
                    "messages": openai_messages,
                    "stream": is_custom_model,
                    "temperature": gen_config.get("temperature", 1.3),
                    "top_p": gen_config.get("top_p", 0.95),
                    "max_tokens": gen_config.get("max_output_tokens", 8192),
                }

                if is_kimi_model:
                    payload["thinking"] = {"type": "disabled"}

                if is_deepseek_model or is_custom_model:
                    if openai_tools:
                        payload["tools"] = openai_tools
                        if is_custom_model:
                            payload["tool_choice"] = "auto"
                            payload["parallel_tool_calls"] = True
                elif is_kimi_model:
                    kimi_request_tools: List[Dict[str, Any]] = []
                    if openai_tools:
                        kimi_request_tools.extend(openai_tools)
                    if kimi_builtin_tools:
                        kimi_request_tools.extend(kimi_builtin_tools)
                    if kimi_request_tools:
                        payload["tools"] = kimi_request_tools

                if is_deepseek_model:
                    request_result = await self._send_with_network_retry_once(
                        channel_label=channel_label,
                        send_coro_factory=lambda: self.deepseek_model_client.send(
                            http_client=http_client,
                            payload=payload,
                            override_base_url=override_base_url,
                        ),
                    )
                elif is_custom_model:
                    request_result = await self._send_with_network_retry_once(
                        channel_label=channel_label,
                        send_coro_factory=lambda: self.custom_model_client.send(
                            http_client=http_client,
                            payload=payload,
                            override_base_url=override_base_url,
                            runtime_config=custom_runtime_config,
                        ),
                        on_retry=recreate_http_client,
                    )
                else:
                    request_result = await self._send_with_network_retry_once(
                        channel_label=channel_label,
                        send_coro_factory=lambda: self.kimi_model_client.send(
                            http_client=http_client,
                            payload=payload,
                            user_id=user_id,
                            override_base_url=override_base_url,
                            log_detailed=log_detailed,
                            skip_custom_site_for_this_turn=skip_custom_site_for_this_turn,
                        ),
                    )
                    skip_custom_site_for_this_turn = bool(
                        request_result.get(
                            "skip_custom_site_for_this_turn",
                            skip_custom_site_for_this_turn,
                        )
                    )

                response: httpx.Response = request_result["response"]
                used_api_url = str(request_result.get("used_api_url", ""))
                used_slot_label = str(request_result.get("used_slot_label", ""))
                used_kimi_slot_id = request_result.get("used_slot_id")
                used_kimi_key_tail = str(request_result.get("used_key_tail", "N/A"))
                used_kimi_model_name = str(
                    request_result.get("used_model_name", request_model_name)
                )
                if response.is_error and is_kimi_model:
                    response_text = response.text or ""
                    lowered_response_text = response_text.lower()
                    if (
                        not kimi_token_too_long_retry_used
                        and (
                            "input token length too long" in lowered_response_text
                            or "context_length_exceeded" in lowered_response_text
                        )
                    ):
                        old_len = len(openai_messages)
                        openai_messages = self.kimi_model_client.trim_messages_for_retry(
                            openai_messages
                        )
                        new_len = len(openai_messages)
                        kimi_token_too_long_retry_used = True
                        log.warning(
                            "[Kimi] 命中输入过长，已裁剪上下文后重试 | status=%s | old_messages=%s | new_messages=%s",
                            response.status_code,
                            old_len,
                            new_len,
                        )
                        continue

                response.raise_for_status()

                result: Optional[Dict[str, Any]] = None
                if is_custom_model and self.custom_model_client.is_sse_chat_completion_response(
                    response
                ):
                    response_text = response.text or ""
                    result = self.custom_model_client.parse_streaming_chat_completion_body(
                        response_text
                    )
                    if not result:
                        self._log_custom_stream_parse_failure(
                            log_prefix="Custom",
                            response=response,
                            used_api_url=used_api_url,
                            response_text=response_text,
                        )
                        return self._apply_blacklist_notice(
                            "custom 通道返回了无法解析的流式响应，请稍后再试。",
                            blacklist_punishment_active,
                        )

                try:
                    if result is None:
                        result = response.json()
                except json.JSONDecodeError:
                    response_text = (response.text or "").strip()

                    if is_kimi_model:
                        body_preview = response_text[:1200] if response_text else "<empty>"
                        content_type = response.headers.get("content-type", "<none>")
                        server = response.headers.get("server", "<none>")
                        log.error(
                            "[Kimi] 返回非 JSON 响应 | status=%s | content_type=%s | server=%s | "
                            "url=%s | source=%s | key_tail=%s | body=%s",
                            response.status_code,
                            content_type,
                            server,
                            used_api_url,
                            used_slot_label,
                            used_kimi_key_tail,
                            body_preview,
                        )
                        return self._apply_blacklist_notice(
                            "Kimi 通道返回了非标准响应（非 JSON），请稍后再试。",
                            blacklist_punishment_active,
                        )

                    if is_custom_model:
                        body_preview = response_text[:1200] if response_text else "<empty>"
                        content_type = response.headers.get("content-type", "<none>")
                        server = response.headers.get("server", "<none>")
                        log.error(
                            "[Custom] 返回非 JSON 响应 | status=%s | content_type=%s | server=%s | "
                            "url=%s | body=%s",
                            response.status_code,
                            content_type,
                            server,
                            used_api_url,
                            body_preview,
                        )
                        return self._apply_blacklist_notice(
                            "custom 通道返回了非标准响应（非 JSON），请稍后再试。",
                            blacklist_punishment_active,
                        )

                    raise

                first_choice = result["choices"][0]
                response_message = first_choice.get("message")
                if not isinstance(response_message, dict):
                    response_message = (
                        first_choice.get("delta")
                        if isinstance(first_choice.get("delta"), dict)
                        else {}
                    )

                reasoning_content = response_message.get("reasoning_content")
                content = response_message.get("content") or ""
                tool_calls = response_message.get("tool_calls")

                if reasoning_content:
                    log.info(
                        f"--- [{channel_label}] 思考过程 ---\n{reasoning_content}\n-----------------------------"
                    )

                msg_to_append: Dict[str, Any] = {
                    "role": "assistant",
                    "content": content,
                }
                if reasoning_content is not None:
                    msg_to_append["reasoning_content"] = reasoning_content
                if tool_calls is not None:
                    msg_to_append["tool_calls"] = tool_calls

                openai_messages.append(msg_to_append)

                if not tool_calls:
                    if content:
                        has_forbidden_phrase = bool(
                            re.search(
                                r"不过话说回来|话说回来|话又说回来|不过话又说回来|不过说真的",
                                content,
                            )
                        )
                        content_len = len(content)

                        if has_forbidden_phrase and content_len <= 800:
                            if bad_format_retries < 3:
                                log.warning(
                                    f"[{channel_label}] 检测到违禁词 (尝试 {bad_format_retries + 1}/3)，正在重试..."
                                )
                                openai_messages.append(
                                    {
                                        "role": "user",
                                        "content": "[系统提示] 检测到你使用了“不过说真的|不过话说回来|话说回来|话又说回来”。这是被禁止的。请重新生成回复，去掉这个短语，保持语气自然。",
                                    }
                                )
                                bad_format_retries += 1
                                continue
                            return self._apply_blacklist_notice(
                                "抱歉，我的说话格式一直达不到要求，我是杂鱼",
                                blacklist_punishment_active,
                            )
                        elif has_forbidden_phrase:
                            log.info(
                                f"[{channel_label}] 检测到违禁词，但回复长度为 {content_len} (>800)，按成本优化策略放行。"
                            )

                    allowed_emoji_names = {"开心","乖巧","害羞","吃瓜","偷笑","比心","desuwa","伤心","生气","加油", "好奇","邀请","傲娇","祝福","你好","叹气","投降",}
                    removed_emoji_tags: List[str] = []

                    def _strip_disallowed_emoji_tag(match):
                        emoji_name = match.group(1).strip()
                        if emoji_name in allowed_emoji_names:
                            return match.group(0)

                        removed_emoji_tags.append(match.group(0))
                        return ""

                    if content:
                        # 先处理只有闭合标签的半截思维链，直接移除从开头到闭合标签的全部内容
                        partial_thinking_match = re.search(
                            r"</(?:think|思考)>",
                            content,
                        )
                        if partial_thinking_match:
                            content = content[partial_thinking_match.end() :].lstrip()

                        # 再移除完整的思维链标签及其内容（如 <think>...</think>、<arg_key>...icism> 等）
                        # 常见的思维链标签对
                        thinking_patterns = [
                            (r"<think>", r"</think>"),
                            (r"<thinking>", r"</thinking>"),
                            (r"<reasoning>", r"</reasoning>"),
                            (r"<reflection>", r"</reflection>"),
                            (r"<思考>", r"</思考>"),
                        ]
                        for start_tag, end_tag in thinking_patterns:
                            # 转义正则特殊字符
                            escaped_start = re.escape(start_tag)
                            escaped_end = re.escape(end_tag)
                            # 使用 DOTALL 标志使 . 匹配换行符
                            content = re.sub(
                                rf"{escaped_start}.*?{escaped_end}",
                                "",
                                content,
                                flags=re.DOTALL,
                            )

                        # 进一步增强正则，处理 </tag>、<tag/>、<tag /> 等标签格式
                        content = re.sub(
                            r"</?((?!@)[^<>\s/]{1,20})\s*/?>",
                            _strip_disallowed_emoji_tag,
                            content,
                        )

                    if removed_emoji_tags:
                        unique_removed_tags = list(dict.fromkeys(removed_emoji_tags))
                        log.warning(
                            f"[{channel_label}] 检测并剔除非白名单表情标签 | user_id=%s | model=%s | count=%s | removed=%s",
                            user_id,
                            effective_model_name,
                            len(removed_emoji_tags),
                            unique_removed_tags,
                        )

                    if log_detailed:
                        log.info(f"--- [{channel_label}] 模型决策：直接生成文本回复 (未调用工具) ---")

                    self.last_called_tools = called_tool_names
                    log.info(f"--- [{channel_label}] 文本生成完成 ---")

                    if "usage" in result:
                        try:
                            usage = result["usage"]
                            input_tokens = usage.get("prompt_tokens", 0)
                            output_tokens = usage.get("completion_tokens", 0)
                            total_tokens = usage.get("total_tokens", 0)

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
                                f"[{channel_label}] Token 记录: In={input_tokens}, Out={output_tokens}, Total={total_tokens}"
                            )
                        except Exception as e:
                            log.error(f"[{channel_label}] Token 记录失败: {e}")

                    response_text = await self.post_process_response(
                        content, user_id, guild_id
                    )
                    return self._apply_blacklist_notice(
                        response_text, blacklist_punishment_active
                    )

                if log_detailed:
                    log.info(
                        f"--- [{channel_label}] 模型决策：建议进行工具调用 (第 {i + 1}/{max_calls} 次) ---"
                    )
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

                    if tool_name == "$web_search":
                        search_tokens = args.get("usage", {}).get("total_tokens")
                        if search_tokens is not None:
                            log.info("[Kimi] $web_search 内容 token 数: %s", search_tokens)

                        openai_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "name": tool_name,
                                "content": json.dumps(args, ensure_ascii=False),
                            }
                        )
                        continue

                    if tool_name == "analyze_image_with_gemini_pro":
                        if deep_vision_used:
                            openai_messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": call["id"],
                                    "name": tool_name,
                                    "content": json.dumps(
                                        {
                                            "error": "深度识图工具本轮已调用过一次。为避免明显变慢，本轮不再重复调用。"
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            )
                            continue
                        deep_vision_used = True

                    mock_gemini_call = types.FunctionCall(name=tool_name, args=args)

                    tool_res = await self.tool_service.execute_tool_call(
                        tool_call=mock_gemini_call,
                        channel=channel,
                        user_id=user_id,
                        log_detailed=log_detailed,
                        user_id_for_settings=user_id_for_settings,
                        image_context_list=tool_image_context_list,
                    )

                    if isinstance(tool_res, types.Part) and tool_res.function_response:
                        raw_response = tool_res.function_response.response
                        if is_deepseek_model:
                            processed_response = await self.deepseek_model_client.post_process_tool_response(
                                raw_response
                            )
                        elif is_custom_model:
                            processed_response = await self.custom_model_client.post_process_tool_response(
                                raw_response,
                                enable_vision=custom_vision_enabled,
                            )
                        else:
                            import copy as _copy

                            processed_response = _copy.deepcopy(raw_response)

                            if isinstance(processed_response, dict):
                                result_data = processed_response.get("result", {})
                                if isinstance(result_data, dict):
                                    profile = result_data.get("profile", {})
                                    if isinstance(profile, dict):
                                        for image_key, label_cn, label in [
                                            ("avatar_image_base64", "头像", "avatar"),
                                            ("banner_image_base64", "横幅", "banner"),
                                        ]:
                                            if image_key in profile:
                                                profile.pop(image_key, None)
                                                profile.setdefault(
                                                    f"{label}_note",
                                                    f"（{label_cn}原始图片数据已省略）",
                                                )

                        content_str = json.dumps(processed_response, ensure_ascii=False)
                    else:
                        content_str = str(tool_res)

                    openai_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": tool_name,
                            "content": content_str,
                        }
                    )

            self.last_called_tools = called_tool_names
            return self._apply_blacklist_notice(
                "哎呀，我好像陷入了一个复杂的思考循环里，换个话题聊聊吧！",
                blacklist_punishment_active,
            )

        except NoAvailableKimiKeyError as e:
            err_msg = str(e) or "所有 Key 当前不可用（可能均在冷却或次日封禁中）。"
            log.error(f"[Kimi] {err_msg}")
            # 仅在特定消息时发送私信
            if "冷却" in err_msg:
                await self.kimi_model_client.notify_alert(err_msg)
            return self._apply_blacklist_notice(
                err_msg, blacklist_punishment_active
            )

        except httpx.HTTPStatusError as e:
            error_info = f"{type(e).__name__}: {str(e)}"
            response_text = ""
            try:
                response_text = e.response.text
            except Exception:
                response_text = "<无法读取响应体>"

            print(f"{channel_label} 致命错误详情: {response_text}")
            log.error(f"{channel_label} API 调用失败: {error_info}", exc_info=True)
            log.error(f"{channel_label} 致命错误详情: {response_text}")

            short_detail = response_text[:500] if response_text else "无响应体"
            return self._apply_blacklist_notice(
                f"{channel_label} 连接失败: {error_info}。详情: {short_detail}",
                blacklist_punishment_active,
            )

        except Exception as e:
            if isinstance(e, httpx.RequestError):
                err_fields = self._build_request_error_log_fields(e)
                log.error(
                    "[%s] 网络层请求失败 | exc_type=%s | exc_repr=%s | exc_str=%s | "
                    "cause_type=%s | cause_repr=%s | request_method=%s | request_url=%s",
                    channel_label,
                    err_fields["exc_type"],
                    err_fields["exc_repr"],
                    err_fields["exc_str"],
                    err_fields["cause_type"],
                    err_fields["cause_repr"],
                    err_fields["request_method"],
                    err_fields["request_url"],
                    exc_info=True,
                )
                return self._apply_blacklist_notice(
                    (
                        f"哎呀，{channel_label} 网关好像闹别扭了：{err_fields['exc_type']}。"
                        "等会再试吧。"
                    ),
                    blacklist_punishment_active,
                )

            error_info = f"{type(e).__name__}: {str(e)}"
            log.error(f"{channel_label} API 调用失败: {error_info}", exc_info=True)
            return self._apply_blacklist_notice(
                f"{channel_label} 连接失败: {error_info}。请检查日志或配置。",
                blacklist_punishment_active,
            )
        finally:
            await http_client.aclose()
