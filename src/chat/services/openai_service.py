# -*- coding: utf-8 -*-

import base64
import io
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union, get_args, get_origin
from zoneinfo import ZoneInfo

import httpx
from PIL import Image
from google.genai import types

from src.chat.config import chat_config as app_config
from src.chat.features.chat_settings.services.chat_settings_service import chat_settings_service
from src.chat.features.tools.services.tool_service import ToolService
from src.chat.services.kimi_key_rotation import (
    KimiKeyRotationService,
    NoAvailableKimiKeyError,
)
from src.chat.services.moonshot_vision_service import moonshot_vision_service
from src.chat.services.prompt_service import prompt_service
from src.database.database import AsyncSessionLocal
from src.database.services.token_usage_service import token_usage_service

log = logging.getLogger(__name__)


class OpenAIService:
    """
    OpenAI 兼容通道服务（DeepSeek / Kimi）。
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

        # OpenAI 兼容模型配置（独立于 Gemini 配置）
        self.deepseek_url = os.getenv("DEEPSEEK_URL")
        self.deepseek_key = os.getenv("DEEPSEEK_API_KEY")

        # Kimi 网关配置：仅保留官方 MOONSHOT_*
        self.moonshot_url = os.getenv("MOONSHOT_URL")
        self.moonshot_key = os.getenv("MOONSHOT_API_KEY")
        self.kimi_model = "kimi-k2.5"

        self.kimi_key_rotation = KimiKeyRotationService(
            moonshot_base_url=self.moonshot_url,
            moonshot_api_key=self.moonshot_key,
        )
        self.kimi_alert_user_id = 1046310552365973524

        if self.deepseek_url and self.deepseek_key:
            log.info(f"✅ [OpenAIService] 已加载 DeepSeek 配置。URL: {self.deepseek_url}")
        if self.moonshot_url and self.moonshot_key:
            log.info(f"✅ [OpenAIService] 已加载 Kimi 官方配置。URL: {self.moonshot_url}")

    async def _notify_kimi_alert(self, content: str) -> None:
        """向指定管理员发送 Kimi 轮换告警私信（失败不影响主流程）。"""
        bot = getattr(self.tool_service, "bot", None)
        if bot is None:
            log.warning("[KimiAlert] Bot 实例尚未注入，跳过私信告警: %s", content)
            return

        try:
            user = bot.get_user(self.kimi_alert_user_id)
            if user is None:
                user = await bot.fetch_user(self.kimi_alert_user_id)
            if user is None:
                log.warning("[KimiAlert] 未找到告警用户: %s", self.kimi_alert_user_id)
                return

            await user.send(content)
        except Exception as e:
            log.warning("[KimiAlert] 发送私信告警失败: %s", e)

    def _build_moonshot_image_payload_from_pil(self, image: Image.Image) -> Dict[str, Any]:
        """将 PIL 图片转换为 Moonshot 识别所需 payload。"""
        mime_type = "image/webp"
        buffered = io.BytesIO()
        try:
            image.save(buffered, format="WEBP")
        except Exception:
            # 兜底为 PNG，避免运行环境缺少 WEBP 编码支持时失败
            mime_type = "image/png"
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")

        image_bytes = buffered.getvalue()
        return {
            "type": "image",
            "mime_type": mime_type,
            "data_size": len(image_bytes),
            # 沿用既有字段命名
            "data_preview": image_bytes.hex(),
        }

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

    async def _build_deepseek_turn_content(self, parts: List[Any]) -> str:
        """
        构建单条消息在 DeepSeek 通道中的文本内容。
        对于图片，调用 Moonshot 进行识别并将结果插入到对应位置。
        """
        content_chunks: List[str] = []

        for part in parts or []:
            if hasattr(part, "thought") and getattr(part, "thought", False):
                continue

            # 1) 文本字典
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                content_chunks.append(part["text"])
                continue

            # 2) PIL 图片对象
            if isinstance(part, Image.Image):
                try:
                    image_payload = self._build_moonshot_image_payload_from_pil(part)
                    vision_text = await moonshot_vision_service.recognize_image(image_payload)
                except Exception as e:
                    log.error("Moonshot 图片识别流程异常: %s", e, exc_info=True)
                    vision_text = "（图片识别失败：处理流程异常）"

                content_chunks.append(f"\n【图片识别结果】{vision_text}\n")
                continue

            # 3) 已是图片结构的字典（兼容扩展）
            if isinstance(part, dict) and part.get("type") == "image":
                try:
                    vision_text = await moonshot_vision_service.recognize_image(part)
                except Exception as e:
                    log.error("Moonshot 图片字典识别异常: %s", e, exc_info=True)
                    vision_text = "（图片识别失败：处理流程异常）"

                content_chunks.append(f"\n【图片识别结果】{vision_text}\n")
                continue

            # 4) 其他类型兜底
            content_chunks.append(str(part))

        return "".join(content_chunks).strip()

    def _extract_text_from_openai_content(
        self, content: Union[str, List[Dict[str, Any]]]
    ) -> str:
        """
        从 OpenAI 消息 content 中提取纯文本。
        content 可能是 string（文本模式）或 block 列表（多模态模式）。
        """
        if isinstance(content, str):
            return content.strip()

        text_chunks: List[str] = []
        for block in content or []:
            if (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                text_value = block["text"].strip()
                if text_value:
                    text_chunks.append(text_value)

        return "\n".join(text_chunks).strip()

    def _build_kimi_turn_content(self, parts: List[Any]) -> List[Dict[str, Any]]:
        """
        构建 Kimi (OpenAI 兼容多模态) 单条消息 content。
        直接把图片作为 image_url(data URI) 发给模型，不做 OCR。
        """
        content_blocks: List[Dict[str, Any]] = []

        for part in parts or []:
            if hasattr(part, "thought") and getattr(part, "thought", False):
                continue

            # 1) 文本字典
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text_value = part["text"].strip()
                if text_value:
                    content_blocks.append({"type": "text", "text": text_value})
                continue

            # 2) Gemini Part（兼容）
            if isinstance(part, types.Part):
                if part.text:
                    text_value = part.text.strip()
                    if text_value:
                        content_blocks.append({"type": "text", "text": text_value})
                    continue

                if part.inline_data and part.inline_data.data:
                    mime_type = part.inline_data.mime_type or "image/png"
                    image_b64 = base64.b64encode(part.inline_data.data).decode("utf-8")
                    content_blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_b64}"
                            },
                        }
                    )
                    continue

            # 3) PIL 图片对象
            if isinstance(part, Image.Image):
                buffered = io.BytesIO()
                part.save(buffered, format="PNG")
                image_bytes = buffered.getvalue()
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                content_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    }
                )
                continue

            # 4) 图片字典
            if isinstance(part, dict) and part.get("type") == "image":
                mime_type = str(part.get("mime_type", "image/png"))
                image_bytes: Optional[bytes] = None

                direct_bytes = part.get("data") or part.get("bytes")
                if isinstance(direct_bytes, (bytes, bytearray)):
                    image_bytes = bytes(direct_bytes)

                if image_bytes is None:
                    image_base64 = part.get("image_base64")
                    if isinstance(image_base64, str) and image_base64.strip():
                        try:
                            image_bytes = base64.b64decode(image_base64)
                        except Exception:
                            image_bytes = None

                if image_bytes is None:
                    data_preview = part.get("data_preview")
                    if isinstance(data_preview, str) and data_preview.strip():
                        try:
                            image_bytes = bytes.fromhex(data_preview.strip())
                        except Exception:
                            try:
                                image_bytes = base64.b64decode(data_preview.strip())
                            except Exception:
                                image_bytes = None

                if image_bytes:
                    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                    content_blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                        }
                    )
                else:
                    content_blocks.append(
                        {"type": "text", "text": "（收到一张图片，但解析失败）"}
                    )
                continue

            # 5) 兜底文本
            fallback_text = str(part).strip()
            if fallback_text:
                content_blocks.append({"type": "text", "text": fallback_text})

        return content_blocks

    @staticmethod
    def _build_chat_completions_url(base_url: str) -> str:
        normalized = (base_url or "").rstrip("/")
        if not normalized.endswith("/chat/completions"):
            normalized += "/chat/completions"
        return normalized

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

    @staticmethod
    def _trim_kimi_messages_for_retry(
        messages: List[Dict[str, Any]], keep_recent_non_system: int = 24
    ) -> List[Dict[str, Any]]:
        """
        当 Kimi 出现输入过长时，保留首条 system + 最近若干非 system 消息进行重试。
        """
        if len(messages) <= keep_recent_non_system + 1:
            return messages

        first_system: Optional[Dict[str, Any]] = None
        non_system_messages: List[Dict[str, Any]] = []

        for item in messages:
            if first_system is None and item.get("role") == "system":
                first_system = item
                continue
            non_system_messages.append(item)

        trimmed: List[Dict[str, Any]] = []
        if first_system is not None:
            trimmed.append(first_system)
        trimmed.extend(non_system_messages[-keep_recent_non_system:])
        return trimmed

    @staticmethod
    def _should_enable_kimi_web_search(
        message: Optional[str], replied_message: Optional[str] = None
    ) -> bool:
        """
        仅在用户明确表达“上网搜索”意图时开启联网搜索：
        - 显式包含“上网”；
        - 或消息内包含 URL；
        - 或出现明确联网关键词（如“联网搜索/在线搜索/打开链接”）。
        """
        text_parts = []
        if isinstance(message, str) and message.strip():
            text_parts.append(message.strip())
        if isinstance(replied_message, str) and replied_message.strip():
            text_parts.append(replied_message.strip())

        if not text_parts:
            return False

        combined_text = " ".join(text_parts)
        lowered_text = combined_text.lower()

        if "上网" in combined_text:
            return True

        if re.search(r"https?://|www\.", lowered_text):
            return True

        explicit_network_keywords = (
            "联网搜索",
            "联网查",
            "在线搜索",
            "打开链接",
            "访问链接",
            "查看链接",
            "网页链接",
            "网址",
        )
        return any(keyword in combined_text for keyword in explicit_network_keywords)

    async def _post_kimi_with_rotation(
        self,
        http_client: httpx.AsyncClient,
        payload: Dict[str, Any],
        override_base_url: Optional[str] = None,
    ) -> Tuple[httpx.Response, str, str, str, str, str]:
        """
        使用 Kimi 官方 key 路由服务发起请求：
        - 仅官方 moonshot key 轮询；
        - 保留 429 两阶段惩罚（2 分钟冷却 -> 次日封禁）；
        - 非 429 报错会优先在同一个 key 上自动重试。
        """
        non_429_same_key_retry_limit = 2  # 额外重试次数（总尝试=1+2）
        while True:
            slot = await self.kimi_key_rotation.acquire_active_slot()
            available_count, total_count = await self.kimi_key_rotation.get_pool_stats()
            current_base_url = (override_base_url or slot.base_url or "").rstrip("/")
            current_api_url = self._build_chat_completions_url(current_base_url)

            request_payload = dict(payload)
            request_payload["model"] = self.kimi_model
            request_payload["thinking"] = {"type": "disabled"}

            log.info(
                "[Kimi] 生成前路由 | site=%s | model=%s | url=%s | key_tail=%s | available_keys=%s/%s",
                "官方站点",
                request_payload["model"],
                current_api_url,
                slot.key_tail,
                available_count,
                total_count,
            )

            total_attempts = non_429_same_key_retry_limit + 1
            hit_429 = False
            hit_tpd = False
            for attempt in range(total_attempts):
                attempt_no = attempt + 1

                try:
                    response = await http_client.post(
                        current_api_url,
                        headers={
                            "Authorization": f"Bearer {slot.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=request_payload,
                    )
                except httpx.RequestError as e:
                    if attempt < non_429_same_key_retry_limit:
                        log.warning(
                            "[Kimi] 非429网络异常，同key自动重试 (%s/%s) | key_tail=%s | reason=%s",
                            attempt_no,
                            total_attempts,
                            slot.key_tail,
                            str(e),
                        )
                        continue

                    err_fields = self._build_request_error_log_fields(e)
                    log.error(
                        "[Kimi] 网络请求异常 | slot_id=%s | key_tail=%s | base_url=%s | api_url=%s | "
                        "exc_type=%s | exc_repr=%s | exc_str=%s | cause_type=%s | cause_repr=%s | "
                        "request_method=%s | request_url=%s",
                        slot.slot_id,
                        slot.key_tail,
                        current_base_url,
                        current_api_url,
                        err_fields["exc_type"],
                        err_fields["exc_repr"],
                        err_fields["exc_str"],
                        err_fields["cause_type"],
                        err_fields["cause_repr"],
                        err_fields["request_method"],
                        err_fields["request_url"],
                        exc_info=True,
                    )
                    raise

                response_text = response.text or ""
                combined_error_text = f"{response.status_code} {response.reason_phrase}"
                if response_text:
                    combined_error_text = f"{combined_error_text} | {response_text}"
                lowered_combined_error_text = combined_error_text.lower()
                reason_preview = combined_error_text[:1000]

                if "tpd" in lowered_combined_error_text:
                    await self._notify_kimi_alert("当前官方key出现TPD报错")
                    until_ts = await self.kimi_key_rotation.report_tpd(slot.slot_id)
                    until_dt = datetime.fromtimestamp(until_ts, ZoneInfo("Asia/Shanghai"))
                    log.error(
                        "[Kimi] Key ...%s 命中 TPD，直接封禁至次日重置: %s。reason=%s",
                        slot.key_tail,
                        until_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        reason_preview,
                    )
                    hit_tpd = True
                    break

                if response.status_code == 429 or "429 too many requests" in lowered_combined_error_text:
                    await self._notify_kimi_alert("当前官方key出现429报错")
                    stage, until_ts = await self.kimi_key_rotation.report_429(slot.slot_id)
                    until_dt = datetime.fromtimestamp(until_ts, ZoneInfo("Asia/Shanghai"))
                    if stage == "temporary_cooldown":
                        log.warning(
                            "[Kimi] Key ...%s 触发 429，临时禁用至 %s，自动切换下一个 key 重试。reason=%s",
                            slot.key_tail,
                            until_dt.strftime("%Y-%m-%d %H:%M:%S"),
                            reason_preview,
                        )
                    else:
                        log.error(
                            "[Kimi] Key ...%s 二次触发 429（TPD），已封禁至次日重置: %s。reason=%s",
                            slot.key_tail,
                            until_dt.strftime("%Y-%m-%d %H:%M:%S"),
                            reason_preview,
                        )
                    hit_429 = True
                    break

                if response.is_error:
                    response_body_preview = response_text[:4000] if response_text else ""
                    if attempt < non_429_same_key_retry_limit:
                        log.warning(
                            "[Kimi] 非429请求失败，同key自动重试 (%s/%s) | status=%s | key_tail=%s | body=%s",
                            attempt_no,
                            total_attempts,
                            response.status_code,
                            slot.key_tail,
                            response_body_preview[:1000],
                        )
                        continue

                    log.error(
                        "[Kimi] 请求失败 | status=%s | url=%s | key_tail=%s | body=%s",
                        response.status_code,
                        current_api_url,
                        slot.key_tail,
                        response_body_preview,
                    )
                    response.raise_for_status()

                await self.kimi_key_rotation.report_success(slot.slot_id)
                return (
                    response,
                    current_api_url,
                    "moonshot",
                    slot.slot_id,
                    slot.key_tail,
                    self.kimi_model,
                )

            if hit_tpd or hit_429:
                continue

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
        override_base_url: Optional[str] = None,
    ) -> str:
        """
        OpenAI 兼容专用通道（DeepSeek / Kimi）。
        """
        effective_model_name = model_name or "deepseek-chat"
        is_deepseek_model = effective_model_name in {
            "deepseek-chat",
            "deepseek-reasoner",
        }
        channel_label = "DeepSeek" if is_deepseek_model else "Kimi"

        # 选择目标网关配置
        if is_deepseek_model:
            target_base_url = self.deepseek_url
            target_api_key = self.deepseek_key
            if not (target_base_url and target_api_key):
                log.warning(f"请求使用 {effective_model_name} 但未配置 DEEPSEEK_URL 或 DEEPSEEK_API_KEY。")
                return "DeepSeek 配置缺失，请检查环境变量。"
        else:
            target_base_url = None
            target_api_key = None
            if not self.kimi_key_rotation.has_configured_keys:
                log.warning("请求使用 kimi-k2.5 但未配置 MOONSHOT_URL 或 MOONSHOT_API_KEY。")
                return "Kimi 配置缺失，请检查 MOONSHOT_URL 与 MOONSHOT_API_KEY 环境变量。"

        if not is_deepseek_model:
            log.info("[Kimi] 本轮回复将使用官方 key 轮询。")

        should_enable_kimi_web_search = False
        if not is_deepseek_model:
            should_enable_kimi_web_search = self._should_enable_kimi_web_search(
                message=message,
                replied_message=replied_message,
            )
            log.info(
                "[Kimi] 联网搜索开关=%s（仅在“上网”/URL/明确联网关键词时启用）",
                "ON" if should_enable_kimi_web_search else "OFF",
            )

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
                    log.info(f"📚 [{channel_label}] 触发世界书/成员设定，已注入 Prompt: {titles}")
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

            for tool in dynamic_tools:
                func_name = getattr(tool, "__name__", "")
                if not is_deepseek_model and func_name == "analyze_image_with_gemini_pro":
                    log.info(
                        f"[{effective_model_name}] 已禁用工具: {func_name}（仅 DeepSeek 模型可用）"
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

        kimi_web_search_tool_description = (
            "工具名称：$web_search。用途：联网检索与阅读网页内容。"
            "调用条件：仅当用户明确提及“上网”，或消息中包含链接(URL)并要求查看时允许调用。"
            "禁止条件：用户仅说“搜一下/查询一下”但未提及上网或链接时，禁止调用。"
        )

        kimi_builtin_tools: List[Dict[str, Any]] = []
        if not is_deepseek_model and should_enable_kimi_web_search:
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
        )

        openai_messages: List[Dict[str, Any]] = []
        is_first_user = True
        for turn in final_conversation:
            gemini_role = turn.get("role")

            if is_deepseek_model:
                content = await self._build_deepseek_turn_content(turn.get("parts", []) or [])
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

            content_blocks = self._build_kimi_turn_content(turn.get("parts", []) or [])
            if not content_blocks:
                continue

            if gemini_role == "model":
                assistant_text = self._extract_text_from_openai_content(content_blocks)
                if assistant_text:
                    openai_messages.append({"role": "assistant", "content": assistant_text})
            else:
                if is_first_user:
                    system_text = self._extract_text_from_openai_content(content_blocks)
                    if system_text:
                        openai_messages.append({"role": "system", "content": system_text})
                    is_first_user = False
                else:
                    openai_messages.append({"role": "user", "content": content_blocks})

        if not is_deepseek_model:
            web_search_policy_prompt = (
                "【联网工具使用规则】"
                f"{kimi_web_search_tool_description}"
                "若未满足条件，请直接基于已有知识回答，并明确说明未联网检索。"
            )
            if (
                openai_messages
                and openai_messages[0].get("role") == "system"
                and isinstance(openai_messages[0].get("content"), str)
            ):
                openai_messages[0]["content"] = (
                    f"{openai_messages[0]['content']}\n\n{web_search_policy_prompt}"
                ).strip()
            else:
                openai_messages.insert(
                    0,
                    {"role": "system", "content": web_search_policy_prompt},
                )

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
            safe_openai_messages = _sanitize_openai_payload_for_log(openai_messages)
            log.info(f"--- [{channel_label}] 完整发送上下文 (用户 {user_id}) ---")
            log.info(json.dumps(safe_openai_messages, ensure_ascii=False, indent=2, default=str))

            request_tools_for_log = (
                openai_tools if is_deepseek_model else (openai_tools + kimi_builtin_tools)
            )
            if request_tools_for_log:
                log.info(f"--- [{channel_label}] 工具列表 ---")
                log.info(
                    json.dumps(
                        request_tools_for_log, ensure_ascii=False, indent=2, default=str
                    )
                )
            log.info("------------------------------------")

        # 核心请求循环
        api_url = ""
        api_key = ""

        if is_deepseek_model:
            api_url = (override_base_url or target_base_url or "").rstrip("/")
            api_key = target_api_key or ""

            if not api_url:
                return "OpenAI 兼容通道 URL 配置缺失，请检查配置。"
            if not api_key:
                return "OpenAI 兼容通道 API Key 配置缺失，请检查配置。"

            if override_base_url:
                log.info(f"🧪 一次性调试已生效：OpenAI 兼容通道临时改用 URL: {api_url}")

            api_url = self._build_chat_completions_url(api_url)
        elif override_base_url:
            log.info(f"🧪 一次性调试已生效：Kimi 通道临时改用 URL: {override_base_url}")

        gen_config = app_config.MODEL_GENERATION_CONFIG.get(
            effective_model_name, app_config.MODEL_GENERATION_CONFIG["default"]
        )

        max_calls = 5
        called_tool_names: List[str] = []
        bad_format_retries = 0
        deep_vision_used = False
        tool_image_context_list = self._build_tool_image_context_list(images)
        switched_to_official_notified = False
        kimi_token_too_long_retry_used = False

        try:
            async with httpx.AsyncClient(timeout=120.0) as http_client:
                for i in range(max_calls):
                    request_model_name = (
                        effective_model_name if is_deepseek_model else self.kimi_model
                    )

                    payload = {
                        "model": request_model_name,
                        "messages": openai_messages,
                        "stream": False,
                        "temperature": gen_config.get("temperature", 1.3),
                        "top_p": gen_config.get("top_p", 0.95),
                        "max_tokens": gen_config.get("max_output_tokens", 8192),
                    }

                    if not is_deepseek_model:
                        payload["thinking"] = {"type": "disabled"}

                    if is_deepseek_model:
                        if openai_tools:
                            payload["tools"] = openai_tools
                    else:
                        kimi_request_tools: List[Dict[str, Any]] = []
                        if openai_tools:
                            kimi_request_tools.extend(openai_tools)
                        if kimi_builtin_tools:
                            kimi_request_tools.extend(kimi_builtin_tools)
                        if kimi_request_tools:
                            payload["tools"] = kimi_request_tools

                    used_api_url = api_url
                    used_slot_label = "deepseek"
                    used_kimi_slot_id: Optional[str] = None
                    used_kimi_key_tail = "N/A"
                    used_kimi_model_name = request_model_name

                    if is_deepseek_model:
                        response = await http_client.post(
                            api_url,
                            headers={
                                "Authorization": f"Bearer {api_key}",
                                "Content-Type": "application/json",
                            },
                            json=payload,
                        )

                        if response.is_error:
                            response_body_preview = response.text[:4000] if response.text else ""
                            log.error(
                                "[OpenAI兼容] 请求失败 | status=%s | model=%s | url=%s | body=%s",
                                response.status_code,
                                effective_model_name,
                                api_url,
                                response_body_preview,
                            )
                    else:
                        (
                            response,
                            used_api_url,
                            used_slot_label,
                            used_kimi_slot_id,
                            used_kimi_key_tail,
                            used_kimi_model_name,
                        ) = await self._post_kimi_with_rotation(
                            http_client=http_client,
                            payload=payload,
                            override_base_url=override_base_url,
                        )

                        if log_detailed:
                            log.info(
                                "[Kimi] 本次请求使用来源=%s, url=%s, model=%s",
                                used_slot_label,
                                used_api_url,
                                used_kimi_model_name,
                            )

                    if response.is_error and not is_deepseek_model:
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
                            openai_messages = self._trim_kimi_messages_for_retry(openai_messages)
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

                    try:
                        result = response.json()
                    except json.JSONDecodeError:
                        response_text = (response.text or "").strip()
                        lowered_response_text = response_text.lower()

                        if not is_deepseek_model and used_kimi_slot_id:
                            if "tpd" in lowered_response_text:
                                await self._notify_kimi_alert("当前官方key出现TPD报错")
                                until_ts = await self.kimi_key_rotation.report_tpd(used_kimi_slot_id)
                                until_dt = datetime.fromtimestamp(
                                    until_ts, ZoneInfo("Asia/Shanghai")
                                )
                                reason_preview = response_text[:1000] if response_text else "<empty>"
                                log.error(
                                    "[Kimi] Key ...%s 在响应体中命中 TPD，直接封禁至次日重置: %s。reason=%s",
                                    used_kimi_key_tail,
                                    until_dt.strftime("%Y-%m-%d %H:%M:%S"),
                                    reason_preview,
                                )
                                continue

                            if response.status_code == 429 or "429 too many requests" in lowered_response_text:
                                await self._notify_kimi_alert("当前官方key出现429报错")
                                stage, until_ts = await self.kimi_key_rotation.report_429(
                                    used_kimi_slot_id
                                )
                                until_dt = datetime.fromtimestamp(
                                    until_ts, ZoneInfo("Asia/Shanghai")
                                )
                                reason_preview = response_text[:1000] if response_text else "<empty>"
                                if stage == "temporary_cooldown":
                                    log.warning(
                                        "[Kimi] Key ...%s 在响应体中命中 429 文本，临时禁用至 %s，自动重试。reason=%s",
                                        used_kimi_key_tail,
                                        until_dt.strftime("%Y-%m-%d %H:%M:%S"),
                                        reason_preview,
                                    )
                                else:
                                    log.error(
                                        "[Kimi] Key ...%s 在响应体中二次命中 429 文本，封禁至次日重置: %s。reason=%s",
                                        used_kimi_key_tail,
                                        until_dt.strftime("%Y-%m-%d %H:%M:%S"),
                                        reason_preview,
                                    )
                                continue

                        if not is_deepseek_model:
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
                            return "Kimi 通道返回了非标准响应（非 JSON），请稍后再试。"

                        raise

                    response_message = result["choices"][0]["message"]
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
                                    r"不过话说回来|话说回来|另外|话又说回来|不过话又说回来",
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
                                            "content": "[系统提示] 检测到你使用了“不过话说回来|话说回来|另外|话又说回来”。这是被禁止的。请重新生成回复，去掉这个短语，保持语气自然。",
                                        }
                                    )
                                    bad_format_retries += 1
                                    continue
                                return "抱歉，我的说话格式一直达不到要求，我是杂鱼"
                            elif has_forbidden_phrase:
                                log.info(
                                    f"[{channel_label}] 检测到违禁词，但回复长度为 {content_len} (>800)，按成本优化策略放行。"
                                )

                        allowed_emoji_names = {
                            "开心",
                            "乖巧",
                            "害羞",
                            "偷笑",
                            "比心",
                            "desuwa",
                            "伤心",
                            "生气",
                            "加油",
                            "好奇",
                            "邀请",
                            "傲娇",
                            "祝福",
                            "你好",
                            "叹气",
                            "投降",
                        }
                        removed_emoji_tags: List[str] = []

                        def _strip_disallowed_emoji_tag(match):
                            emoji_name = match.group(1).strip()
                            if emoji_name in allowed_emoji_names:
                                return match.group(0)

                            removed_emoji_tags.append(match.group(0))
                            return ""

                        if content:
                            content = re.sub(
                                r"<([^<>\s/]{1,20})>",
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

                        return await self.post_process_response(content, user_id, guild_id)

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
                            import copy as _copy

                            clean_response = _copy.deepcopy(raw_response)

                            if isinstance(clean_response, dict):
                                result_data = clean_response.get("result", {})
                                if isinstance(result_data, dict):
                                    profile = result_data.get("profile", {})
                                    if isinstance(profile, dict):
                                        avatar_b64 = profile.get("avatar_image_base64")
                                        if isinstance(avatar_b64, str) and avatar_b64.strip():
                                            if is_deepseek_model:
                                                try:
                                                    avatar_bytes = base64.b64decode(avatar_b64)
                                                    avatar_mime_type = profile.get(
                                                        "avatar_mime_type", "image/png"
                                                    )
                                                    if (
                                                        not isinstance(avatar_mime_type, str)
                                                        or not avatar_mime_type
                                                    ):
                                                        avatar_mime_type = "image/png"

                                                    avatar_payload = {
                                                        "type": "image",
                                                        "mime_type": avatar_mime_type,
                                                        "data_size": len(avatar_bytes),
                                                        "data_preview": avatar_bytes.hex(),
                                                    }
                                                    avatar_vision_text = (
                                                        await moonshot_vision_service.recognize_image(
                                                            avatar_payload,
                                                            prompt="请识别这张用户头像图片，简洁描述可见人物、风格、配色与关键元素。",
                                                        )
                                                    )
                                                    profile["avatar_image_vision"] = avatar_vision_text
                                                except Exception as e:
                                                    log.error(
                                                        "处理 get_user_profile 头像识图失败: %s",
                                                        e,
                                                        exc_info=True,
                                                    )
                                                    profile["avatar_image_vision"] = "（头像识图失败：处理异常）"
                                                finally:
                                                    profile.pop("avatar_image_base64", None)
                                                    profile.setdefault(
                                                        "avatar_note",
                                                        "（头像原始图片数据已省略，已提供识图摘要）",
                                                    )
                                            else:
                                                profile.pop("avatar_image_base64", None)
                                                profile.setdefault(
                                                    "avatar_note",
                                                    "（头像原始图片数据已省略）",
                                                )

                            content_str = json.dumps(clean_response, ensure_ascii=False)
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
                return "哎呀，我好像陷入了一个复杂的思考循环里，换个话题聊聊吧！"

        except NoAvailableKimiKeyError:
            log.error("[Kimi] 所有 Key 当前不可用（可能均在冷却或次日封禁中）。")
            await self._notify_kimi_alert("所有key均失效")
            return "目前全部key均已失效"

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
            return f"{channel_label} 连接失败: {error_info}。详情: {short_detail}"

        except httpx.RequestError as e:
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
            return (
                f"{channel_label} 连接失败: {err_fields['exc_type']}。"
                "请检查网关地址、DNS/TLS 与网络连通性。"
            )

        except Exception as e:
            error_info = f"{type(e).__name__}: {str(e)}"
            log.error(f"{channel_label} API 调用失败: {error_info}", exc_info=True)
            return f"{channel_label} 连接失败: {error_info}。请检查日志或配置。"