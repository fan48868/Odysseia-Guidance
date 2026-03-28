# -*- coding: utf-8 -*-

import base64
import io
import logging
import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

import httpx
from PIL import Image
from google.genai import types

from src.chat.config import chat_config as app_config
from src.chat.services.kimi_key_rotation import (
    KimiKeyRotationService,
    NoAvailableKimiKeyError,
)

log = logging.getLogger(__name__)


class KimiModelClient:
    """Kimi 通道客户端：负责公益站/官方站路由、Key 轮换与 Kimi 专属多模态处理。"""

    def __init__(
        self,
        bot_getter: Optional[Callable[[], Any]] = None,
        alert_user_id: int = 1449321391412215908,
    ) -> None:
        self.bot_getter = bot_getter
        self.alert_user_id = alert_user_id

        self.moonshot_url = os.getenv("MOONSHOT_URL")
        self.moonshot_key = os.getenv("MOONSHOT_API_KEY")
        self.model_name = "kimi-k2.5"

        self.custom_moonshot_url = os.getenv("CUSTOM_MOONSHOT_URL")
        self.custom_moonshot_api_key = os.getenv("CUSTOM_MOONSHOT_API_KEY")
        self.custom_moonshot_model_name = os.getenv("CUSTOM_MOONSHOT_MODEL_NAME")
        self.custom_site_disabled = False
        self.is_nv_model = str(os.getenv("IS_NV_MODEL", "True")).strip().lower() in [
            "true",
            "1",
            "t",
            "yes",
            "y",
        ]

        self.kimi_key_rotation = KimiKeyRotationService(
            moonshot_base_url=self.moonshot_url,
            moonshot_api_key=self.moonshot_key,
        )

        if (
            self.custom_moonshot_url
            and self.custom_moonshot_api_key
            and self.custom_moonshot_model_name
        ):
            log.info(
                "✅ [KimiModelClient] 已加载 Kimi 公益站配置。URL: %s, Model: %s",
                self.custom_moonshot_url,
                self.custom_moonshot_model_name,
            )
        if self.moonshot_url and self.moonshot_key:
            log.info(
                "✅ [KimiModelClient] 已加载 Kimi 官方站（兜底）配置。URL: %s",
                self.moonshot_url,
            )

    async def notify_alert(self, content: str) -> None:
        """向指定管理员发送 Kimi 轮换告警私信（失败不影响主流程）。"""
        bot = self.bot_getter() if callable(self.bot_getter) else None
        if bot is None:
            log.warning("[KimiAlert] Bot 实例尚未注入，跳过私信告警: %s", content)
            return

        try:
            user = bot.get_user(self.alert_user_id)
            if user is None:
                user = await bot.fetch_user(self.alert_user_id)
            if user is None:
                log.warning("[KimiAlert] 未找到告警用户: %s", self.alert_user_id)
                return

            await user.send(content)
        except Exception as e:
            log.warning("[KimiAlert] 发送私信告警失败: %s", e)

    @property
    def has_configured_keys(self) -> bool:
        return self.kimi_key_rotation.has_configured_keys

    def get_validation_error(self) -> Optional[str]:
        if self.has_configured_keys:
            return None

        log.warning("请求使用 kimi-k2.5 但未配置 MOONSHOT_URL 或 MOONSHOT_API_KEY。")
        return "Kimi 配置缺失，请检查 MOONSHOT_URL 与 MOONSHOT_API_KEY 环境变量。"

    def get_request_model_name(self) -> str:
        return self.model_name

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
    def should_enable_web_search(
        message: Optional[str], replied_message: Optional[str] = None
    ) -> bool:
        text_parts = []
        if isinstance(message, str) and message.strip():
            text_parts.append(message.strip())
        if isinstance(replied_message, str) and replied_message.strip():
            text_parts.append(replied_message.strip())

        if not text_parts:
            return False

        combined_text = " ".join(text_parts)
        lowered_text = combined_text.lower()

        if "网" in combined_text:
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

    def resolve_web_search_enabled(
        self, message: Optional[str], replied_message: Optional[str] = None
    ) -> bool:
        enabled = self.should_enable_web_search(
            message=message,
            replied_message=replied_message,
        )
        log.info(
            "[Kimi] 联网搜索开关=%s（仅在“上网”/URL/明确联网关键词时启用）",
            "ON" if enabled else "OFF",
        )
        return enabled

    @staticmethod
    def trim_messages_for_retry(
        messages: List[Dict[str, Any]], keep_recent_non_system: int = 24
    ) -> List[Dict[str, Any]]:
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
    def extract_text_from_openai_content(content: Union[str, List[Dict[str, Any]]]) -> str:
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

    def build_turn_content(self, parts: List[Any]) -> List[Dict[str, Any]]:
        """
        构建 Kimi (OpenAI 兼容多模态) 单条消息 content。
        直接把图片作为 image_url(data URI) 发给模型，不做 OCR。
        """
        content_blocks: List[Dict[str, Any]] = []

        for part in parts or []:
            if hasattr(part, "thought") and getattr(part, "thought", False):
                continue

            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text_value = part["text"].strip()
                if text_value:
                    content_blocks.append({"type": "text", "text": text_value})
                continue

            if isinstance(part, types.Part):
                if part.text:
                    text_value = part.text.strip()
                    if text_value:
                        content_blocks.append({"type": "text", "text": text_value})
                    continue

                if part.inline_data and part.inline_data.data:
                    mime_type = part.inline_data.mime_type or "image/png"
                    image_bytes = bytes(part.inline_data.data)

                    if mime_type.lower() == "image/gif":
                        max_gif_size_mb = float(
                            app_config.IMAGE_PROCESSING_CONFIG.get("MAX_GIF_SIZE_MB", 8)
                        )
                        max_gif_size_bytes = int(max_gif_size_mb * 1024 * 1024)
                        if len(image_bytes) > max_gif_size_bytes:
                            content_blocks.append(
                                {
                                    "type": "text",
                                    "text": "（收到一张 GIF，但体积超过限制，已跳过）",
                                }
                            )
                            continue

                    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                    content_blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                        }
                    )
                    continue

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
                    if mime_type.lower() == "image/gif":
                        source = str(part.get("source", "attachment"))
                        image_cfg = app_config.IMAGE_PROCESSING_CONFIG
                        if source == "emoji":
                            max_gif_size_mb = float(
                                image_cfg.get("MAX_ANIMATED_EMOJI_SIZE_MB", 2)
                            )
                        else:
                            max_gif_size_mb = float(image_cfg.get("MAX_GIF_SIZE_MB", 8))
                        max_gif_size_bytes = int(max_gif_size_mb * 1024 * 1024)

                        if len(image_bytes) > max_gif_size_bytes:
                            content_blocks.append(
                                {
                                    "type": "text",
                                    "text": "（收到一张 GIF，但体积超过限制，已跳过）",
                                }
                            )
                            continue

                    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                    content_blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                        }
                    )
                else:
                    content_blocks.append({"type": "text", "text": "（收到一张图片，但解析失败）"})
                continue

            fallback_text = str(part).strip()
            if fallback_text:
                content_blocks.append({"type": "text", "text": fallback_text})

        return content_blocks

    @staticmethod
    def _extract_video_bytes_from_dict(video: Dict[str, Any]) -> Optional[bytes]:
        direct_bytes = video.get("data") or video.get("bytes")
        if isinstance(direct_bytes, (bytes, bytearray)):
            return bytes(direct_bytes)

        base64_candidates = [
            video.get("video_base64"),
            video.get("data_base64"),
            video.get("data_preview"),
        ]
        for candidate in base64_candidates:
            if not isinstance(candidate, str) or not candidate.strip():
                continue
            raw_value = candidate.strip()
            try:
                return base64.b64decode(raw_value)
            except Exception:
                try:
                    return bytes.fromhex(raw_value)
                except Exception:
                    continue

        return None

    def build_video_content_blocks(
        self, videos: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        if not videos:
            return []

        config = app_config.VIDEO_PROCESSING_CONFIG
        max_videos = int(config.get("MAX_VIDEOS_PER_MESSAGE", 1))
        max_size_mb = float(config.get("MAX_VIDEO_SIZE_MB", 20))
        max_size_bytes = int(max_size_mb * 1024 * 1024)
        allowed_mime_types = {
            str(m).lower() for m in config.get("ALLOWED_VIDEO_MIME_TYPES", set())
        }

        content_blocks: List[Dict[str, Any]] = []
        for idx, video in enumerate(videos[:max_videos], start=1):
            if not isinstance(video, dict):
                continue

            mime_type = str(video.get("mime_type", "video/mp4")).lower()
            if allowed_mime_types and mime_type not in allowed_mime_types:
                log.warning(
                    "[Kimi] 视频 MIME 不在白名单内，已跳过第 %s 个视频: %s",
                    idx,
                    mime_type,
                )
                continue

            video_bytes = self._extract_video_bytes_from_dict(video)
            if not video_bytes:
                log.warning("[Kimi] 第 %s 个视频解析失败，已跳过。", idx)
                continue

            if len(video_bytes) > max_size_bytes:
                log.warning(
                    "[Kimi] 第 %s 个视频大小超限，已跳过: %s bytes > %s bytes",
                    idx,
                    len(video_bytes),
                    max_size_bytes,
                )
                continue

            try:
                video_b64 = base64.b64encode(video_bytes).decode("utf-8")
            except Exception as e:
                log.warning("[Kimi] 第 %s 个视频 Base64 编码失败，已跳过: %s", idx, e)
                continue

            content_blocks.append(
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:{mime_type};base64,{video_b64}"},
                }
            )

        return content_blocks

    async def _post_kimi_with_rotation(
        self,
        http_client: httpx.AsyncClient,
        payload: Dict[str, Any],
        user_id: int,
        override_base_url: Optional[str] = None,
    ) -> Tuple[httpx.Response, str, str, str, str, str]:
        max_non_429_attempts = 3
        while True:
            slot = await self.kimi_key_rotation.acquire_active_slot(user_id=user_id)
            available_count, total_count = await self.kimi_key_rotation.get_pool_stats()
            current_base_url = (override_base_url or slot.base_url or "").rstrip("/")
            current_api_url = self._build_chat_completions_url(current_base_url)

            request_payload = dict(payload)
            request_payload["model"] = self.model_name
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

            total_attempts = max_non_429_attempts
            hit_429 = False
            hit_tpd = False

            def _retry_non_429_if_needed(attempt_no: int, reason: str) -> bool:
                if attempt_no < total_attempts:
                    log.warning(
                        "[Kimi] 非429网络异常，同key自动重试 (%s/%s) | key_tail=%s | reason=%s",
                        attempt_no,
                        total_attempts,
                        slot.key_tail,
                        reason,
                    )
                    return True
                return False

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
                    if _retry_non_429_if_needed(attempt_no, str(e)):
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

                is_429 = (
                    response.status_code == 429
                    or "429 too many requests" in lowered_combined_error_text
                )
                is_tpd = "tpd" in lowered_combined_error_text

                if is_429 and is_tpd:
                    until_ts = await self.kimi_key_rotation.report_tpd(slot.slot_id)
                    until_dt = datetime.fromtimestamp(until_ts, ZoneInfo("Asia/Shanghai"))
                    tpd_alert_msg = (
                        f"当前key触发TPD，已被封禁至 {until_dt.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    await self.notify_alert(tpd_alert_msg)
                    log.error(
                        "[Kimi] Key ...%s 命中 429 及 TPD，已封禁至次日重置。reason=%s",
                        slot.key_tail,
                        reason_preview,
                    )
                    hit_tpd = True
                    break

                if is_429:
                    # 检测余额耗尽
                    if "balance" in lowered_combined_error_text:
                        await self.notify_alert("当前key余额已耗尽")
                    
                    _, until_ts = await self.kimi_key_rotation.report_429(slot.slot_id)
                    until_dt = datetime.fromtimestamp(until_ts, ZoneInfo("Asia/Shanghai"))
                    log.warning(
                        "[Kimi] Key ...%s 触发 429（无TPD），临时禁用至 %s，自动切换下一个 key 重试。reason=%s",
                        slot.key_tail,
                        until_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        reason_preview,
                    )
                    hit_429 = True
                    break

                if response.is_error:
                    if _retry_non_429_if_needed(attempt_no, reason_preview):
                        continue

                    response_body_preview = response_text[:4000] if response_text else ""
                    log.error(
                        "[Kimi] 非429请求失败，已达到最大重试次数 | status=%s | url=%s | key_tail=%s | body=%s",
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
                    self.model_name,
                )

            if hit_tpd or hit_429:
                continue

    async def send(
        self,
        http_client: httpx.AsyncClient,
        payload: Dict[str, Any],
        user_id: int,
        override_base_url: Optional[str] = None,
        log_detailed: bool = False,
        skip_custom_site_for_this_turn: bool = False,
    ) -> Dict[str, Any]:
        fallback_to_official = False

        if (
            self.custom_moonshot_url
            and self.custom_moonshot_api_key
            and not self.custom_site_disabled
            and not skip_custom_site_for_this_turn
        ):
            log.info("[Kimi] 尝试使用公益站...")
            custom_api_url = self._build_chat_completions_url(self.custom_moonshot_url)

            custom_payload = dict(payload)
            custom_payload["model"] = self.custom_moonshot_model_name or self.model_name

            if self.is_nv_model:
                custom_payload["chat_template_kwargs"] = {"thinking": False}
                custom_payload.pop("thinking", None)
                if log_detailed:
                    log.info(
                        "[Kimi] 公益站 Payload Thinking Check (NV Mode): %s",
                        custom_payload.get("chat_template_kwargs"),
                    )
            else:
                custom_payload["thinking"] = {"type": "disabled"}
                custom_payload.pop("chat_template_kwargs", None)
                if log_detailed:
                    log.info(
                        "[Kimi] 公益站 Payload Thinking Check (Official Mode): %s",
                        custom_payload.get("thinking"),
                    )

            try:
                response = await http_client.post(
                    custom_api_url,
                    headers={
                        "Authorization": f"Bearer {self.custom_moonshot_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=custom_payload,
                )
                response_text = response.text or ""

                if "用户额度不足" in response_text:
                    self.custom_site_disabled = True
                    log.error("[Kimi] 公益站因“用户额度不足”已被禁用。")
                    await self.notify_alert("公益站余额不足，已被禁用")
                    fallback_to_official = True

                elif response.is_error:
                    log.warning(
                        "[Kimi] 公益站请求失败 (status: %s)，将使用官方 Key。Reason: %s",
                        response.status_code,
                        response_text[:500],
                    )
                    await self.notify_alert("公益站报错，本次使用官方key")
                    fallback_to_official = True
                    skip_custom_site_for_this_turn = True

                else:
                    log.info("[Kimi] 公益站调用成功。")
                    return {
                        "response": response,
                        "used_api_url": custom_api_url,
                        "used_slot_label": "custom",
                        "used_slot_id": "custom",
                        "used_key_tail": self.custom_moonshot_api_key[-4:],
                        "used_model_name": custom_payload["model"],
                        "skip_custom_site_for_this_turn": skip_custom_site_for_this_turn,
                    }

            except httpx.RequestError as e:
                log.warning("[Kimi] 公益站网络请求失败，将使用官方 Key。Reason: %s", e)
                await self.notify_alert("公益站报错，本次使用官方key")
                fallback_to_official = True
                skip_custom_site_for_this_turn = True

        else:
            fallback_to_official = True

        if fallback_to_official:
            log.info("[Kimi] Fallback 到官方站...")
            official_payload = dict(payload)
            official_payload["model"] = self.model_name
            (
                response,
                used_api_url,
                used_slot_label,
                used_slot_id,
                used_key_tail,
                used_model_name,
            ) = await self._post_kimi_with_rotation(
                http_client=http_client,
                payload=official_payload,
                user_id=user_id,
                override_base_url=override_base_url,
            )

            return {
                "response": response,
                "used_api_url": used_api_url,
                "used_slot_label": used_slot_label,
                "used_slot_id": used_slot_id,
                "used_key_tail": used_key_tail,
                "used_model_name": used_model_name,
                "skip_custom_site_for_this_turn": skip_custom_site_for_this_turn,
            }

        raise NoAvailableKimiKeyError("Kimi 请求路由异常，未获取到可用响应。")