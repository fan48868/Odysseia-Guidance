# -*- coding: utf-8 -*-

import asyncio
import base64
import copy
import io
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx
from PIL import Image
from dotenv import set_key

from src import config
from src.chat.services.moonshot_vision_service import moonshot_vision_service
from src.chat.utils.custom_model_api_keys import (
    get_custom_model_api_key_raw_value,
    persist_custom_model_api_keys_to_file,
    resolve_custom_model_api_keys,
    serialize_custom_model_api_keys,
    split_custom_model_inline_api_keys,
)

log = logging.getLogger(__name__)


class CustomModelClient:
    _PRIMARY_API_KEY_ENV_NAME = "CUSTOM_MODEL_API_KEY"
    _LEGACY_API_KEY_ENV_NAME = "CUSTON_MODEL_API_KEY"
    """Custom OpenAI 兼容通道客户端：负责配置管理、请求发送与 Custom 专属解析。"""

    def __init__(self) -> None:
        self._api_keys: List[str] = []
        self._active_api_key_index = 0
        self._key_rotation_lock = asyncio.Lock()
        self._provider_order_rotation_lock = asyncio.Lock()
        self._provider_order_offset_by_model: Dict[str, int] = {}
        self.api_key: Optional[str] = None
        self.raw_api_key: Optional[str] = None
        self._api_key_source_type = "inline"
        self._api_key_file_path: Optional[str] = None
        self._gateway_options_log_once: set[str] = set()
        runtime_config = self.refresh_from_env()

        if (
            runtime_config["base_url"]
            and runtime_config["api_key"]
            and runtime_config["model_name"]
        ):
            log.info(
                "✅ [CustomModelClient] 已加载 Custom OpenAI 配置。URL: %s, Model: %s, Vision: %s, Video Input: %s",
                runtime_config["base_url"],
                runtime_config["model_name"],
                runtime_config["enable_vision"],
                runtime_config["enable_video_input"],
            )

    @staticmethod
    def _normalize_bool_flag(raw_value: Optional[str]) -> bool:
        return str(raw_value or "").strip().lower() in {"true", "1", "yes"}

    @staticmethod
    def _normalize_positive_float(
        raw_value: Optional[str], *, default: float
    ) -> float:
        try:
            parsed = float(str(raw_value or "").strip())
            if parsed > 0:
                return parsed
        except (TypeError, ValueError):
            pass
        return default

    @staticmethod
    def _normalize_positive_int(
        raw_value: Optional[str], *, default: int
    ) -> int:
        try:
            parsed = int(str(raw_value or "").strip())
            if parsed > 0:
                return parsed
        except (TypeError, ValueError):
            pass
        return default

    @staticmethod
    def _clamp_int(value: int, *, minimum: int, maximum: int) -> int:
        return max(minimum, min(maximum, int(value)))

    @staticmethod
    def _split_api_keys(raw_api_keys: Optional[str]) -> List[str]:
        return list(split_custom_model_inline_api_keys(raw_api_keys))

    @staticmethod
    def _mask_key_tail(api_key: Optional[str]) -> str:
        if not api_key:
            return "N/A"
        return api_key[-4:]

    @staticmethod
    def _serialize_api_keys(api_keys: List[str]) -> str:
        return serialize_custom_model_api_keys(api_keys)

    @staticmethod
    def _get_env_path() -> str:
        return os.path.join(config.BASE_DIR, ".env")

    def _apply_api_key_state(
        self,
        api_keys: List[str],
        *,
        active_index: int = 0,
        raw_api_key_source: Optional[str] = None,
        source_type: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> str:
        normalized_keys = self._split_api_keys(self._serialize_api_keys(api_keys))
        serialized_api_keys = self._serialize_api_keys(normalized_keys)
        self._api_keys = normalized_keys

        if raw_api_key_source is not None:
            normalized_source = str(raw_api_key_source or "").strip()
            self.raw_api_key = normalized_source or None
        elif self.raw_api_key is None:
            self.raw_api_key = serialized_api_keys or None

        if source_type is not None:
            normalized_source_type = str(source_type or "").strip().lower()
            self._api_key_source_type = normalized_source_type or "inline"

        if file_path is not None:
            normalized_file_path = str(file_path or "").strip()
            self._api_key_file_path = normalized_file_path or None

        if not normalized_keys:
            self._active_api_key_index = 0
            self.api_key = None
            return serialized_api_keys

        normalized_index = min(max(active_index, 0), len(normalized_keys) - 1)
        self._active_api_key_index = normalized_index
        self.api_key = normalized_keys[normalized_index]
        return serialized_api_keys

    @classmethod
    def _persist_api_keys_to_env(cls, serialized_api_keys: str) -> bool:
        os.environ[cls._PRIMARY_API_KEY_ENV_NAME] = serialized_api_keys
        if cls._LEGACY_API_KEY_ENV_NAME in os.environ:
            os.environ[cls._LEGACY_API_KEY_ENV_NAME] = serialized_api_keys

        env_path = cls._get_env_path()
        try:
            os.makedirs(os.path.dirname(env_path), exist_ok=True)
            set_key(
                env_path,
                cls._PRIMARY_API_KEY_ENV_NAME,
                serialized_api_keys,
                quote_mode="always",
                encoding="utf-8",
            )
            return True
        except Exception as exc:
            log.warning(
                "[Custom] Failed to persist %s to .env: %s",
                cls._PRIMARY_API_KEY_ENV_NAME,
                exc,
                exc_info=True,
            )
            return False

    @staticmethod
    def _is_payment_required_error(
        status_code: int, combined_error_text: str
    ) -> bool:
        lowered_error_text = str(combined_error_text or "").lower()
        return status_code == 402 or "402 payment required" in lowered_error_text

    def _sync_api_key_state(
        self,
        api_keys: List[str],
        *,
        raw_api_key_source: Optional[str] = None,
        source_type: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> List[str]:
        keys = self._split_api_keys(self._serialize_api_keys(api_keys))
        current_key: Optional[str] = None
        normalized_source = str(raw_api_key_source or "").strip()

        if self._api_keys and 0 <= self._active_api_key_index < len(self._api_keys):
            current_key = self._api_keys[self._active_api_key_index]

        if not keys:
            self._apply_api_key_state(
                [],
                active_index=0,
                raw_api_key_source=normalized_source or None,
                source_type=source_type,
                file_path=file_path,
            )
            return self._api_keys

        if normalized_source and self.raw_api_key == normalized_source and self._api_keys:
            reordered_keys = [key for key in self._api_keys if key in keys]
            reordered_keys.extend(key for key in keys if key not in reordered_keys)
            keys = reordered_keys

        if current_key in keys:
            active_index = keys.index(current_key)
        else:
            active_index = 0

        self._apply_api_key_state(
            keys,
            active_index=active_index,
            raw_api_key_source=normalized_source or None,
            source_type=source_type,
            file_path=file_path,
        )
        return self._api_keys

    def _get_active_api_key_snapshot(
        self, runtime_config: Dict[str, Any]
    ) -> Tuple[str, int, int]:
        keys = self._sync_api_key_state(
            list(runtime_config.get("api_keys") or []),
            raw_api_key_source=runtime_config.get("api_key_source_value"),
            source_type=runtime_config.get("api_key_source_type"),
            file_path=runtime_config.get("api_key_file_path"),
        )
        if not keys:
            return "", 0, 0

        active_index = min(max(self._active_api_key_index, 0), len(keys) - 1)
        active_key = keys[active_index]
        self._active_api_key_index = active_index
        self.api_key = active_key
        return active_key, active_index, len(keys)

    async def _rotate_to_next_api_key(
        self,
        *,
        failed_api_key: str,
        failed_index: int,
        total_keys: int,
        reason_preview: str,
        runtime_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if total_keys <= 1:
            return False

        async with self._key_rotation_lock:
            if len(self._api_keys) <= 1:
                return False

            active_index = min(
                max(self._active_api_key_index, 0), len(self._api_keys) - 1
            )
            active_key = self._api_keys[active_index]

            if active_key == failed_api_key and active_index == failed_index:
                reordered_keys = list(self._api_keys)
                failed_key = reordered_keys.pop(active_index)
                reordered_keys.append(failed_key)
                next_index = active_index if active_index < len(reordered_keys) - 1 else 0
                serialized_api_keys = self._apply_api_key_state(
                    reordered_keys,
                    active_index=next_index,
                )
                if runtime_config is not None:
                    runtime_config["api_key"] = serialized_api_keys
                    runtime_config["api_keys"] = list(self._api_keys)
                source_type = str(
                    (runtime_config or {}).get("api_key_source_type")
                    or self._api_key_source_type
                ).lower()
                if source_type == "file":
                    file_path = str(
                        (runtime_config or {}).get("api_key_file_path")
                        or self._api_key_file_path
                        or ""
                    ).strip()
                    try:
                        persist_custom_model_api_keys_to_file(file_path, self._api_keys)
                        persisted = True
                        persist_note: Any = f"file:{file_path}"
                    except Exception as exc:
                        persisted = False
                        persist_note = f"file_failed:{type(exc).__name__}"
                        log.warning(
                            "[Custom] Failed to persist rotated API key order to file %s: %s",
                            file_path or "<empty>",
                            exc,
                            exc_info=True,
                        )
                else:
                    persisted = self._persist_api_keys_to_env(serialized_api_keys)
                    persist_note = persisted
                next_key = self.api_key or ""
                log.warning(
                    "[Custom] Key ...%s hit 402 Payment Required, moved it to the end and switched to key ...%s (%s/%s) | persisted=%s | reason=%s",
                    self._mask_key_tail(failed_api_key),
                    self._mask_key_tail(next_key),
                    next_index + 1,
                    len(self._api_keys),
                    persist_note,
                    reason_preview,
                )
                return True

            self.api_key = active_key
            if runtime_config is not None:
                runtime_config["api_key"] = self._serialize_api_keys(self._api_keys)
                runtime_config["api_keys"] = list(self._api_keys)
            log.info(
                "[Custom] Key ...%s hit 402 Payment Required, but active key is already ...%s; will retry with the current active key.",
                self._mask_key_tail(failed_api_key),
                self._mask_key_tail(active_key),
            )
            return True

    @classmethod
    def get_runtime_config(cls) -> Dict[str, Any]:
        base_url = str(os.environ.get("CUSTOM_MODEL_URL", "") or "").strip()
        raw_api_key = get_custom_model_api_key_raw_value(
            primary_env_name=cls._PRIMARY_API_KEY_ENV_NAME,
            legacy_env_name=cls._LEGACY_API_KEY_ENV_NAME,
        )
        api_key_resolution_error: Optional[str] = None
        resolved_api_keys: List[str] = []
        api_key_source_type = "inline"
        api_key_file_path: Optional[str] = None

        try:
            resolved_api_key_config = resolve_custom_model_api_keys(raw_api_key)
            resolved_api_keys = list(resolved_api_key_config.api_keys)
            api_key_source_type = resolved_api_key_config.source_type
            api_key_file_path = resolved_api_key_config.file_path
        except ValueError as exc:
            api_key_resolution_error = str(exc)

        api_key = cls._serialize_api_keys(resolved_api_keys)
        model_name = str(os.environ.get("CUSTOM_MODEL_NAME", "") or "").strip()
        enable_vision = cls._normalize_bool_flag(
            os.environ.get("CUSTOM_MODEL_ENABLE_VISION")
        )
        enable_video_input = cls._normalize_bool_flag(
            os.environ.get("CUSTOM_MODEL_ENABLE_VIDEO_INPUT")
        )
        gateway_provider_timeout_ms = cls._clamp_int(
            cls._normalize_positive_int(
                os.environ.get("CUSTOM_MODEL_GATEWAY_PROVIDER_TIMEOUT_MS"),
                default=6000,
            ),
            minimum=1000,
            maximum=789000,
        )
        gateway_provider_name = str(
            os.environ.get("CUSTOM_MODEL_GATEWAY_PROVIDER_NAME", "") or ""
        ).strip()
        stream_idle_timeout_ms_raw = os.environ.get("CUSTOM_MODEL_STREAM_IDLE_TIMEOUT_MS")
        if stream_idle_timeout_ms_raw is not None:
            stream_idle_timeout_seconds = (
                cls._normalize_positive_float(
                    stream_idle_timeout_ms_raw,
                    default=5000.0,
                )
                / 1000.0
            )
        else:
            stream_idle_timeout_seconds = cls._normalize_positive_float(
                os.environ.get("CUSTOM_MODEL_STREAM_IDLE_TIMEOUT_SECONDS"),
                default=5.0,
            )
        return {
            "base_url": base_url,
            "api_key": api_key,
            "api_keys": resolved_api_keys,
            "api_key_source_type": api_key_source_type,
            "api_key_source_value": raw_api_key,
            "api_key_file_path": api_key_file_path,
            "api_key_error": api_key_resolution_error,
            "model_name": model_name,
            "enable_vision": enable_vision,
            "enable_video_input": enable_video_input,
            "gateway_provider_timeout_ms": gateway_provider_timeout_ms,
            "gateway_provider_name": gateway_provider_name,
            "stream_idle_timeout_seconds": stream_idle_timeout_seconds,
        }

    def refresh_from_env(self) -> Dict[str, Any]:
        runtime_config = self.get_runtime_config()
        effective_api_keys = self._sync_api_key_state(
            list(runtime_config.get("api_keys") or []),
            raw_api_key_source=runtime_config.get("api_key_source_value"),
            source_type=runtime_config.get("api_key_source_type"),
            file_path=runtime_config.get("api_key_file_path"),
        )
        runtime_config["api_keys"] = list(effective_api_keys)
        runtime_config["api_key"] = self._serialize_api_keys(effective_api_keys)
        self.base_url = runtime_config["base_url"] or None
        self.model_name = runtime_config["model_name"] or None
        self.enable_vision = bool(runtime_config["enable_vision"])
        self.enable_video_input = bool(runtime_config["enable_video_input"])
        self.gateway_provider_timeout_ms = int(
            runtime_config["gateway_provider_timeout_ms"]
        )
        self.gateway_provider_name = (
            str(runtime_config.get("gateway_provider_name", "") or "").strip() or None
        )
        self.stream_idle_timeout_seconds = float(
            runtime_config["stream_idle_timeout_seconds"]
        )
        return runtime_config

    @staticmethod
    def _build_chat_completions_url(base_url: str) -> str:
        normalized = (base_url or "").rstrip("/")
        if not normalized.endswith("/chat/completions"):
            normalized += "/chat/completions"
        return normalized

    @staticmethod
    def _resolve_gateway_provider_name(
        model_name: str, configured_provider_name: Optional[str] = None
    ) -> str:
        explicit = str(configured_provider_name or "").strip().lower()
        if explicit:
            return explicit

        model = str(model_name or "").strip().lower()
        if "/" not in model:
            return ""

        return model.split("/", 1)[0].strip()

    async def _get_rotated_provider_order(
        self, *, model_key: str, base_order: List[str]
    ) -> List[str]:
        if not base_order:
            return []

        normalized_key = str(model_key or "").strip().lower()
        if not normalized_key:
            return list(base_order)

        async with self._provider_order_rotation_lock:
            offset = self._provider_order_offset_by_model.get(normalized_key, 0)

        normalized_offset = offset % len(base_order)
        return base_order[normalized_offset:] + base_order[:normalized_offset]

    async def _advance_rotated_provider_order(
        self, *, model_key: str, base_order: List[str]
    ) -> List[str]:
        if not base_order:
            return []

        normalized_key = str(model_key or "").strip().lower()
        if not normalized_key:
            return list(base_order)

        async with self._provider_order_rotation_lock:
            current_offset = self._provider_order_offset_by_model.get(normalized_key, 0)
            next_offset = (current_offset + 1) % len(base_order)
            self._provider_order_offset_by_model[normalized_key] = next_offset

        return base_order[next_offset:] + base_order[:next_offset]

    @staticmethod
    def _has_meaningful_sse_value(
        value: Any, *, ignored_keys: Optional[set[str]] = None
    ) -> bool:
        ignored = ignored_keys or set()

        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, list):
            return any(
                CustomModelClient._has_meaningful_sse_value(
                    item, ignored_keys=ignored
                )
                for item in value
            )
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key) in ignored:
                    continue
                if CustomModelClient._has_meaningful_sse_value(
                    item, ignored_keys=ignored
                ):
                    return True
            return False

        return True

    @classmethod
    def _has_meaningful_sse_output(
        cls, payload: Dict[str, Any]
    ) -> bool:
        choices = payload.get("choices")
        if not isinstance(choices, list):
            return False

        for choice in choices:
            if not isinstance(choice, dict):
                continue

            for block_key in ("delta", "message"):
                block = choice.get(block_key)
                if not isinstance(block, dict):
                    continue

                if isinstance(block.get("content"), str) and block["content"].strip():
                    return True
                if cls._has_meaningful_sse_value(
                    block.get("content"),
                    ignored_keys={"type"},
                ):
                    return True
                if (
                    isinstance(block.get("reasoning_content"), str)
                    and block["reasoning_content"].strip()
                ):
                    return True
                if isinstance(block.get("tool_calls"), list) and block["tool_calls"]:
                    return True
                if cls._has_meaningful_sse_value(
                    block,
                    ignored_keys={"role", "type", "index"},
                ):
                    return True

        return False

    @classmethod
    def _is_meaningful_sse_data_line(cls, line: str) -> bool:
        stripped = str(line or "").strip()
        if not stripped.startswith("data:"):
            return False

        payload_str = stripped[5:].strip()
        if not payload_str or payload_str == "[DONE]":
            return False

        try:
            payload = json.loads(payload_str)
        except json.JSONDecodeError:
            return False

        return cls._has_meaningful_sse_output(payload)

    @staticmethod
    def _build_sse_log_sample(line: str, *, max_length: int = 240) -> str:
        stripped = str(line or "").strip()
        if not stripped:
            return ""

        sample = stripped
        if stripped.startswith("data:"):
            payload_str = stripped[5:].strip()
            if payload_str and payload_str != "[DONE]":
                try:
                    sample = json.dumps(
                        json.loads(payload_str),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                except json.JSONDecodeError:
                    sample = payload_str

        sample = re.sub(r"\s+", " ", sample).strip()
        if len(sample) > max_length:
            sample = sample[: max_length - 3] + "..."
        return sample

    @staticmethod
    def _build_buffered_response(
        response: httpx.Response, body_text: str
    ) -> httpx.Response:
        return httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            content=body_text.encode(response.encoding or "utf-8"),
            request=response.request,
            history=list(response.history),
            extensions=dict(response.extensions),
        )

    @staticmethod
    def _extract_text_from_openai_content(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()

        if not isinstance(content, list):
            return ""

        text_chunks: List[str] = []

        for block in content:
            if not isinstance(block, dict):
                continue

            block_type = str(block.get("type") or "").strip().lower()
            block_text = block.get("text")
            if isinstance(block_text, str) and (
                not block_type or "text" in block_type
            ):
                normalized_text = block_text.strip()
                if normalized_text:
                    text_chunks.append(normalized_text)
                continue

            nested_content = block.get("content")
            if isinstance(nested_content, str) and nested_content.strip():
                text_chunks.append(nested_content.strip())

        return "\n".join(text_chunks).strip()

    @classmethod
    def diagnose_streaming_chat_completion_body(cls, body: str) -> Dict[str, Any]:
        normalized_body = str(body or "")
        diagnostics: Dict[str, Any] = {
            "body_length": len(normalized_body),
            "body_is_empty": not bool(normalized_body.strip()),
            "non_empty_line_count": 0,
            "comment_line_count": 0,
            "non_data_line_count": 0,
            "data_line_count": 0,
            "empty_data_line_count": 0,
            "done_line_count": 0,
            "json_payload_count": 0,
            "json_decode_error_count": 0,
            "error_payload_count": 0,
            "choices_payload_count": 0,
            "delta_content_chunk_count": 0,
            "message_content_chunk_count": 0,
            "reasoning_chunk_count": 0,
            "tool_call_chunk_count": 0,
            "response_id": None,
            "model": None,
            "latest_finish_reason": None,
            "data_line_samples": [],
            "non_data_line_samples": [],
            "first_json_error": None,
            "first_error_payload": None,
        }

        for line_number, raw_line in enumerate(normalized_body.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue

            diagnostics["non_empty_line_count"] += 1

            if line.startswith(":"):
                diagnostics["comment_line_count"] += 1
                continue

            if not line.startswith("data:"):
                diagnostics["non_data_line_count"] += 1
                if len(diagnostics["non_data_line_samples"]) < 5:
                    diagnostics["non_data_line_samples"].append(
                        {"line": line_number, "sample": cls._build_sse_log_sample(line)}
                    )
                continue

            diagnostics["data_line_count"] += 1

            payload_str = line[5:].strip()
            if not payload_str:
                diagnostics["empty_data_line_count"] += 1
                continue

            if payload_str == "[DONE]":
                diagnostics["done_line_count"] += 1
                continue

            if len(diagnostics["data_line_samples"]) < 5:
                diagnostics["data_line_samples"].append(
                    {"line": line_number, "sample": cls._build_sse_log_sample(line)}
                )

            try:
                payload = json.loads(payload_str)
            except json.JSONDecodeError as exc:
                diagnostics["json_decode_error_count"] += 1
                if diagnostics["first_json_error"] is None:
                    diagnostics["first_json_error"] = {
                        "line": line_number,
                        "message": exc.msg,
                        "column": exc.colno,
                        "sample": cls._build_sse_log_sample(line),
                    }
                continue

            diagnostics["json_payload_count"] += 1

            if not isinstance(payload, dict):
                continue

            if diagnostics["response_id"] is None and isinstance(payload.get("id"), str):
                diagnostics["response_id"] = payload["id"]
            if diagnostics["model"] is None and isinstance(payload.get("model"), str):
                diagnostics["model"] = payload["model"]

            error_payload = payload.get("error")
            if error_payload is not None:
                diagnostics["error_payload_count"] += 1
                if diagnostics["first_error_payload"] is None:
                    diagnostics["first_error_payload"] = error_payload

            choices = payload.get("choices")
            if not isinstance(choices, list) or not choices:
                continue

            diagnostics["choices_payload_count"] += 1

            first_choice = choices[0]
            if not isinstance(first_choice, dict):
                continue

            finish_reason = first_choice.get("finish_reason")
            if isinstance(finish_reason, str) and finish_reason:
                diagnostics["latest_finish_reason"] = finish_reason

            delta = first_choice.get("delta")
            if isinstance(delta, dict):
                if cls._extract_text_from_openai_content(delta.get("content")):
                    diagnostics["delta_content_chunk_count"] += 1
                if cls._extract_text_from_openai_content(
                    delta.get("reasoning_content")
                ):
                    diagnostics["reasoning_chunk_count"] += 1
                elif (
                    isinstance(delta.get("reasoning_content"), str)
                    and delta["reasoning_content"].strip()
                ):
                    diagnostics["reasoning_chunk_count"] += 1

                delta_tool_calls = delta.get("tool_calls")
                if isinstance(delta_tool_calls, list) and delta_tool_calls:
                    diagnostics["tool_call_chunk_count"] += len(delta_tool_calls)

            message_block = first_choice.get("message")
            if isinstance(message_block, dict):
                if cls._extract_text_from_openai_content(message_block.get("content")):
                    diagnostics["message_content_chunk_count"] += 1
                if cls._extract_text_from_openai_content(
                    message_block.get("reasoning_content")
                ):
                    diagnostics["reasoning_chunk_count"] += 1
                elif (
                    isinstance(message_block.get("reasoning_content"), str)
                    and message_block["reasoning_content"].strip()
                ):
                    diagnostics["reasoning_chunk_count"] += 1

                message_tool_calls = message_block.get("tool_calls")
                if isinstance(message_tool_calls, list) and message_tool_calls:
                    diagnostics["tool_call_chunk_count"] += len(message_tool_calls)

        return diagnostics

    @classmethod
    def explain_streaming_chat_completion_parse_failure(
        cls, body: str
    ) -> Tuple[str, Dict[str, Any]]:
        diagnostics = cls.diagnose_streaming_chat_completion_body(body)

        if diagnostics["body_is_empty"]:
            return "response body is empty", diagnostics
        if diagnostics["data_line_count"] == 0:
            return "response body does not contain any SSE data lines", diagnostics
        if (
            diagnostics["json_payload_count"] == 0
            and diagnostics["json_decode_error_count"] > 0
        ):
            return "all SSE data lines failed JSON decoding", diagnostics
        if (
            diagnostics["done_line_count"] > 0
            and diagnostics["choices_payload_count"] == 0
        ):
            return "stream ended with [DONE] before any assistant payload arrived", diagnostics
        if (
            diagnostics["error_payload_count"] > 0
            and diagnostics["choices_payload_count"] == 0
        ):
            return "SSE payload only contained upstream error objects", diagnostics
        if diagnostics["choices_payload_count"] == 0:
            return "no SSE payload contained a non-empty choices array", diagnostics
        if (
            diagnostics["delta_content_chunk_count"] == 0
            and diagnostics["message_content_chunk_count"] == 0
            and diagnostics["reasoning_chunk_count"] == 0
            and diagnostics["tool_call_chunk_count"] == 0
        ):
            return (
                "choices were present but no content, reasoning_content, or tool_calls could be reconstructed",
                diagnostics,
            )
        return "stream body format was not recognized by the parser", diagnostics

    @staticmethod
    def _build_moonshot_image_payload_from_pil(image: Image.Image) -> Dict[str, Any]:
        """将 PIL 图片转换为 Moonshot 识别所需 payload。"""
        mime_type = "image/webp"
        buffered = io.BytesIO()
        try:
            image.save(buffered, format="WEBP")
        except Exception:
            mime_type = "image/png"
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")

        image_bytes = buffered.getvalue()
        return {
            "type": "image",
            "mime_type": mime_type,
            "data_size": len(image_bytes),
            "data_preview": image_bytes.hex(),
        }

    async def build_turn_content(
        self, parts: List[Any], *, enable_vision: Optional[bool] = None
    ) -> str:
        """
        构建单条消息在 Custom 通道中的文本内容。

        当 CUSTOM_MODEL_ENABLE_VISION=true 时：
        - 对图片调用 Moonshot 进行识别
        - 将所有【图片识别结果】统一追加到该条消息文本末尾（仿 deepseek）
        """
        if enable_vision is None:
            runtime_config = self.refresh_from_env()
            enable_vision = bool(runtime_config["enable_vision"])

        content_chunks: List[str] = []
        vision_chunks: List[str] = []

        # 额外注入“当前用户输入”到识图请求中（放在图片前），帮助视觉模型理解提问意图：
        # - 只传入输入本身，不包含“上下文提示/引用回复”等前缀
        # - 若该轮只有图片（无文本输入），则不注入
        user_input_text_for_vision: Optional[str] = None
        try:
            text_fragments: List[str] = []
            for part in parts or []:
                if hasattr(part, "thought") and getattr(part, "thought", False):
                    continue

                if isinstance(part, Image.Image) or (
                    isinstance(part, dict) and part.get("type") == "image"
                ):
                    continue

                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    text_fragments.append(part["text"])
                else:
                    text_fragments.append(str(part))

            combined_text_for_vision = "".join(text_fragments).strip()
            if combined_text_for_vision:
                marker = "以下是当前输入内容:"
                candidate = (
                    combined_text_for_vision.split(marker, 1)[-1].strip()
                    if marker in combined_text_for_vision
                    else combined_text_for_vision
                )

                matches = list(re.finditer(r"\[[^\]]+\]:(?:\s*)", candidate))
                if matches:
                    candidate = candidate[matches[-1].end() :].strip()
                    if candidate and "用户消息:(图片消息)" not in candidate:
                        user_input_text_for_vision = candidate
        except Exception:
            user_input_text_for_vision = None

        first_effective_part_seen = False
        first_effective_part_is_image = False

        for part in parts or []:
            if hasattr(part, "thought") and getattr(part, "thought", False):
                continue

            if not first_effective_part_seen:
                first_effective_part_seen = True
                first_effective_part_is_image = isinstance(part, Image.Image) or (
                    isinstance(part, dict) and part.get("type") == "image"
                )

            if isinstance(part, dict) and isinstance(part.get("text"), str):
                content_chunks.append(part["text"])
                continue

            # 未开启识图：直接跳过图片
            if not enable_vision:
                if isinstance(part, Image.Image) or (
                    isinstance(part, dict) and part.get("type") == "image"
                ):
                    continue
                content_chunks.append(str(part))
                continue

            if isinstance(part, Image.Image):
                try:
                    image_payload = self._build_moonshot_image_payload_from_pil(part)
                    vision_text = await moonshot_vision_service.recognize_image(
                        image_payload,
                        user_input_text=user_input_text_for_vision,
                    )
                except Exception as e:
                    log.error("[Custom] Moonshot 图片识别流程异常: %s", e, exc_info=True)
                    vision_text = "（图片识别失败：处理流程异常）"

                vision_chunks.append(str(vision_text))
                continue

            if isinstance(part, dict) and part.get("type") == "image":
                try:
                    vision_text = await moonshot_vision_service.recognize_image(
                        part,
                        user_input_text=user_input_text_for_vision,
                    )
                except Exception as e:
                    log.error("[Custom] Moonshot 图片字典识别异常: %s", e, exc_info=True)
                    vision_text = "（图片识别失败：处理流程异常）"

                vision_chunks.append(str(vision_text))
                continue

            content_chunks.append(str(part))

        base_text = "".join(content_chunks).strip()

        vision_items: List[str] = []
        for vision_text in vision_chunks:
            cleaned = str(vision_text).strip()
            if not cleaned:
                continue
            vision_items.append(f"【图片识别结果】{cleaned}")

        if not vision_items:
            return base_text

        ocr_text = "   ".join(vision_items)

        if not base_text:
            return ocr_text

        combined = f"{base_text}   {ocr_text}"
        if first_effective_part_is_image:
            combined = "\n" + combined
        return combined

    async def post_process_tool_response(
        self, raw_response: Any, *, enable_vision: Optional[bool] = None
    ) -> Any:
        """
        Custom 专属工具结果后处理（仿 DeepSeek）：

        - 对 get_user_profile 返回的头像/横幅图片做识图摘要（仅 CUSTOM_MODEL_ENABLE_VISION=true 时）
        - 移除原始 base64，避免将大字段回传给模型
        """
        if enable_vision is None:
            runtime_config = self.refresh_from_env()
            enable_vision = bool(runtime_config["enable_vision"])

        clean_response = copy.deepcopy(raw_response)

        if not isinstance(clean_response, dict):
            return clean_response

        result_data = clean_response.get("result", {})
        if not isinstance(result_data, dict):
            return clean_response

        profile = result_data.get("profile", {})
        if not isinstance(profile, dict):
            return clean_response

        image_fields = [
            (
                "avatar_image_base64",
                "avatar_mime_type",
                "avatar",
                "请识别这张用户头像图片，简洁描述可见人物、风格、配色与关键元素。",
                "头像",
            ),
            (
                "banner_image_base64",
                "banner_mime_type",
                "banner",
                "请识别这张用户横幅图片，简洁描述画面内容、风格、配色与关键信息。",
                "横幅",
            ),
        ]

        for (
            image_key,
            mime_key,
            label,
            vision_prompt,
            label_cn,
        ) in image_fields:
            image_b64 = profile.get(image_key)
            if not (isinstance(image_b64, str) and image_b64.strip()):
                continue

            # 未开启识图：仅做裁剪，避免把 base64 回灌给模型
            if not enable_vision:
                profile.pop(image_key, None)
                profile.setdefault(f"{label}_note", f"（{label_cn}原始图片数据已省略）")
                continue

            try:
                image_bytes = base64.b64decode(image_b64)
                image_mime_type = profile.get(mime_key, "image/png")
                if not isinstance(image_mime_type, str) or not image_mime_type:
                    image_mime_type = "image/png"

                image_payload = {
                    "type": "image",
                    "mime_type": image_mime_type,
                    "data_size": len(image_bytes),
                    "data_preview": image_bytes.hex(),
                }
                image_vision_text = await moonshot_vision_service.recognize_image(
                    image_payload,
                    prompt=vision_prompt,
                )
                profile[f"{label}_image_vision"] = image_vision_text
            except Exception as e:
                log.error(
                    "[Custom] 处理 get_user_profile %s识图失败: %s",
                    label_cn,
                    e,
                    exc_info=True,
                )
                profile[f"{label}_image_vision"] = f"（{label_cn}识图失败：处理异常）"
            finally:
                profile.pop(image_key, None)
                profile.setdefault(
                    f"{label}_note",
                    f"（{label_cn}原始图片数据已省略，已提供识图摘要）",
                )

        return clean_response

    def get_validation_error(
        self, runtime_config: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        if runtime_config is None:
            runtime_config = self.refresh_from_env()

        api_key_error = str(runtime_config.get("api_key_error", "") or "").strip()
        if api_key_error:
            log.warning(
                "请求使用 custom 但 CUSTOM_MODEL_API_KEY 无法解析：%s",
                api_key_error,
            )
            return api_key_error

        if (
            runtime_config["base_url"]
            and runtime_config.get("api_keys")
            and runtime_config["model_name"]
        ):
            return None

        log.warning(
            "请求使用 custom 但未配置 CUSTOM_MODEL_URL / CUSTOM_MODEL_API_KEY / CUSTOM_MODEL_NAME。"
        )
        return "custom 配置缺失，请检查 CUSTOM_MODEL_URL、CUSTOM_MODEL_API_KEY、CUSTOM_MODEL_NAME。"

    def get_request_model_name(
        self,
        fallback_model_name: str,
        runtime_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        if runtime_config is None:
            runtime_config = self.refresh_from_env()
        return str(runtime_config["model_name"] or fallback_model_name)

    @staticmethod
    def is_sse_chat_completion_response(response: httpx.Response) -> bool:
        content_type = (response.headers.get("content-type") or "").lower()
        response_text = (response.text or "").lstrip()
        return "text/event-stream" in content_type or response_text.startswith("data:")

    @staticmethod
    def parse_streaming_chat_completion_body(body: str) -> Optional[Dict[str, Any]]:
        if not body or "data:" not in body:
            return None

        response_id: Optional[str] = None
        model_name: Optional[str] = None
        created_ts: Optional[int] = None
        usage_payload: Optional[Dict[str, Any]] = None
        latest_finish_reason: Optional[str] = None

        content_chunks: List[str] = []
        reasoning_chunks: List[str] = []
        tool_call_map: Dict[int, Dict[str, Any]] = {}

        def _upsert_tool_call(
            tool_call_payload: Dict[str, Any], fallback_index: int, from_delta: bool
        ) -> None:
            index = tool_call_payload.get("index", fallback_index)
            if not isinstance(index, int):
                index = fallback_index

            existing = tool_call_map.setdefault(
                index,
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )

            tool_call_id = tool_call_payload.get("id")
            if isinstance(tool_call_id, str) and tool_call_id:
                existing["id"] = tool_call_id

            tool_call_type = tool_call_payload.get("type")
            if isinstance(tool_call_type, str) and tool_call_type:
                existing["type"] = tool_call_type

            function_payload = tool_call_payload.get("function")
            if not isinstance(function_payload, dict):
                return

            name_value = function_payload.get("name")
            if isinstance(name_value, str) and name_value:
                if from_delta:
                    existing["function"]["name"] += name_value
                else:
                    existing["function"]["name"] = name_value

            arguments_value = function_payload.get("arguments")
            if isinstance(arguments_value, str):
                if from_delta:
                    existing["function"]["arguments"] += arguments_value
                else:
                    existing["function"]["arguments"] = arguments_value
            elif isinstance(arguments_value, (dict, list)):
                existing["function"]["arguments"] = json.dumps(
                    arguments_value, ensure_ascii=False
                )

        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue

            payload_str = line[5:].strip()
            if not payload_str or payload_str == "[DONE]":
                continue

            try:
                payload = json.loads(payload_str)
            except json.JSONDecodeError:
                continue

            if response_id is None and isinstance(payload.get("id"), str):
                response_id = payload["id"]
            if model_name is None and isinstance(payload.get("model"), str):
                model_name = payload["model"]
            if created_ts is None and isinstance(payload.get("created"), int):
                created_ts = payload["created"]

            if isinstance(payload.get("usage"), dict):
                usage_payload = payload["usage"]

            choices = payload.get("choices")
            if not isinstance(choices, list) or not choices:
                continue

            first_choice = choices[0]
            if not isinstance(first_choice, dict):
                continue

            finish_reason = first_choice.get("finish_reason")
            if isinstance(finish_reason, str) and finish_reason:
                latest_finish_reason = finish_reason

            delta = first_choice.get("delta")
            if isinstance(delta, dict):
                delta_content = delta.get("content")
                if isinstance(delta_content, str):
                    content_chunks.append(delta_content)
                elif isinstance(delta_content, list):
                    extracted_delta_content = (
                        CustomModelClient._extract_text_from_openai_content(
                            delta_content
                        )
                    )
                    if extracted_delta_content:
                        content_chunks.append(extracted_delta_content)

                delta_reasoning = delta.get("reasoning_content")
                if isinstance(delta_reasoning, str):
                    reasoning_chunks.append(delta_reasoning)
                elif isinstance(delta_reasoning, list):
                    extracted_delta_reasoning = (
                        CustomModelClient._extract_text_from_openai_content(
                            delta_reasoning
                        )
                    )
                    if extracted_delta_reasoning:
                        reasoning_chunks.append(extracted_delta_reasoning)

                delta_tool_calls = delta.get("tool_calls")
                if isinstance(delta_tool_calls, list):
                    for idx, tool_call_delta in enumerate(delta_tool_calls):
                        if isinstance(tool_call_delta, dict):
                            _upsert_tool_call(
                                tool_call_delta, fallback_index=idx, from_delta=True
                            )

            message_block = first_choice.get("message")
            if isinstance(message_block, dict):
                message_content = message_block.get("content")
                if isinstance(message_content, str) and message_content and not content_chunks:
                    content_chunks.append(message_content)
                elif isinstance(message_content, list) and not content_chunks:
                    extracted_message_content = (
                        CustomModelClient._extract_text_from_openai_content(
                            message_content
                        )
                    )
                    if extracted_message_content:
                        content_chunks.append(extracted_message_content)

                message_reasoning = message_block.get("reasoning_content")
                if (
                    isinstance(message_reasoning, str)
                    and message_reasoning
                    and not reasoning_chunks
                ):
                    reasoning_chunks.append(message_reasoning)
                elif isinstance(message_reasoning, list) and not reasoning_chunks:
                    extracted_message_reasoning = (
                        CustomModelClient._extract_text_from_openai_content(
                            message_reasoning
                        )
                    )
                    if extracted_message_reasoning:
                        reasoning_chunks.append(extracted_message_reasoning)

                message_tool_calls = message_block.get("tool_calls")
                if isinstance(message_tool_calls, list):
                    for idx, tool_call_message in enumerate(message_tool_calls):
                        if isinstance(tool_call_message, dict):
                            _upsert_tool_call(
                                tool_call_message, fallback_index=idx, from_delta=False
                            )

        final_content = "".join(content_chunks).strip()
        final_reasoning = "".join(reasoning_chunks).strip()
        final_tool_calls = [
            tool_call_map[idx]
            for idx in sorted(tool_call_map.keys())
            if tool_call_map[idx].get("function", {}).get("name")
        ]

        if not final_content and not final_reasoning and not final_tool_calls:
            return None

        message: Dict[str, Any] = {"role": "assistant", "content": final_content}
        if final_reasoning:
            message["reasoning_content"] = final_reasoning
        if final_tool_calls:
            message["tool_calls"] = final_tool_calls

        final_finish_reason = latest_finish_reason or (
            "tool_calls" if final_tool_calls else "stop"
        )

        result: Dict[str, Any] = {
            "id": response_id or "custom-sse-reconstructed",
            "object": "chat.completion",
            "created": created_ts or int(datetime.now().timestamp()),
            "model": model_name or "",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": final_finish_reason,
                }
            ],
        }
        if usage_payload:
            result["usage"] = usage_payload

        return result

    async def send(
        self,
        http_client: httpx.AsyncClient,
        payload: Dict[str, Any],
        override_base_url: Optional[str] = None,
        runtime_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if runtime_config is None:
            runtime_config = self.refresh_from_env()

        api_url = (override_base_url or runtime_config["base_url"] or "").rstrip("/")
        api_key, _, total_api_keys = self._get_active_api_key_snapshot(runtime_config)

        if not api_url:
            raise ValueError("OpenAI 兼容通道 URL 配置缺失，请检查配置。")
        if not api_key:
            raise ValueError("OpenAI 兼容通道 API Key 配置缺失，请检查配置。")

        api_url = self._build_chat_completions_url(api_url)
        request_payload = dict(payload)
        request_payload["stream"] = True
        network_timeout_seconds = 10.0
        first_token_timeout_seconds = 12.0
        idle_timeout_seconds = float(
            runtime_config.get("stream_idle_timeout_seconds", 5.0) or 5.0
        )
        request_timeout = httpx.Timeout(
            connect=network_timeout_seconds,
            read=network_timeout_seconds,
            write=max(network_timeout_seconds, 10.0),
            pool=network_timeout_seconds,
        )
        headers = {
            "Content-Type": "application/json",
        }
        gateway_provider_timeout_ms = int(
            runtime_config.get("gateway_provider_timeout_ms", 4000) or 4000
        )
        request_model_name = str(request_payload.get("model", ""))
        normalized_request_model = request_model_name.strip().lower()
        gateway_provider_name = self._resolve_gateway_provider_name(
            request_model_name,
            runtime_config.get("gateway_provider_name"),
        )
        is_vercel_gateway = "ai-gateway.vercel.sh" in api_url.lower()
        is_kimi_k2_5_model = normalized_request_model == "moonshotai/kimi-k2.5"
        kimi_k2_5_provider_order = [
            "moonshotai",
            "fireworks",
            "togetherai",
            "novita",
            "bedrock",

        ]
        active_kimi_provider_order = list(kimi_k2_5_provider_order)
        if is_kimi_k2_5_model:
            active_kimi_provider_order = await self._get_rotated_provider_order(
                model_key=normalized_request_model,
                base_order=kimi_k2_5_provider_order,
            )
        provider_timeout_injected = False
        provider_order_injected = False

        if is_vercel_gateway:
            provider_options = request_payload.get("providerOptions")
            if not isinstance(provider_options, dict):
                provider_options = {}

            gateway_options = provider_options.get("gateway")
            if not isinstance(gateway_options, dict):
                gateway_options = {}

            provider_timeouts = gateway_options.get("providerTimeouts")
            if not isinstance(provider_timeouts, dict):
                provider_timeouts = {}

            byok_timeouts = provider_timeouts.get("byok")
            if not isinstance(byok_timeouts, dict):
                byok_timeouts = {}

            if gateway_provider_timeout_ms > 0:
                if is_kimi_k2_5_model:
                    for provider_name in kimi_k2_5_provider_order:
                        if provider_name not in byok_timeouts:
                            byok_timeouts[provider_name] = gateway_provider_timeout_ms
                            provider_timeout_injected = True
                elif gateway_provider_name and gateway_provider_name not in byok_timeouts:
                    byok_timeouts[gateway_provider_name] = gateway_provider_timeout_ms
                    provider_timeout_injected = True

            if is_kimi_k2_5_model:
                existing_order = gateway_options.get("order")
                if existing_order != active_kimi_provider_order:
                    gateway_options["order"] = list(active_kimi_provider_order)
                    provider_order_injected = True

            provider_timeouts["byok"] = byok_timeouts
            gateway_options["providerTimeouts"] = provider_timeouts
            provider_options["gateway"] = gateway_options
            request_payload["providerOptions"] = provider_options

            if provider_timeout_injected or provider_order_injected:
                order_for_log = gateway_options.get("order")
                if isinstance(order_for_log, list):
                    order_key = ",".join(str(item) for item in order_for_log)
                else:
                    order_key = str(order_for_log)
                log_key = (
                    f"{api_url}|{request_model_name}|{gateway_provider_timeout_ms}|"
                    f"{order_key}"
                )
                if log_key not in self._gateway_options_log_once:
                    self._gateway_options_log_once.add(log_key)
                    log.info(
                        "[Custom] Injected AI Gateway provider options | model=%s | order=%s | timeout_ms=%s | url=%s",
                        request_model_name,
                        gateway_options.get("order"),
                        gateway_provider_timeout_ms,
                        api_url,
                    )

        used_streaming = False
        response: Optional[httpx.Response] = None
        max_key_attempts = max(total_api_keys, 1)
        loop = asyncio.get_running_loop()

        for attempt in range(max_key_attempts):
            api_key, api_key_index, total_api_keys = self._get_active_api_key_snapshot(
                runtime_config
            )
            headers["Authorization"] = f"Bearer {api_key}"
            request_started_at = loop.time()
            stream_phase = "awaiting_response_headers"
            body_line_count = 0
            meaningful_line_count = 0
            used_streaming = False
            received_first_meaningful_token = False
            saw_non_sse_payload = False

            try:
                async with http_client.stream(
                    "POST",
                    api_url,
                    headers=headers,
                    json=request_payload,
                    timeout=request_timeout,
                ) as current_response:
                    stream_phase = "response_headers_received"

                    if current_response.is_error:
                        stream_phase = "error_response_body"
                        await current_response.aread()
                        response_text = current_response.text or ""
                        combined_error_text = (
                            f"{current_response.status_code} {current_response.reason_phrase}"
                        )
                        if response_text:
                            combined_error_text = (
                                f"{combined_error_text} | {response_text}"
                            )

                        if self._is_payment_required_error(
                            current_response.status_code, combined_error_text
                        ):
                            rotated = await self._rotate_to_next_api_key(
                                failed_api_key=api_key,
                                failed_index=api_key_index,
                                total_keys=total_api_keys,
                                reason_preview=combined_error_text[:1000],
                                runtime_config=runtime_config,
                            )
                            if (
                                rotated
                                and total_api_keys > 1
                                and attempt < max_key_attempts - 1
                            ):
                                continue

                        current_response.raise_for_status()

                    body_lines: List[str] = []
                    line_iterator = current_response.aiter_lines()
                    used_streaming = False
                    stream_phase = "response_body"
                    stream_started_at = loop.time()
                    last_meaningful_output_at = stream_started_at
                    sse_line_samples: List[str] = []

                    while True:
                        now = loop.time()
                        if received_first_meaningful_token:
                            timeout_seconds = max(
                                idle_timeout_seconds
                                - (now - last_meaningful_output_at),
                                0.0,
                            )
                        else:
                            timeout_seconds = max(
                                first_token_timeout_seconds
                                - (now - stream_started_at),
                                0.0,
                            )

                        if timeout_seconds <= 0:
                            elapsed_total_seconds = now - request_started_at
                            if received_first_meaningful_token:
                                log.warning(
                                    "[Custom] Stream idle timeout | attempt=%s/%s | elapsed=%.2fs | "
                                    "idle_timeout=%.1fs | body_lines=%s | meaningful_lines=%s | "
                                    "samples=%s | model=%s | url=%s",
                                    attempt + 1,
                                    max_key_attempts,
                                    elapsed_total_seconds,
                                    idle_timeout_seconds,
                                    body_line_count,
                                    meaningful_line_count,
                                    sse_line_samples,
                                    str(request_payload.get("model", "")),
                                    api_url,
                                )
                                raise httpx.ReadTimeout(
                                    (
                                        "[Custom] Stream idle timeout: no meaningful output "
                                        f"received within {idle_timeout_seconds:.1f}s"
                                    ),
                                    request=current_response.request,
                                )

                            log.warning(
                                "[Custom] First token timeout | attempt=%s/%s | elapsed=%.2fs | "
                                "first_token_timeout=%.1fs | body_lines=%s | meaningful_lines=%s | "
                                "samples=%s | model=%s | url=%s",
                                attempt + 1,
                                max_key_attempts,
                                elapsed_total_seconds,
                                first_token_timeout_seconds,
                                body_line_count,
                                meaningful_line_count,
                                sse_line_samples,
                                str(request_payload.get("model", "")),
                                api_url,
                            )
                            raise httpx.ReadTimeout(
                                (
                                    "[Custom] First token timeout: no meaningful output "
                                    f"received within {first_token_timeout_seconds:.1f}s"
                                ),
                                request=current_response.request,
                            )

                        try:
                            line = await asyncio.wait_for(
                                line_iterator.__anext__(),
                                timeout=timeout_seconds,
                            )
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError as exc:
                            elapsed_total_seconds = loop.time() - request_started_at
                            if received_first_meaningful_token:
                                log.warning(
                                    "[Custom] Stream idle timeout | attempt=%s/%s | elapsed=%.2fs | "
                                    "idle_timeout=%.1fs | body_lines=%s | meaningful_lines=%s | "
                                    "samples=%s | model=%s | url=%s",
                                    attempt + 1,
                                    max_key_attempts,
                                    elapsed_total_seconds,
                                    idle_timeout_seconds,
                                    body_line_count,
                                    meaningful_line_count,
                                    sse_line_samples,
                                    str(request_payload.get("model", "")),
                                    api_url,
                                )
                                raise httpx.ReadTimeout(
                                    (
                                        "[Custom] Stream idle timeout: no meaningful output "
                                        f"received within {idle_timeout_seconds:.1f}s"
                                    ),
                                    request=current_response.request,
                                ) from exc

                            log.warning(
                                "[Custom] First token timeout | attempt=%s/%s | elapsed=%.2fs | "
                                "first_token_timeout=%.1fs | body_lines=%s | meaningful_lines=%s | "
                                "samples=%s | model=%s | url=%s",
                                attempt + 1,
                                max_key_attempts,
                                elapsed_total_seconds,
                                first_token_timeout_seconds,
                                body_line_count,
                                meaningful_line_count,
                                sse_line_samples,
                                str(request_payload.get("model", "")),
                                api_url,
                            )
                            raise httpx.ReadTimeout(
                                (
                                    "[Custom] First token timeout: no meaningful output "
                                    f"received within {first_token_timeout_seconds:.1f}s"
                                ),
                                request=current_response.request,
                            ) from exc

                        body_lines.append(line)
                        body_line_count += 1
                        stripped = line.strip()

                        if not stripped or stripped.startswith(":"):
                            continue

                        if not stripped.startswith("data:"):
                            saw_non_sse_payload = True
                            break

                        used_streaming = True

                        if len(sse_line_samples) < 5:
                            sample = self._build_sse_log_sample(stripped)
                            if sample:
                                sse_line_samples.append(sample)

                        if self._is_meaningful_sse_data_line(stripped):
                            received_first_meaningful_token = True
                            meaningful_line_count += 1
                            last_meaningful_output_at = loop.time()

                    if saw_non_sse_payload and not used_streaming:
                        log.info(
                            "[Custom] Upstream returned non-SSE payload after stream request | model=%s | url=%s",
                            str(request_payload.get("model", "")),
                            api_url,
                        )

                    response = self._build_buffered_response(
                        current_response,
                        "\n".join(body_lines),
                    )
                    break
            except httpx.RequestError as exc:
                elapsed_total_seconds = loop.time() - request_started_at
                next_kimi_provider_order: Optional[List[str]] = None
                if is_vercel_gateway and is_kimi_k2_5_model:
                    next_kimi_provider_order = await self._advance_rotated_provider_order(
                        model_key=normalized_request_model,
                        base_order=kimi_k2_5_provider_order,
                    )
                    log.warning(
                        "[Custom] Rotated provider order for retry after network error | model=%s | from=%s | to=%s",
                        request_model_name,
                        active_kimi_provider_order,
                        next_kimi_provider_order,
                    )

                log.warning(
                    "[Custom] Stream request error | attempt=%s/%s | phase=%s | elapsed=%.2fs | "
                    "net_timeout(connect/read/write/pool)=%.1f/%.1f/%.1f/%.1f | "
                    "first_token_timeout=%.1fs | idle_timeout=%.1fs | "
                    "received_first_meaningful_token=%s | body_lines=%s | meaningful_lines=%s | "
                    "used_streaming=%s | saw_non_sse_payload=%s | samples=%s | model=%s | url=%s | "
                    "order=%s | next_order=%s | "
                    "exc_type=%s | exc=%s",
                    attempt + 1,
                    max_key_attempts,
                    stream_phase,
                    elapsed_total_seconds,
                    request_timeout.connect,
                    request_timeout.read,
                    request_timeout.write,
                    request_timeout.pool,
                    first_token_timeout_seconds,
                    idle_timeout_seconds,
                    received_first_meaningful_token,
                    body_line_count,
                    meaningful_line_count,
                    used_streaming,
                    saw_non_sse_payload,
                    sse_line_samples if "sse_line_samples" in locals() else [],
                    str(request_payload.get("model", "")),
                    api_url,
                    active_kimi_provider_order if is_kimi_k2_5_model else None,
                    next_kimi_provider_order,
                    type(exc).__name__,
                    exc,
                )
                raise

        if response is None:
            raise RuntimeError("[Custom] Request completed without a valid response.")

        return {
            "response": response,
            "used_api_url": api_url,
            "used_slot_label": "custom",
            "used_slot_id": None,
            "used_key_tail": self._mask_key_tail(api_key),
            "used_model_name": str(request_payload.get("model", "")),
            "stream_enabled": used_streaming,
            "stream_idle_timeout_seconds": idle_timeout_seconds,
            "skip_custom_site_for_this_turn": False,
        }
