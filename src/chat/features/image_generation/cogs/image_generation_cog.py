# -*- coding: utf-8 -*-

import asyncio
import base64
import io
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands

from src import config

log = logging.getLogger(__name__)


def _truncate_text(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def _escape_code_block(text: str) -> str:
    return (text or "").replace("```", "`\u200b``")


def _format_prompt_block(prompt: str, limit: int = 3200) -> str:
    prompt_preview = _truncate_text(prompt, limit)
    return f"```text\n{_escape_code_block(prompt_preview)}\n```"


def _strip_wrapping_quotes(value: str) -> str:
    value = (value or "").strip()
    quote_pairs = [
        ('"', '"'),
        ("'", "'"),
        ("“", "”"),
        ("‘", "’"),
    ]

    while value:
        changed = False
        for left, right in quote_pairs:
            if value.startswith(left) and value.endswith(right):
                value = value[len(left) : len(value) - len(right)].strip()
                changed = True
                break
        if not changed:
            break

    return value


def _strip_inline_comment(value: str) -> str:
    if not value:
        return ""

    result: list[str] = []
    open_quote_to_close = {
        '"': '"',
        "'": "'",
        "“": "”",
        "‘": "’",
    }
    expected_closers: list[str] = []

    for char in value:
        if expected_closers and char == expected_closers[-1]:
            result.append(char)
            expected_closers.pop()
            continue

        if not expected_closers and char in open_quote_to_close:
            result.append(char)
            expected_closers.append(open_quote_to_close[char])
            continue

        if not expected_closers and char == "#":
            break

        result.append(char)

    return "".join(result).strip()


def _extract_error_message(payload: Any) -> str:
    if payload is None:
        return ""

    if isinstance(payload, str):
        return payload.strip()

    if isinstance(payload, dict):
        for key in ("error", "message", "detail", "msg"):
            if key not in payload:
                continue
            nested = _extract_error_message(payload.get(key))
            if nested:
                return nested

        for value in payload.values():
            nested = _extract_error_message(value)
            if nested:
                return nested
        return ""

    if isinstance(payload, list):
        for item in payload:
            nested = _extract_error_message(item)
            if nested:
                return nested
        return ""

    return str(payload).strip()


def _get_developer_user_ids() -> set[int]:
    developer_ids = set(config.DEVELOPER_USER_IDS)
    raw_value = _strip_wrapping_quotes(str(os.getenv("DEVELOPER_USER_IDS", "") or ""))

    if not raw_value:
        env_path = os.path.join(config.BASE_DIR, ".env")
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as env_file:
                    for raw_line in env_file:
                        line = raw_line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue

                        key, value = line.split("=", 1)
                        if key.strip() != "DEVELOPER_USER_IDS":
                            continue

                        raw_value = _strip_wrapping_quotes(_strip_inline_comment(value))
                        break
            except Exception as exc:
                log.warning("读取 .env 中的 DEVELOPER_USER_IDS 失败: %s", exc)

    if not raw_value:
        return developer_ids

    for part in raw_value.split(","):
        cleaned = _strip_wrapping_quotes(part).strip()
        if cleaned.isdigit():
            developer_ids.add(int(cleaned))

    return developer_ids


@dataclass(slots=True)
class GeneratedImage:
    data: bytes
    filename: str


@dataclass(slots=True)
class ImageGenerationError:
    message: str
    status_code: int | None = None
    payload: Any = None

    def should_rotate_key(self) -> bool:
        return self.status_code == 402


@dataclass(slots=True)
class GlobalQuotaStatus:
    date_key: str
    daily_limit: int
    used_count: int

    @property
    def remaining(self) -> int:
        return max(self.daily_limit - self.used_count, 0)


@dataclass(slots=True)
class QuotaReservation:
    date_key: str


class GrokImagineImageClient:
    API_URL = "https://ai-gateway.vercel.sh/v1/images/generations"
    MODEL = "xai/grok-imagine-image"
    ENV_KEY = "GROK_IMAGINE_API_KEY"

    def __init__(self):
        self._lock = asyncio.Lock()
        self._active_key_index = 0
        self._active_key_value: str | None = None

    @staticmethod
    def _get_env_path() -> str:
        return os.path.join(config.BASE_DIR, ".env")

    def _read_raw_api_key_value(self) -> str:
        env_path = self._get_env_path()
        if not os.path.exists(env_path):
            return str(os.getenv(self.ENV_KEY, "") or "").strip()

        try:
            with open(env_path, "r", encoding="utf-8") as env_file:
                for raw_line in env_file:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue

                    key, value = line.split("=", 1)
                    if key.strip() != self.ENV_KEY:
                        continue

                    return _strip_inline_comment(value)
        except Exception as exc:
            log.warning("读取 %s 失败，将回退到进程环境变量: %s", env_path, exc)

        return str(os.getenv(self.ENV_KEY, "") or "").strip()

    def _load_api_keys(self) -> list[str]:
        raw_value = self._read_raw_api_key_value()
        cleaned_value = _strip_wrapping_quotes(raw_value)
        return [
            _strip_wrapping_quotes(part)
            for part in cleaned_value.split(",")
            if _strip_wrapping_quotes(part)
        ]

    @staticmethod
    async def _read_response_payload(response: aiohttp.ClientResponse) -> Any:
        text = await response.text()
        if not text:
            return None

        try:
            return await response.json(content_type=None)
        except Exception:
            return text

    @staticmethod
    def _build_api_error(status_code: int, payload: Any) -> ImageGenerationError:
        detail = _truncate_text(_extract_error_message(payload), 300)
        if detail:
            message = f"生图接口返回错误（HTTP {status_code}）：{detail}"
        else:
            message = f"生图接口返回错误（HTTP {status_code}）。"
        return ImageGenerationError(
            message=message,
            status_code=status_code,
            payload=payload,
        )

    @staticmethod
    async def _download_image(
        session: aiohttp.ClientSession,
        image_url: str,
        filename: str,
    ) -> tuple[GeneratedImage | None, ImageGenerationError | None]:
        try:
            async with session.get(image_url) as response:
                if response.status != 200:
                    payload = await GrokImagineImageClient._read_response_payload(response)
                    return None, GrokImagineImageClient._build_api_error(
                        response.status,
                        payload,
                    )

                return GeneratedImage(data=await response.read(), filename=filename), None
        except asyncio.TimeoutError:
            return None, ImageGenerationError("下载生成图片超时，请稍后再试。")
        except aiohttp.ClientError as exc:
            return None, ImageGenerationError(f"下载生成图片失败：{exc}")

    async def _generate_with_key(
        self,
        api_key: str,
        prompt: str,
    ) -> tuple[GeneratedImage | None, ImageGenerationError | None]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.MODEL,
            "prompt": prompt,
        }
        timeout = aiohttp.ClientTimeout(total=180)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.API_URL,
                    headers=headers,
                    json=payload,
                ) as response:
                    response_payload = await self._read_response_payload(response)
                    if response.status != 200:
                        return None, self._build_api_error(
                            response.status,
                            response_payload,
                        )

                    items = (
                        response_payload.get("data", [])
                        if isinstance(response_payload, dict)
                        else []
                    )
                    for item in items:
                        if not isinstance(item, dict):
                            continue

                        filename = f"grok_imagine_{int(time.time())}.png"
                        image_url = str(item.get("url") or "").strip()
                        image_b64 = str(item.get("b64_json") or "").strip()

                        if image_url:
                            return await self._download_image(session, image_url, filename)

                        if image_b64:
                            try:
                                image_bytes = base64.b64decode(image_b64)
                                return GeneratedImage(data=image_bytes, filename=filename), None
                            except Exception as exc:
                                return None, ImageGenerationError(
                                    f"解析图片数据失败：{exc}"
                                )

                    return None, ImageGenerationError(
                        "生图接口返回成功，但没有找到可用的图片数据。",
                        status_code=response.status,
                        payload=response_payload,
                    )
        except asyncio.TimeoutError:
            return None, ImageGenerationError("请求生图接口超时，请稍后再试。")
        except aiohttp.ClientError as exc:
            return None, ImageGenerationError(f"请求生图接口失败：{exc}")
        except Exception as exc:
            log.error("调用 Grok Imagine 生图接口时发生未预期错误: %s", exc, exc_info=True)
            return None, ImageGenerationError(f"生成图片时发生错误：{exc}")

    async def generate_image(
        self,
        prompt: str,
    ) -> tuple[GeneratedImage | None, ImageGenerationError | None]:
        prompt = (prompt or "").strip()
        if not prompt:
            return None, ImageGenerationError("提示词不能为空。")

        async with self._lock:
            api_keys = self._load_api_keys()
            if not api_keys:
                return None, ImageGenerationError(
                    f"未在 .env 中找到可用的 {self.ENV_KEY} 配置。"
                )

            if self._active_key_value and self._active_key_value in api_keys:
                self._active_key_index = api_keys.index(self._active_key_value)
            elif self._active_key_index >= len(api_keys):
                self._active_key_index = 0

            for key_index in range(self._active_key_index, len(api_keys)):
                api_key = api_keys[key_index]
                self._active_key_index = key_index
                self._active_key_value = api_key

                image, error = await self._generate_with_key(api_key, prompt)
                if image is not None:
                    return image, None

                if error is None:
                    return None, ImageGenerationError("生成图片失败，但没有拿到具体错误信息。")

                if error.should_rotate_key():
                    next_index = key_index + 1
                    if next_index < len(api_keys):
                        self._active_key_index = next_index
                        self._active_key_value = api_keys[next_index]
                        log.warning(
                            "Grok Imagine 当前 key 触发 402 payment required，切换到下一个 key。"
                        )
                        continue

                    return None, ImageGenerationError(
                        "所有 GROK_IMAGINE_API_KEY 都已触发 402 payment required，请检查余额。",
                        status_code=402,
                        payload=error.payload,
                    )

                return None, error

            return None, ImageGenerationError("没有可用的 Grok Imagine API key。")


class GlobalDailyQuotaService:
    DEFAULT_DAILY_LIMIT = 10

    def __init__(self):
        self._lock = asyncio.Lock()
        try:
            self._tz = ZoneInfo("Asia/Shanghai")
        except ZoneInfoNotFoundError:
            self._tz = timezone(timedelta(hours=8))
        self._state_path = os.path.join(config.DATA_DIR, "image_generation_quota.json")

    def _today_key(self) -> str:
        return datetime.now(self._tz).date().isoformat()

    def _default_state(self) -> dict[str, Any]:
        return {
            "daily_limit": self.DEFAULT_DAILY_LIMIT,
            "days": {},
        }

    def _load_state_unlocked(self) -> dict[str, Any]:
        if not os.path.exists(self._state_path):
            return self._default_state()

        try:
            with open(self._state_path, "r", encoding="utf-8") as state_file:
                data = json.load(state_file)
                if not isinstance(data, dict):
                    return self._default_state()
                return data
        except Exception as exc:
            log.warning("读取生图配额文件失败，将使用默认值: %s", exc)
            return self._default_state()

    def _save_state_unlocked(self, state: dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
        with open(self._state_path, "w", encoding="utf-8") as state_file:
            json.dump(state, state_file, ensure_ascii=False, indent=2)

    def _normalize_state(self, state: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(state, dict):
            state = self._default_state()

        daily_limit = state.get("daily_limit", self.DEFAULT_DAILY_LIMIT)
        try:
            daily_limit = int(daily_limit)
        except Exception:
            daily_limit = self.DEFAULT_DAILY_LIMIT
        daily_limit = max(daily_limit, 0)

        days = state.get("days", {})
        if not isinstance(days, dict):
            days = {}

        normalized_days: dict[str, dict[str, int]] = {}
        for date_key, raw_day in days.items():
            if not isinstance(date_key, str) or not isinstance(raw_day, dict):
                continue
            used_count = raw_day.get("used_count", 0)
            try:
                used_count = int(used_count)
            except Exception:
                used_count = 0
            normalized_days[date_key] = {"used_count": max(used_count, 0)}

        return {
            "daily_limit": daily_limit,
            "days": normalized_days,
        }

    def _ensure_day_record(
        self,
        state: dict[str, Any],
        date_key: str,
    ) -> dict[str, int]:
        days = state.setdefault("days", {})
        day_record = days.get(date_key)
        if not isinstance(day_record, dict):
            day_record = {"used_count": 0}
            days[date_key] = day_record

        used_count = day_record.get("used_count", 0)
        try:
            used_count = int(used_count)
        except Exception:
            used_count = 0
        day_record["used_count"] = max(used_count, 0)
        return day_record

    def _build_status_from_state(
        self,
        state: dict[str, Any],
        date_key: str,
    ) -> GlobalQuotaStatus:
        day_record = self._ensure_day_record(state, date_key)
        return GlobalQuotaStatus(
            date_key=date_key,
            daily_limit=max(int(state.get("daily_limit", self.DEFAULT_DAILY_LIMIT)), 0),
            used_count=max(int(day_record.get("used_count", 0)), 0),
        )

    async def get_status(self) -> GlobalQuotaStatus:
        async with self._lock:
            state = self._normalize_state(self._load_state_unlocked())
            date_key = self._today_key()
            status = self._build_status_from_state(state, date_key)
            self._save_state_unlocked(state)
            return status

    async def set_daily_limit(self, new_limit: int) -> GlobalQuotaStatus:
        async with self._lock:
            state = self._normalize_state(self._load_state_unlocked())
            state["daily_limit"] = max(int(new_limit), 0)
            date_key = self._today_key()
            status = self._build_status_from_state(state, date_key)
            self._save_state_unlocked(state)
            return status

    async def reserve_generation(
        self,
        *,
        exempt: bool,
    ) -> tuple[bool, GlobalQuotaStatus, QuotaReservation | None]:
        async with self._lock:
            state = self._normalize_state(self._load_state_unlocked())
            date_key = self._today_key()
            status = self._build_status_from_state(state, date_key)

            if exempt:
                self._save_state_unlocked(state)
                return True, status, None

            if status.remaining <= 0:
                self._save_state_unlocked(state)
                return False, status, None

            day_record = self._ensure_day_record(state, date_key)
            day_record["used_count"] += 1
            status = self._build_status_from_state(state, date_key)
            self._save_state_unlocked(state)
            return True, status, QuotaReservation(date_key=date_key)

    async def refund_generation(
        self,
        reservation: QuotaReservation | None,
    ) -> GlobalQuotaStatus:
        async with self._lock:
            state = self._normalize_state(self._load_state_unlocked())
            if reservation is not None:
                day_record = self._ensure_day_record(state, reservation.date_key)
                if day_record["used_count"] > 0:
                    day_record["used_count"] -= 1

            today_key = self._today_key()
            status = self._build_status_from_state(state, today_key)
            self._save_state_unlocked(state)
            return status


class PromptInputModal(discord.ui.Modal, title="输入提示词"):
    def __init__(self, parent_view: "ImageGenerationPanelView", default_prompt: str = ""):
        super().__init__()
        self.parent_view = parent_view

        self.prompt_input = discord.ui.TextInput(
            label="提示词",
            placeholder="例如：一只站在雨夜霓虹街头的白狐，电影感构图，超高细节",
            style=discord.TextStyle.paragraph,
            required=True,
            default=default_prompt[:2000],
            max_length=2000,
        )
        self.add_item(self.prompt_input)

    async def on_submit(self, interaction: discord.Interaction):
        if not interaction.user or interaction.user.id != self.parent_view.opener_user_id:
            await interaction.response.send_message("这不是你的绘图面板。", ephemeral=True)
            return

        prompt = (self.prompt_input.value or "").strip()
        if not prompt:
            await interaction.response.send_message("提示词不能为空。", ephemeral=True)
            return

        self.parent_view.prompt = prompt
        self.parent_view.last_status = "提示词已更新，可以开始生成。"
        self.parent_view.last_error = None
        self.parent_view.last_public_message_url = None

        await interaction.response.defer(ephemeral=True)
        await self.parent_view.refresh_panel()


class ImageGenerationPanelView(discord.ui.View):
    def __init__(
        self,
        *,
        origin_interaction: discord.Interaction,
        image_client: GrokImagineImageClient,
        quota_service: GlobalDailyQuotaService,
        is_developer: bool,
    ):
        super().__init__(timeout=config.VIEW_TIMEOUT)
        self.origin_interaction = origin_interaction
        self.opener_user_id = origin_interaction.user.id
        self.image_client = image_client
        self.quota_service = quota_service
        self.is_developer = is_developer

        self.prompt = ""
        self.use_spoiler = False
        self.is_generating = False
        self.last_status = "点击按钮输入提示词。"
        self.last_error: str | None = None
        self.last_public_message_url: str | None = None
        self.quota_status = GlobalQuotaStatus(
            date_key="",
            daily_limit=GlobalDailyQuotaService.DEFAULT_DAILY_LIMIT,
            used_count=0,
        )

        self._rebuild_items()

    async def initialize(self, initial_status_message: str | None = None) -> None:
        await self._sync_quota_status()
        if initial_status_message:
            self.last_status = initial_status_message
        self._rebuild_items()

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user and interaction.user.id == self.opener_user_id:
            return True

        await interaction.response.send_message("这不是你的绘图面板。", ephemeral=True)
        return False

    async def _sync_quota_status(self) -> GlobalQuotaStatus:
        self.quota_status = await self.quota_service.get_status()
        return self.quota_status

    def _author_name(self) -> str:
        return self.origin_interaction.user.display_name

    def _author_icon(self) -> str:
        return self.origin_interaction.user.display_avatar.url

    def _quota_footer_text(self) -> str:
        remaining_text = (
            f"今日全局剩余次数：{self.quota_status.remaining}/{self.quota_status.daily_limit}"
        )
        if self.is_developer:
            return f"模型：{self.image_client.MODEL} | {remaining_text} | 开发者不限次"
        return f"模型：{self.image_client.MODEL} | {remaining_text}"

    def _can_generate_now(self) -> bool:
        if self.is_generating or not self.prompt:
            return False
        if self.is_developer:
            return True
        return self.quota_status.remaining > 0

    def _build_panel_embed(self) -> discord.Embed:
        color = config.EMBED_COLOR_PRIMARY
        if self.last_error:
            color = config.EMBED_COLOR_ERROR
        elif self.is_generating:
            color = config.EMBED_COLOR_WARNING

        embed = discord.Embed(title="AI 图片生成", color=color)
        embed.set_author(name=self._author_name(), icon_url=self._author_icon())

        if self.prompt:
            description_lines = [
                "**提示词：**",
                _format_prompt_block(self.prompt, limit=3000),
                f"**遮罩：** {'开启' if self.use_spoiler else '关闭'}",
            ]
        else:
            description_lines = [
                "还没有提示词。",
                "",
                "点击下方按钮输入提示词。",
            ]

        if self.last_error:
            description_lines.extend(["", f"**错误：** {self.last_error}"])
        elif self.last_status:
            description_lines.extend(["", f"**状态：** {self.last_status}"])

        if self.last_public_message_url:
            description_lines.extend(
                ["", f"**跳转：** [查看刚刚发送的图片]({self.last_public_message_url})"]
            )

        embed.description = "\n".join(description_lines)
        embed.set_footer(text=self._quota_footer_text())
        return embed

    def _build_public_embed(self, requester: discord.abc.User) -> discord.Embed:
        embed = discord.Embed(
            title="AI 图片生成",
            description=f"**提示词：**\n{_format_prompt_block(self.prompt, limit=3200)}",
            color=config.EMBED_COLOR_PRIMARY,
        )
        embed.set_author(
            name=requester.display_name,
            icon_url=requester.display_avatar.url,
        )
        embed.set_footer(text=f"模型：{self.image_client.MODEL}")
        return embed

    def _rebuild_items(self) -> None:
        self.clear_items()

        if not self.prompt:
            input_button = discord.ui.Button(
                label="点击输入提示词",
                style=discord.ButtonStyle.primary,
                disabled=self.is_generating,
                row=0,
            )
            input_button.callback = self._open_prompt_modal
            self.add_item(input_button)
            return

        generate_label = "正在生成..." if self.is_generating else "生成图片"
        if not self.is_generating and not self.is_developer and self.quota_status.remaining <= 0:
            generate_label = "今日次数已用尽"

        generate_button = discord.ui.Button(
            label=generate_label,
            style=discord.ButtonStyle.success,
            disabled=not self._can_generate_now(),
            row=0,
        )
        generate_button.callback = self._generate_image
        self.add_item(generate_button)

        modify_button = discord.ui.Button(
            label="修改提示词",
            style=discord.ButtonStyle.secondary,
            disabled=self.is_generating,
            row=0,
        )
        modify_button.callback = self._open_prompt_modal
        self.add_item(modify_button)

        spoiler_button = discord.ui.Button(
            label=f"遮罩：{'开' if self.use_spoiler else '关'}",
            style=(
                discord.ButtonStyle.primary
                if self.use_spoiler
                else discord.ButtonStyle.secondary
            ),
            disabled=self.is_generating,
            row=0,
        )
        spoiler_button.callback = self._toggle_spoiler
        self.add_item(spoiler_button)

    async def refresh_panel(self) -> None:
        await self._sync_quota_status()
        self._rebuild_items()
        await self.origin_interaction.edit_original_response(
            embed=self._build_panel_embed(),
            view=self,
        )

    async def _open_prompt_modal(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_modal(
            PromptInputModal(self, default_prompt=self.prompt)
        )

    async def _toggle_spoiler(self, interaction: discord.Interaction) -> None:
        await self._sync_quota_status()
        self.use_spoiler = not self.use_spoiler
        self.last_status = f"遮罩已{'开启' if self.use_spoiler else '关闭'}。"
        self.last_error = None
        self._rebuild_items()
        await interaction.response.edit_message(
            embed=self._build_panel_embed(),
            view=self,
        )

    async def _send_public_result(
        self,
        channel: discord.abc.Messageable,
        requester: discord.abc.User,
        image: GeneratedImage,
    ) -> discord.Message:
        public_filename = image.filename
        if self.use_spoiler and not public_filename.startswith("SPOILER_"):
            public_filename = f"SPOILER_{public_filename}"

        image_file = discord.File(io.BytesIO(image.data), filename=public_filename)
        return await channel.send(
            file=image_file,
            embed=self._build_public_embed(requester),
        )

    async def _generate_image(self, interaction: discord.Interaction) -> None:
        if not self.prompt:
            await interaction.response.send_message("请先输入提示词。", ephemeral=True)
            return

        channel = interaction.channel
        if channel is None or not hasattr(channel, "send"):
            await interaction.response.send_message("当前频道无法公开发送图片。", ephemeral=True)
            return

        allowed, reserved_status, reservation = await self.quota_service.reserve_generation(
            exempt=self.is_developer
        )
        self.quota_status = reserved_status

        if not allowed:
            self.last_error = "今日全局生成次数已经用完了。"
            self.last_status = "请明天再来，或者联系开发者调整每日次数。"
            self._rebuild_items()
            await interaction.response.edit_message(
                embed=self._build_panel_embed(),
                view=self,
            )
            return

        self.is_generating = True
        self.last_error = None
        self.last_public_message_url = None
        self.last_status = "正在生成图片，请稍候..."
        self._rebuild_items()
        await interaction.response.edit_message(
            embed=self._build_panel_embed(),
            view=self,
        )

        try:
            image, error = await self.image_client.generate_image(self.prompt)
            if error is not None:
                self.last_error = error.message
                self.last_status = "生成失败。"
                if reservation is not None:
                    self.quota_status = await self.quota_service.refund_generation(reservation)
                return

            if image is None:
                self.last_error = "没有返回图片结果。"
                self.last_status = "生成失败。"
                if reservation is not None:
                    self.quota_status = await self.quota_service.refund_generation(reservation)
                return

            public_message = await self._send_public_result(
                channel=channel,
                requester=interaction.user,
                image=image,
            )
            self.last_public_message_url = public_message.jump_url
            self.last_status = "图片已公开发送到当前频道。"
            self.last_error = None
        except Exception as exc:
            log.error("公开发送图片时发生错误: %s", exc, exc_info=True)
            self.last_error = f"公开发送失败：{exc}"
            self.last_status = "发送失败。"
            if reservation is not None:
                self.quota_status = await self.quota_service.refund_generation(reservation)
        finally:
            self.is_generating = False
            await self.refresh_panel()


class ImageGenerationCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.image_client = GrokImagineImageClient()
        self.quota_service = GlobalDailyQuotaService()
        self.developer_user_ids = _get_developer_user_ids()

    def _is_developer(self, user_id: int) -> bool:
        return user_id in self.developer_user_ids

    @app_commands.command(name="绘制图片", description="打开图片生成面板")
    @app_commands.describe(daily_limit="开发者可用：设置全局每日次数")
    async def draw_image(
        self,
        interaction: discord.Interaction,
        daily_limit: app_commands.Range[int, 0, 9999] | None = None,
    ):
        is_developer = self._is_developer(interaction.user.id)
        initial_status_message: str | None = None

        if daily_limit is not None:
            if not is_developer:
                await interaction.response.send_message(
                    "只有 DEVELOPER_USER_IDS 中的开发者可以修改每日次数。",
                    ephemeral=True,
                )
                return

            updated_status = await self.quota_service.set_daily_limit(int(daily_limit))
            initial_status_message = (
                f"已将全局每日次数设置为 {updated_status.daily_limit} 次。"
                f" 当前剩余 {updated_status.remaining} 次。"
            )

        view = ImageGenerationPanelView(
            origin_interaction=interaction,
            image_client=self.image_client,
            quota_service=self.quota_service,
            is_developer=is_developer,
        )
        await view.initialize(initial_status_message=initial_status_message)

        await interaction.response.send_message(
            embed=view._build_panel_embed(),
            view=view,
            ephemeral=True,
        )


async def setup(bot: commands.Bot):
    await bot.add_cog(ImageGenerationCog(bot))
