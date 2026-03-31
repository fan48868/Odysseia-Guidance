# -*- coding: utf-8 -*-
import discord
from discord import app_commands
from discord.ext import commands
import logging
import re

from sqlalchemy import select, update

from src.chat.features.personal_memory.services.personal_memory_vector_service import (
    personal_memory_vector_service,
)
from src.config import DEVELOPER_USER_IDS
from src.database.database import AsyncSessionLocal
from src.database.models import CommunityMemberProfile

log = logging.getLogger(__name__)

_LONG_TERM_HEADER = "### 长期记忆"
_RECENT_HEADER = "### 近期动态"
_MEMORY_PAGE_CHAR_LIMIT = 4000
_MEMORY_PREVIEW_TIMEOUT = 300

_BULLET_ITEM_RE = re.compile(r"^[-*•]\s*(.+)$")
_NUMBERED_ITEM_RE = re.compile(r"^\d+[\.\、]\s*(.+)$")


def _can_manage_memory(actor_user_id: int, target_user_id: int) -> bool:
    return actor_user_id == target_user_id or actor_user_id in DEVELOPER_USER_IDS


async def _personal_memory_profile_exists(user_id: int) -> bool:
    async with AsyncSessionLocal() as session:
        stmt = select(CommunityMemberProfile.discord_id).where(
            CommunityMemberProfile.discord_id == str(user_id)
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none() is not None


def _build_memory_view_content(user_id: int, current_summary_raw: str | None) -> str:
    content = f"当前操作对象：`{user_id}`\n点击下方绿色按钮即可查看或修改个人记忆摘要。"
    if not (current_summary_raw or "").strip():
        content += (
            "\n\n格式示例：\n"
            "```text\n"
            f"{_build_personal_summary_format_example()}\n"
            "```"
        )
    return content


def _build_empty_personal_summary() -> str:
    return f"{_LONG_TERM_HEADER}\n\n{_RECENT_HEADER}\n"


def _build_personal_summary_format_example() -> str:
    return f"{_LONG_TERM_HEADER}\n- 条目1\n{_RECENT_HEADER}\n- 条目1"


def _prepare_memory_summary_for_view(raw_summary: str | None) -> str:
    summary = raw_summary or ""
    if summary.strip():
        return summary
    return _build_empty_personal_summary()


def _split_memory_summary_into_pages(
    summary: str,
    max_chars: int = _MEMORY_PAGE_CHAR_LIMIT,
) -> list[str]:
    if max_chars <= 0:
        raise ValueError("分页长度必须大于 0。")

    if summary == "":
        return [""]

    pages: list[str] = []
    current_chunks: list[str] = []
    current_len = 0

    for segment in summary.splitlines(keepends=True) or [summary]:
        if len(segment) <= max_chars:
            if current_len + len(segment) <= max_chars:
                current_chunks.append(segment)
                current_len += len(segment)
            else:
                if current_chunks:
                    pages.append("".join(current_chunks))
                current_chunks = [segment]
                current_len = len(segment)
            continue

        if current_chunks:
            pages.append("".join(current_chunks))
            current_chunks = []
            current_len = 0

        start = 0
        while start < len(segment):
            chunk = segment[start : start + max_chars]
            start += max_chars

            if len(chunk) == max_chars:
                pages.append(chunk)
            else:
                current_chunks = [chunk]
                current_len = len(chunk)

    if current_chunks or not pages:
        pages.append("".join(current_chunks))

    return pages


def _replace_memory_page(
    summary: str,
    page_index: int,
    new_page_text: str,
    max_chars: int = _MEMORY_PAGE_CHAR_LIMIT,
) -> str:
    pages = _split_memory_summary_into_pages(summary, max_chars=max_chars)
    if page_index < 0 or page_index >= len(pages):
        raise IndexError("页码超出范围。")

    updated_pages = list(pages)
    updated_pages[page_index] = new_page_text
    return "".join(updated_pages)


def _validate_personal_summary_format(summary: str) -> None:
    lines = (summary or "").splitlines()

    long_indices = [i for i, l in enumerate(lines) if (l or "").strip() == _LONG_TERM_HEADER]
    if not long_indices:
        raise ValueError(f"记忆格式不正确：缺少“{_LONG_TERM_HEADER}”标题。")
    if len(long_indices) > 1:
        raise ValueError(f"记忆格式不正确：检测到多个“{_LONG_TERM_HEADER}”标题。")

    recent_indices = [i for i, l in enumerate(lines) if (l or "").strip() == _RECENT_HEADER]
    if not recent_indices:
        raise ValueError(f"记忆格式不正确：缺少“{_RECENT_HEADER}”标题。")
    if len(recent_indices) > 1:
        raise ValueError(f"记忆格式不正确：检测到多个“{_RECENT_HEADER}”标题。")

    if long_indices[0] > recent_indices[0]:
        raise ValueError(
            f"记忆格式不正确：“{_LONG_TERM_HEADER}”必须出现在“{_RECENT_HEADER}”之前。"
        )


def _validate_personal_summary_items_bulleted(summary: str) -> None:
    """
    严格校验 personal_summary 的条目格式：
    - 仅允许使用 `- ` 作为条目前缀（缺少时直接打回）
    """
    _validate_personal_summary_format(summary)

    lines = (summary or "").splitlines()
    long_header_index = next(
        (i for i, l in enumerate(lines) if (l or "").strip() == _LONG_TERM_HEADER),
        None,
    )
    recent_header_index = next(
        (i for i, l in enumerate(lines) if (l or "").strip() == _RECENT_HEADER),
        None,
    )
    if long_header_index is None or recent_header_index is None:
        raise ValueError("记忆格式不正确，无法定位段落。")

    def _check_section(section_lines: list[str], section_name: str) -> None:
        for raw_line in section_lines:
            if not (raw_line or "").strip():
                continue
            line = (raw_line or "").lstrip()
            if not line.startswith("- "):
                raise ValueError(
                    f"{section_name}中检测到未按 `- ` 开头的条目行：`{raw_line}`"
                )

            text = line[2:].strip()
            if not text:
                raise ValueError(
                    f"{section_name}中检测到空条目：`{raw_line}`（请写成 `- 条目内容`）"
                )

    _check_section(
        section_lines=lines[long_header_index + 1 : recent_header_index],
        section_name="长期记忆",
    )
    _check_section(
        section_lines=lines[recent_header_index + 1 :],
        section_name="近期动态",
    )


async def _get_personal_summary_raw(user_id: int) -> str | None:
    async with AsyncSessionLocal() as session:
        stmt = select(CommunityMemberProfile.personal_summary).where(
            CommunityMemberProfile.discord_id == str(user_id)
        )
        result = await session.execute(stmt)
        return result.scalars().first()


async def _set_personal_summary_raw(user_id: int, new_summary: str | None) -> None:
    async with AsyncSessionLocal() as session:
        async with session.begin():
            result = await session.execute(
                update(CommunityMemberProfile)
                .where(CommunityMemberProfile.discord_id == str(user_id))
                .values(personal_summary=new_summary)
            )
            affected = int(getattr(result, "rowcount", 0) or 0)

    if affected <= 0:
        raise RuntimeError("未找到你的个人档案记录，无法保存记忆。")


async def _apply_personal_summary_update_with_vector_sync(
    user_id: int,
    old_summary: str | None,
    new_summary: str,
) -> dict:
    await _set_personal_summary_raw(user_id, new_summary)

    try:
        return await personal_memory_vector_service.sync_vectors_for_user_strict(
            discord_id=user_id,
            personal_summary=new_summary,
        )
    except Exception:
        try:
            await _set_personal_summary_raw(user_id, old_summary)
        except Exception as rollback_err:
            log.error(
                "记忆向量同步失败，且回滚摘要失败: user_id=%s, err=%s",
                user_id,
                rollback_err,
                exc_info=True,
            )
            raise

        try:
            await personal_memory_vector_service.sync_vectors_for_user(
                discord_id=user_id,
                personal_summary=old_summary,
            )
        except Exception as rollback_vec_err:
            log.error(
                "记忆向量同步失败，摘要已回滚，但向量回滚失败: user_id=%s, err=%s",
                user_id,
                rollback_vec_err,
                exc_info=True,
            )

        raise


def _parse_memory_line_numbers(raw_text: str) -> list[int]:
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("请输入要删除的行号，例如：56,88")

    normalized = text.replace("，", ",").replace("\n", ",")
    parts = [p.strip() for p in re.split(r"[,\s]+", normalized) if p.strip()]
    if not parts:
        raise ValueError("请输入要删除的行号，例如：56,88")

    numbers: list[int] = []
    invalid_parts: list[str] = []
    for part in parts:
        if not re.fullmatch(r"\d+", part):
            invalid_parts.append(part)
            continue
        n = int(part)
        if n <= 0:
            invalid_parts.append(part)
            continue
        numbers.append(n)

    if invalid_parts:
        raise ValueError(f"行号格式不正确：{', '.join(invalid_parts)}（示例：56,88）")

    deduped: list[int] = []
    seen: set[int] = set()
    for n in numbers:
        if n in seen:
            continue
        seen.add(n)
        deduped.append(n)

    return sorted(deduped)


def _extract_long_term_items_from_lines(
    lines: list[str],
    long_header_index: int,
    recent_header_index: int,
) -> list[dict]:
    items: list[dict] = []
    item_no = 0

    for idx in range(long_header_index + 1, recent_header_index):
        raw_line = lines[idx]
        line = (raw_line or "").strip()
        if not line:
            continue

        text = ""
        bullet_match = _BULLET_ITEM_RE.match(line)
        if bullet_match:
            text = bullet_match.group(1).strip()
        else:
            numbered_match = _NUMBERED_ITEM_RE.match(line)
            if numbered_match:
                text = numbered_match.group(1).strip()

        if not text:
            continue

        item_no += 1
        items.append(
            {
                "no": item_no,
                "line_index": idx,
                "text": text,
            }
        )

    return items


def _delete_long_term_memory_by_item_numbers(
    summary: str, item_numbers: list[int]
) -> tuple[str, list[dict], set[str]]:
    """
    Delete long-term memory items (1-based) from a summary string.

    Returns: (new_summary, deleted_items, removed_texts_for_vector_deletion)
    - deleted_items: list of {"no": int, "text": str}
    - removed_texts_for_vector_deletion: unique memory_texts that no longer exist after deletion
    """
    if not summary or not summary.strip():
        raise ValueError("当前没有可删除的长期记忆。")

    lines = summary.splitlines()

    long_header_index = next(
        (i for i, l in enumerate(lines) if (l or "").strip() == _LONG_TERM_HEADER), None
    )
    if long_header_index is None:
        raise ValueError(f"未找到“{_LONG_TERM_HEADER}”段落，无法按行删除。")

    recent_header_index = next(
        (
            i
            for i in range(long_header_index + 1, len(lines))
            if (lines[i] or "").strip() == _RECENT_HEADER
        ),
        len(lines),
    )

    old_items = _extract_long_term_items_from_lines(
        lines=lines,
        long_header_index=long_header_index,
        recent_header_index=recent_header_index,
    )

    if not old_items:
        raise ValueError("当前没有可删除的长期记忆条目。")

    max_no = len(old_items)
    invalid = [n for n in item_numbers if n < 1 or n > max_no]
    if invalid:
        invalid_str = ", ".join(str(x) for x in invalid)
        raise ValueError(f"行号超出范围：{invalid_str}（当前长期记忆共有 {max_no} 条）")

    delete_set = set(item_numbers)
    deleted_items = [{"no": it["no"], "text": it["text"]} for it in old_items if it["no"] in delete_set]
    delete_line_indices = {it["line_index"] for it in old_items if it["no"] in delete_set}

    new_lines = [line for idx, line in enumerate(lines) if idx not in delete_line_indices]
    new_summary = "\n".join(new_lines)

    # Vector deletion should only remove texts that are gone entirely (handles duplicate lines safely)
    old_texts = {str(it.get("text") or "").strip() for it in old_items if str(it.get("text") or "").strip()}

    new_lines_for_parse = new_summary.splitlines()
    new_long_header_index = next(
        (i for i, l in enumerate(new_lines_for_parse) if (l or "").strip() == _LONG_TERM_HEADER),
        None,
    )
    new_recent_header_index = next(
        (
            i
            for i in range((new_long_header_index or 0) + 1, len(new_lines_for_parse))
            if (new_lines_for_parse[i] or "").strip() == _RECENT_HEADER
        ),
        len(new_lines_for_parse),
    )

    new_texts: set[str] = set()
    if new_long_header_index is not None:
        new_items = _extract_long_term_items_from_lines(
            lines=new_lines_for_parse,
            long_header_index=new_long_header_index,
            recent_header_index=new_recent_header_index,
        )
        new_texts = {str(it.get("text") or "").strip() for it in new_items if str(it.get("text") or "").strip()}

    removed_texts = old_texts - new_texts

    return new_summary, deleted_items, removed_texts


def _search_long_term_memory_items(summary: str, keyword: str) -> list[dict]:
    search_text = (keyword or "").strip()
    if not search_text:
        raise ValueError("请输入要搜索的关键词。")

    if not summary or not summary.strip():
        return []

    lines = summary.splitlines()
    long_header_index = next(
        (i for i, l in enumerate(lines) if (l or "").strip() == _LONG_TERM_HEADER),
        None,
    )
    if long_header_index is None:
        return []

    recent_header_index = next(
        (
            i
            for i in range(long_header_index + 1, len(lines))
            if (lines[i] or "").strip() == _RECENT_HEADER
        ),
        len(lines),
    )

    items = _extract_long_term_items_from_lines(
        lines=lines,
        long_header_index=long_header_index,
        recent_header_index=recent_header_index,
    )
    lowered_keyword = search_text.casefold()
    return [
        {"no": it["no"], "text": it["text"]}
        for it in items
        if lowered_keyword in str(it.get("text") or "").casefold()
    ]


def _normalize_long_term_memory_input(raw_text: str) -> list[str]:
    raw_lines = (raw_text or "").splitlines()
    items: list[str] = []
    invalid_lines: list[str] = []

    for raw_line in raw_lines:
        line = (raw_line or "").strip()
        if not line:
            continue

        if (line.startswith("###") and "记忆" in line) or line in {
            _LONG_TERM_HEADER,
            _RECENT_HEADER,
        }:
            invalid_lines.append(raw_line)
            continue

        lstripped = (raw_line or "").lstrip()
        if not lstripped.startswith("- "):
            invalid_lines.append(raw_line)
            continue

        text = lstripped[2:].strip()
        if not text:
            invalid_lines.append(raw_line)
            continue

        items.append(text)

    if invalid_lines or not items:
        example = "- 记忆1\n- 记忆2"
        details = ""
        if invalid_lines:
            preview = "\n".join(invalid_lines[:5])
            details = f"\n\n无法识别的行（示例，最多显示 5 行）：\n```text\n{preview}\n```"
        raise ValueError(
            "输入格式不正确，请按以下格式每行一条长期记忆：\n"
            f"```text\n{example}\n```"
            + details
        )

    return items


def _append_long_term_memory_items(summary: str, new_items: list[str]) -> str:
    base = summary or ""
    if not base.strip():
        base = _build_empty_personal_summary()

    _validate_personal_summary_format(base)

    lines = base.splitlines()
    long_header_index = next(
        (i for i, l in enumerate(lines) if (l or "").strip() == _LONG_TERM_HEADER), None
    )
    recent_header_index = next(
        (i for i, l in enumerate(lines) if (l or "").strip() == _RECENT_HEADER), None
    )
    if long_header_index is None or recent_header_index is None:
        raise ValueError("记忆格式不正确，无法定位长期记忆段落。")

    insert_at = recent_header_index
    while insert_at > long_header_index + 1 and not (lines[insert_at - 1] or "").strip():
        insert_at -= 1

    to_insert = [f"- {it.strip()}" for it in new_items if (it or "").strip()]
    if not to_insert:
        raise ValueError("没有可添加的长期记忆条目。")

    new_lines = list(lines)
    new_lines[insert_at:insert_at] = to_insert
    return "\n".join(new_lines) + ("\n" if base.endswith("\n") else "")


class UserEditMemoryModal(discord.ui.Modal, title="编辑个人记忆"):
    def __init__(self, actor_user_id: int, target_user_id: int, current_memory: str):
        super().__init__()
        self.actor_user_id = actor_user_id
        self.target_user_id = target_user_id

        # 截断过长的记忆以防止 API 报错 (Discord 限制为 4000 字符)
        default_value = current_memory
        warning_msg = None
        if len(default_value) > _MEMORY_PAGE_CHAR_LIMIT:
            warning_msg = (
                f"记忆摘要过长 ({len(default_value)} 字符)，"
                f"已截断至 {_MEMORY_PAGE_CHAR_LIMIT} 字符。"
            )
            default_value = default_value[:_MEMORY_PAGE_CHAR_LIMIT]

        self.memory_input = discord.ui.TextInput(
            label="个人记忆摘要",
            style=discord.TextStyle.paragraph,
            placeholder="在这里输入你的个人记忆...",
            default=default_value,
            required=False,
            max_length=_MEMORY_PAGE_CHAR_LIMIT,
        )
        self.add_item(self.memory_input)

        if warning_msg:
            self.add_item(discord.ui.TextInput(
                label="⚠️ 系统提示",
                default=warning_msg,
                style=discord.TextStyle.short,
                required=False
            ))

    async def on_submit(self, interaction: discord.Interaction):
        if not interaction.user or interaction.user.id != self.actor_user_id:
            await interaction.response.send_message("❌ 这不是你的操作面板。", ephemeral=True)
            return
        if not _can_manage_memory(self.actor_user_id, self.target_user_id):
            await interaction.response.send_message("❌ 你没有权限修改该用户的记忆。", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        draft_summary = self.memory_input.value or ""
        new_summary = draft_summary.strip()
        if not new_summary:
            new_summary = _build_empty_personal_summary()

        try:
            _validate_personal_summary_format(new_summary)
            _validate_personal_summary_items_bulleted(new_summary)
        except ValueError as e:
            view = RetryEditMemoryView(
                actor_user_id=self.actor_user_id,
                target_user_id=self.target_user_id,
                draft_summary=draft_summary,
            )
            await interaction.followup.send(
                f"❌ {e}\n\n正确格式示例：\n```text\n{_build_personal_summary_format_example()}\n```",
                view=view,
                ephemeral=True,
            )
            return

        old_summary = await _get_personal_summary_raw(self.target_user_id)
        try:
            await _apply_personal_summary_update_with_vector_sync(
                user_id=self.target_user_id,
                old_summary=old_summary,
                new_summary=new_summary,
            )
            await interaction.followup.send("✅ 你的个人记忆已成功更新。", ephemeral=True)
        except Exception as e:
            view = RetryEditMemoryView(
                actor_user_id=self.actor_user_id,
                target_user_id=self.target_user_id,
                draft_summary=draft_summary,
            )
            log.error(f"用户 {self.actor_user_id} 更新用户 {self.target_user_id} 记忆失败（已回滚）: {e}", exc_info=True)
            await interaction.followup.send(
                f"❌ 向量同步失败，修改未生效（已回滚）。\n错误：{e}\n\n请重新填写。",
                view=view,
                ephemeral=True,
            )


class RetryEditMemoryView(discord.ui.View):
    def __init__(self, actor_user_id: int, target_user_id: int, draft_summary: str):
        super().__init__(timeout=120)
        self.actor_user_id = actor_user_id
        self.target_user_id = target_user_id
        self.draft_summary = draft_summary

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user and interaction.user.id == self.actor_user_id:
            return True
        await interaction.response.send_message("❌ 这不是你的操作面板。", ephemeral=True)
        return False

    @discord.ui.button(label="重新填写", style=discord.ButtonStyle.primary, emoji="✏️")
    async def retry(self, interaction: discord.Interaction, button: discord.ui.Button):
        modal = UserEditMemoryModal(self.actor_user_id, self.target_user_id, self.draft_summary)
        await interaction.response.send_modal(modal)


class PagedMemoryPreviewView(discord.ui.View):
    def __init__(
        self,
        actor_user_id: int,
        target_user_id: int,
        summary_snapshot: str,
    ):
        super().__init__(timeout=_MEMORY_PREVIEW_TIMEOUT)
        self.actor_user_id = actor_user_id
        self.target_user_id = target_user_id
        self.summary_snapshot = summary_snapshot
        self.pages = _split_memory_summary_into_pages(summary_snapshot)
        self.current_page = 0
        self.message: discord.Message | None = None

        self.first_button = discord.ui.Button(
            emoji="⏮️",
            style=discord.ButtonStyle.secondary,
            row=0,
        )
        self.first_button.callback = self.go_to_first_page
        self.add_item(self.first_button)

        self.prev_button = discord.ui.Button(
            emoji="⏪",
            style=discord.ButtonStyle.secondary,
            row=0,
        )
        self.prev_button.callback = self.go_to_previous_page
        self.add_item(self.prev_button)

        self.edit_button = discord.ui.Button(
            label="编辑当前",
            style=discord.ButtonStyle.primary,
            row=0,
        )
        self.edit_button.callback = self.edit_current_page
        self.add_item(self.edit_button)

        self.next_button = discord.ui.Button(
            emoji="⏩",
            style=discord.ButtonStyle.secondary,
            row=0,
        )
        self.next_button.callback = self.go_to_next_page
        self.add_item(self.next_button)

        self.last_button = discord.ui.Button(
            emoji="⏭️",
            style=discord.ButtonStyle.secondary,
            row=0,
        )
        self.last_button.callback = self.go_to_last_page
        self.add_item(self.last_button)

        self._sync_button_state()

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user and interaction.user.id == self.actor_user_id:
            return True
        await interaction.response.send_message("❌ 这不是你的操作面板。", ephemeral=True)
        return False

    def _sync_button_state(self) -> None:
        total_pages = len(self.pages)
        is_first_page = self.current_page <= 0
        is_last_page = self.current_page >= total_pages - 1

        self.first_button.disabled = is_first_page
        self.prev_button.disabled = is_first_page
        self.next_button.disabled = is_last_page
        self.last_button.disabled = is_last_page
        self.edit_button.disabled = total_pages <= 0

    def _clamp_page_index(self) -> None:
        if not self.pages:
            self.pages = [""]
        self.current_page = max(0, min(self.current_page, len(self.pages) - 1))

    def build_embed(self) -> discord.Embed:
        self._clamp_page_index()
        self._sync_button_state()

        page_text = self.pages[self.current_page]
        embed = discord.Embed(
            title="个人记忆分页预览",
            description=page_text or "\u200b",
            color=discord.Color.blurple(),
        )
        embed.set_footer(
            text=(
                f"第 {self.current_page + 1} / {len(self.pages)} 页"
                f" | 本页 {len(page_text)} 字符"
                f" | 全文 {len(self.summary_snapshot)} 字符"
            )
        )
        return embed

    async def update_message(self) -> None:
        self._clamp_page_index()
        self._sync_button_state()
        if self.message is None:
            return
        await self.message.edit(embed=self.build_embed(), view=self)

    async def refresh_summary(self, new_summary: str) -> None:
        self.summary_snapshot = new_summary
        self.pages = _split_memory_summary_into_pages(new_summary)
        self._clamp_page_index()
        await self.update_message()

    async def go_to_first_page(self, interaction: discord.Interaction) -> None:
        self.current_page = 0
        await interaction.response.edit_message(embed=self.build_embed(), view=self)

    async def go_to_previous_page(self, interaction: discord.Interaction) -> None:
        if self.current_page > 0:
            self.current_page -= 1
        await interaction.response.edit_message(embed=self.build_embed(), view=self)

    async def edit_current_page(self, interaction: discord.Interaction) -> None:
        self._clamp_page_index()
        modal = EditMemoryPageModal(
            preview_view=self,
            page_index=self.current_page,
            summary_snapshot=self.summary_snapshot,
            current_page_text=self.pages[self.current_page],
        )
        await interaction.response.send_modal(modal)

    async def go_to_next_page(self, interaction: discord.Interaction) -> None:
        if self.current_page < len(self.pages) - 1:
            self.current_page += 1
        await interaction.response.edit_message(embed=self.build_embed(), view=self)

    async def go_to_last_page(self, interaction: discord.Interaction) -> None:
        self.current_page = len(self.pages) - 1
        await interaction.response.edit_message(embed=self.build_embed(), view=self)

    async def on_timeout(self) -> None:
        for child in self.children:
            child.disabled = True
        if self.message is not None:
            try:
                await self.message.edit(view=self)
            except Exception:
                log.debug("分页记忆预览超时后禁用按钮失败。", exc_info=True)


class EditMemoryPageModal(discord.ui.Modal):
    def __init__(
        self,
        preview_view: PagedMemoryPreviewView,
        page_index: int,
        summary_snapshot: str,
        current_page_text: str,
    ):
        super().__init__(title=f"编辑当前页（第 {page_index + 1} 页）")
        self.preview_view = preview_view
        self.actor_user_id = preview_view.actor_user_id
        self.target_user_id = preview_view.target_user_id
        self.page_index = page_index
        self.summary_snapshot = summary_snapshot

        self.page_input = discord.ui.TextInput(
            label="当前页记忆",
            style=discord.TextStyle.paragraph,
            default=current_page_text,
            required=False,
            max_length=_MEMORY_PAGE_CHAR_LIMIT,
        )
        self.add_item(self.page_input)

    async def on_submit(self, interaction: discord.Interaction):
        if not interaction.user or interaction.user.id != self.actor_user_id:
            await interaction.response.send_message("❌ 这不是你的操作面板。", ephemeral=True)
            return
        if not _can_manage_memory(self.actor_user_id, self.target_user_id):
            await interaction.response.send_message("❌ 你没有权限修改该用户的记忆。", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True)

        latest_summary_raw = await _get_personal_summary_raw(self.target_user_id)
        latest_summary = latest_summary_raw or ""
        if latest_summary != self.summary_snapshot:
            await interaction.followup.send(
                "❌ 当前记忆已被其他操作改动，请重新打开分页预览后再编辑。",
                ephemeral=True,
            )
            return

        draft_page = self.page_input.value or ""

        try:
            new_summary = _replace_memory_page(
                summary=self.summary_snapshot,
                page_index=self.page_index,
                new_page_text=draft_page,
            )
            _validate_personal_summary_format(new_summary)
            _validate_personal_summary_items_bulleted(new_summary)
        except (IndexError, ValueError) as e:
            view = RetryEditMemoryPageView(
                preview_view=self.preview_view,
                page_index=self.page_index,
                summary_snapshot=self.summary_snapshot,
                draft_page_text=draft_page,
            )
            await interaction.followup.send(
                f"❌ 当前页修改会破坏整份记忆格式：{e}\n\n正确格式示例：\n```text\n{_build_personal_summary_format_example()}\n```",
                view=view,
                ephemeral=True,
            )
            return

        try:
            await _apply_personal_summary_update_with_vector_sync(
                user_id=self.target_user_id,
                old_summary=latest_summary_raw,
                new_summary=new_summary,
            )
        except Exception as e:
            view = RetryEditMemoryPageView(
                preview_view=self.preview_view,
                page_index=self.page_index,
                summary_snapshot=self.summary_snapshot,
                draft_page_text=draft_page,
            )
            log.error(
                "用户 %s 分页编辑用户 %s 的记忆失败（已回滚）: %s",
                self.actor_user_id,
                self.target_user_id,
                e,
                exc_info=True,
            )
            await interaction.followup.send(
                f"❌ 向量同步失败，当前页修改未生效（已回滚）。\n错误：{e}",
                view=view,
                ephemeral=True,
            )
            return

        success_message = "✅ 当前页已成功更新，分页预览已刷新。"
        try:
            await self.preview_view.refresh_summary(new_summary)
        except Exception as refresh_err:
            success_message = "✅ 当前页已成功更新，但预览刷新失败，请重新打开分页预览查看。"
            log.warning(
                "分页记忆预览刷新失败: actor_user_id=%s, target_user_id=%s, err=%s",
                self.actor_user_id,
                self.target_user_id,
                refresh_err,
                exc_info=True,
            )

        await interaction.followup.send(success_message, ephemeral=True)


class RetryEditMemoryPageView(discord.ui.View):
    def __init__(
        self,
        preview_view: PagedMemoryPreviewView,
        page_index: int,
        summary_snapshot: str,
        draft_page_text: str,
    ):
        super().__init__(timeout=120)
        self.preview_view = preview_view
        self.actor_user_id = preview_view.actor_user_id
        self.page_index = page_index
        self.summary_snapshot = summary_snapshot
        self.draft_page_text = draft_page_text

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user and interaction.user.id == self.actor_user_id:
            return True
        await interaction.response.send_message("❌ 这不是你的操作面板。", ephemeral=True)
        return False

    @discord.ui.button(label="重新编辑当前页", style=discord.ButtonStyle.primary, emoji="✏️")
    async def retry(self, interaction: discord.Interaction, button: discord.ui.Button):
        modal = EditMemoryPageModal(
            preview_view=self.preview_view,
            page_index=self.page_index,
            summary_snapshot=self.summary_snapshot,
            current_page_text=self.draft_page_text,
        )
        await interaction.response.send_modal(modal)


class AddLongTermMemoryModal(discord.ui.Modal, title="增加长期记忆"):
    def __init__(self, actor_user_id: int, target_user_id: int, default_text: str = ""):
        super().__init__()
        self.actor_user_id = actor_user_id
        self.target_user_id = target_user_id

        self.memories_input = discord.ui.TextInput(
            label="分行输入长期记忆",
            style=discord.TextStyle.paragraph,
            placeholder="- 记忆1\n- 记忆2",
            default=(default_text or "")[:4000],
            required=True,
            max_length=4000,
        )
        self.add_item(self.memories_input)

    async def on_submit(self, interaction: discord.Interaction):
        if not interaction.user or interaction.user.id != self.actor_user_id:
            await interaction.response.send_message("❌ 这不是你的操作面板。", ephemeral=True)
            return
        if not _can_manage_memory(self.actor_user_id, self.target_user_id):
            await interaction.response.send_message("❌ 你没有权限修改该用户的记忆。", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)

        try:
            items = _normalize_long_term_memory_input(self.memories_input.value)
        except ValueError as e:
            view = RetryAddLongTermMemoryView(
                actor_user_id=self.actor_user_id,
                target_user_id=self.target_user_id,
                draft_text=self.memories_input.value,
            )
            await interaction.followup.send(f"❌ {e}", view=view, ephemeral=True)
            return

        old_summary = await _get_personal_summary_raw(self.target_user_id)
        base_summary = old_summary or ""
        if base_summary.strip():
            try:
                _validate_personal_summary_format(base_summary)
                _validate_personal_summary_items_bulleted(base_summary)
            except ValueError as e:
                view = RetryAddLongTermMemoryView(
                    actor_user_id=self.actor_user_id,
                    target_user_id=self.target_user_id,
                    draft_text=self.memories_input.value,
                )
                await interaction.followup.send(
                    f"❌ 当前记忆摘要格式不正确，无法追加长期记忆：{e}\n\n请先使用“查看并修改记忆”修复格式。",
                    view=view,
                    ephemeral=True,
                )
                return
        else:
            base_summary = _build_empty_personal_summary()

        try:
            new_summary = _append_long_term_memory_items(base_summary, items)
            _validate_personal_summary_items_bulleted(new_summary)
        except ValueError as e:
            view = RetryAddLongTermMemoryView(
                actor_user_id=self.actor_user_id,
                target_user_id=self.target_user_id,
                draft_text=self.memories_input.value,
            )
            await interaction.followup.send(f"❌ {e}", view=view, ephemeral=True)
            return

        preview_lines = "\n".join([f"- {t}" for t in items])
        if len(preview_lines) > 1600:
            preview_lines = preview_lines[:1600] + "\n...（预览已截断）"

        content = (
            "你将追加以下【长期记忆】条目（会加在长期记忆列表末尾，不影响“近期动态”）：\n\n"
            f"```text\n{preview_lines}\n```\n"
            "确认要增加吗？"
        )
        view = ConfirmAddLongTermMemoryView(
            actor_user_id=self.actor_user_id,
            target_user_id=self.target_user_id,
            old_summary=old_summary,
            new_summary=new_summary,
            added_items=items,
        )
        await interaction.followup.send(content, view=view, ephemeral=True)


class RetryAddLongTermMemoryView(discord.ui.View):
    def __init__(self, actor_user_id: int, target_user_id: int, draft_text: str):
        super().__init__(timeout=120)
        self.actor_user_id = actor_user_id
        self.target_user_id = target_user_id
        self.draft_text = draft_text

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user and interaction.user.id == self.actor_user_id:
            return True
        await interaction.response.send_message("❌ 这不是你的操作面板。", ephemeral=True)
        return False

    @discord.ui.button(label="重新填写", style=discord.ButtonStyle.primary, emoji="✏️")
    async def retry(self, interaction: discord.Interaction, button: discord.ui.Button):
        modal = AddLongTermMemoryModal(
            self.actor_user_id,
            self.target_user_id,
            default_text=self.draft_text,
        )
        await interaction.response.send_modal(modal)


class ConfirmAddLongTermMemoryView(discord.ui.View):
    def __init__(
        self,
        actor_user_id: int,
        target_user_id: int,
        old_summary: str | None,
        new_summary: str,
        added_items: list[str],
    ):
        super().__init__(timeout=120)
        self.actor_user_id = actor_user_id
        self.target_user_id = target_user_id
        self.old_summary = old_summary
        self.new_summary = new_summary
        self.added_items = added_items

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user and interaction.user.id == self.actor_user_id:
            return True
        await interaction.response.send_message("❌ 这不是你的操作面板。", ephemeral=True)
        return False

    @discord.ui.button(label="增加", style=discord.ButtonStyle.success, emoji="➕")
    async def confirm_add(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not _can_manage_memory(self.actor_user_id, self.target_user_id):
            await interaction.response.send_message("❌ 你没有权限修改该用户的记忆。", ephemeral=True)
            return
        await interaction.response.defer()

        try:
            sync_result = await _apply_personal_summary_update_with_vector_sync(
                user_id=self.target_user_id,
                old_summary=self.old_summary,
                new_summary=self.new_summary,
            )
        except Exception as e:
            log.error(f"用户 {self.actor_user_id} 追加用户 {self.target_user_id} 长期记忆失败（已回滚）: {e}", exc_info=True)
            await interaction.followup.send(
                f"❌ 向量同步失败，追加未生效（已回滚）。\n错误：{e}",
                ephemeral=True,
            )
            return

        inserted = int(sync_result.get("inserted", 0) or 0) if isinstance(sync_result, dict) else 0
        deleted = int(sync_result.get("deleted", 0) or 0) if isinstance(sync_result, dict) else 0
        deduped = int(sync_result.get("deduped", 0) or 0) if isinstance(sync_result, dict) else 0
        success_content = (
            f"✅ 已增加 {len(self.added_items)} 条长期记忆"
            f"（向量 inserted={inserted}, deleted={deleted}, deduped={deduped}）。"
        )
        await interaction.edit_original_response(content=success_content, view=None)

    @discord.ui.button(label="返回", style=discord.ButtonStyle.secondary, emoji="⬅️")
    async def go_back(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(content="已取消增加。", view=None)


class SearchMemoryModal(discord.ui.Modal, title="搜索记忆"):
    def __init__(self, actor_user_id: int, target_user_id: int):
        super().__init__()
        self.actor_user_id = actor_user_id
        self.target_user_id = target_user_id

        self.keyword_input = discord.ui.TextInput(
            label="输入要搜索的关键词",
            placeholder="例如：约定",
            required=True,
            max_length=200,
        )
        self.add_item(self.keyword_input)

    async def on_submit(self, interaction: discord.Interaction):
        if not interaction.user or interaction.user.id != self.actor_user_id:
            await interaction.response.send_message("❌ 这不是你的操作面板。", ephemeral=True)
            return
        if not _can_manage_memory(self.actor_user_id, self.target_user_id):
            await interaction.response.send_message("❌ 你没有权限查看该用户的记忆。", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)

        keyword = (self.keyword_input.value or "").strip()
        if not keyword:
            await interaction.followup.send("❌ 请输入要搜索的关键词。", ephemeral=True)
            return

        try:
            current_summary = await _get_personal_summary_raw(self.target_user_id) or ""
            matched_items = _search_long_term_memory_items(current_summary, keyword)
        except ValueError as e:
            await interaction.followup.send(f"❌ {e}", ephemeral=True)
            return
        except Exception as e:
            log.error(f"用户 {self.actor_user_id} 搜索用户 {self.target_user_id} 记忆失败: {e}", exc_info=True)
            await interaction.followup.send(f"❌ 搜索失败: {e}", ephemeral=True)
            return

        if matched_items:
            preview = "\n".join(
                [f"{item['no']}. {item['text']}" for item in matched_items]
            )
            if len(preview) > 1600:
                preview = preview[:1600] + "\n...（结果已截断）"
            content = (
                f"搜索关键词：`{keyword}`\n\n"
                "匹配到以下【长期记忆】条目：\n\n"
                f"```text\n{preview}\n```"
            )
        else:
            content = (
                f"搜索关键词：`{keyword}`\n\n"
                "未找到匹配的【长期记忆】条目。"
            )

        view = SearchMemoryResultsView(
            actor_user_id=self.actor_user_id,
            target_user_id=self.target_user_id,
        )
        await interaction.followup.send(content, view=view, ephemeral=True)


class SearchMemoryResultsView(discord.ui.View):
    def __init__(self, actor_user_id: int, target_user_id: int):
        super().__init__(timeout=120)
        self.actor_user_id = actor_user_id
        self.target_user_id = target_user_id

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user and interaction.user.id == self.actor_user_id:
            return True
        await interaction.response.send_message("❌ 这不是你的操作面板。", ephemeral=True)
        return False

    @discord.ui.button(label="删除特定记忆", style=discord.ButtonStyle.danger, emoji="🗑️")
    async def delete_specific_memory(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        modal = DeleteSpecificMemoryModal(
            actor_user_id=self.actor_user_id,
            target_user_id=self.target_user_id,
        )
        await interaction.response.send_modal(modal)

    @discord.ui.button(label="返回", style=discord.ButtonStyle.secondary, emoji="⬅️")
    async def go_back(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(content="已退出搜索。", view=None)


class DeleteSpecificMemoryModal(discord.ui.Modal, title="删除特定记忆"):
    def __init__(self, actor_user_id: int, target_user_id: int):
        super().__init__()
        self.actor_user_id = actor_user_id
        self.target_user_id = target_user_id

        self.line_numbers_input = discord.ui.TextInput(
            label="删除第几行长期记忆？",
            placeholder="例如：56,88",
            required=True,
            max_length=200,
        )
        self.add_item(self.line_numbers_input)

    async def on_submit(self, interaction: discord.Interaction):
        if not interaction.user or interaction.user.id != self.actor_user_id:
            await interaction.response.send_message("❌ 这不是你的操作面板。", ephemeral=True)
            return
        if not _can_manage_memory(self.actor_user_id, self.target_user_id):
            await interaction.response.send_message("❌ 你没有权限修改该用户的记忆。", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)

        try:
            item_numbers = _parse_memory_line_numbers(self.line_numbers_input.value)
        except ValueError as e:
            await interaction.followup.send(f"❌ {e}", ephemeral=True)
            return

        try:
            current_summary = await _get_personal_summary_raw(self.target_user_id) or ""

            new_summary, deleted_items, removed_texts = _delete_long_term_memory_by_item_numbers(
                summary=current_summary,
                item_numbers=item_numbers,
            )
        except ValueError as e:
            await interaction.followup.send(f"❌ {e}", ephemeral=True)
            return
        except Exception as e:
            log.error(f"用户 {self.actor_user_id} 解析/删除用户 {self.target_user_id} 记忆失败: {e}", exc_info=True)
            await interaction.followup.send(f"❌ 操作失败: {e}", ephemeral=True)
            return

        if not deleted_items:
            await interaction.followup.send("❌ 未匹配到任何可删除的长期记忆条目。", ephemeral=True)
            return

        items_preview = "\n".join([f"{it['no']}. {it['text']}" for it in deleted_items])
        content = (
            "你将删除以下【长期记忆】条目（不会影响“近期动态”）：\n\n"
            f"```text\n{items_preview}\n```\n"
            "确认要删除吗？"
        )

        view = ConfirmDeleteSpecificMemoryView(
            actor_user_id=self.actor_user_id,
            target_user_id=self.target_user_id,
            old_summary=current_summary,
            new_summary=new_summary,
            deleted_items=deleted_items,
            removed_texts=removed_texts,
        )
        await interaction.followup.send(content, view=view, ephemeral=True)


class ConfirmDeleteSpecificMemoryView(discord.ui.View):
    def __init__(
        self,
        actor_user_id: int,
        target_user_id: int,
        old_summary: str,
        new_summary: str,
        deleted_items: list[dict],
        removed_texts: set[str],
    ):
        super().__init__(timeout=120)
        self.actor_user_id = actor_user_id
        self.target_user_id = target_user_id
        self.old_summary = old_summary
        self.new_summary = new_summary
        self.deleted_items = deleted_items
        self.removed_texts = removed_texts

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user and interaction.user.id == self.actor_user_id:
            return True
        await interaction.response.send_message("❌ 这不是你的操作面板。", ephemeral=True)
        return False

    @discord.ui.button(label="删除", style=discord.ButtonStyle.danger, emoji="🗑️")
    async def confirm_delete(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        if not _can_manage_memory(self.actor_user_id, self.target_user_id):
            await interaction.response.send_message("❌ 你没有权限修改该用户的记忆。", ephemeral=True)
            return
        await interaction.response.defer()

        try:
            sync_result = await _apply_personal_summary_update_with_vector_sync(
                user_id=self.target_user_id,
                old_summary=self.old_summary,
                new_summary=self.new_summary,
            )
        except Exception as e:
            log.error(f"用户 {self.actor_user_id} 删除用户 {self.target_user_id} 记忆失败: {e}", exc_info=True)
            await interaction.followup.send(f"❌ 删除失败: {e}", ephemeral=True)
            return

        deleted_vectors = int(sync_result.get("deleted", 0) or 0) if isinstance(sync_result, dict) else 0
        success_content = (
            f"✅ 已删除 {len(self.deleted_items)} 条长期记忆"
            f"（向量同步删除 {deleted_vectors} 条）。"
        )

        await interaction.edit_original_response(content=success_content, view=None)

    @discord.ui.button(label="返回", style=discord.ButtonStyle.secondary, emoji="⬅️")
    async def go_back(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(content="已取消删除。", view=None)


class MemoryView(discord.ui.View):
    def __init__(
        self,
        actor_user_id: int,
        target_user_id: int,
        current_memory: str,
        source_interaction: discord.Interaction,
        current_summary_raw: str | None = None,
    ):
        super().__init__(timeout=60)
        self.actor_user_id = actor_user_id
        self.target_user_id = target_user_id
        self.current_memory = current_memory
        self.current_summary_raw = current_summary_raw
        self.source_interaction = source_interaction

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user and interaction.user.id == self.actor_user_id:
            return True
        await interaction.response.send_message("❌ 这不是你的操作面板。", ephemeral=True)
        return False

    @discord.ui.button(label="查看并修改记忆", style=discord.ButtonStyle.success, emoji="🧠")
    async def edit_memory(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not _can_manage_memory(self.actor_user_id, self.target_user_id):
            await interaction.response.send_message("❌ 你没有权限修改该用户的记忆。", ephemeral=True)
            return

        current_summary_raw = await _get_personal_summary_raw(self.target_user_id)
        current_memory = _prepare_memory_summary_for_view(current_summary_raw)
        self.current_memory = current_memory
        self.current_summary_raw = current_summary_raw

        if len(current_memory) > _MEMORY_PAGE_CHAR_LIMIT:
            preview_view = PagedMemoryPreviewView(
                actor_user_id=self.actor_user_id,
                target_user_id=self.target_user_id,
                summary_snapshot=current_memory,
            )
            await interaction.response.send_message(
                content="记忆摘要超过 4000 字符，已切换为分页预览。",
                embed=preview_view.build_embed(),
                view=preview_view,
                ephemeral=True,
            )
            preview_view.message = await interaction.original_response()
            return

        modal = UserEditMemoryModal(
            self.actor_user_id,
            self.target_user_id,
            self.current_memory,
        )
        await interaction.response.send_modal(modal)

    @discord.ui.button(label="切换用户", style=discord.ButtonStyle.primary, emoji="🔁")
    async def switch_user(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id not in DEVELOPER_USER_IDS:
            await interaction.response.send_message("❌ 你没有权限切换用户。", ephemeral=True)
            return
        await interaction.response.send_modal(SwitchUserModal(self))

    @discord.ui.button(label="增加长期记忆", style=discord.ButtonStyle.success, emoji="➡️", row=1)
    async def add_long_term_memory(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        modal = AddLongTermMemoryModal(
            actor_user_id=self.actor_user_id,
            target_user_id=self.target_user_id,
        )
        await interaction.response.send_modal(modal)

    @discord.ui.button(
        label="删除特定记忆", style=discord.ButtonStyle.danger, emoji="🗑️", row=2
    )
    async def delete_specific_memory(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        modal = DeleteSpecificMemoryModal(
            actor_user_id=self.actor_user_id,
            target_user_id=self.target_user_id,
        )
        await interaction.response.send_modal(modal)

    @discord.ui.button(
        label="搜索", style=discord.ButtonStyle.primary, emoji="🔎", row=2
    )
    async def search_memory(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        modal = SearchMemoryModal(
            actor_user_id=self.actor_user_id,
            target_user_id=self.target_user_id,
        )
        await interaction.response.send_modal(modal)

class SwitchUserModal(discord.ui.Modal, title="切换用户"):
    def __init__(self, parent_view: MemoryView):
        super().__init__()
        self.parent_view = parent_view

        self.user_id_input = discord.ui.TextInput(
            label="Discord ID",
            placeholder="123456789012345678",
            required=True,
            max_length=30,
        )
        self.add_item(self.user_id_input)

    async def on_submit(self, interaction: discord.Interaction):
        raw_user_id = (self.user_id_input.value or "").strip()
        if not raw_user_id.isdigit():
            await interaction.response.send_message("未找到当前用户记忆", ephemeral=True)
            return

        target_user_id = int(raw_user_id)
        if not await _personal_memory_profile_exists(target_user_id):
            await interaction.response.send_message("未找到当前用户记忆", ephemeral=True)
            return

        current_summary_raw = await _get_personal_summary_raw(target_user_id)
        current_memory = _prepare_memory_summary_for_view(current_summary_raw)

        self.parent_view.target_user_id = target_user_id
        self.parent_view.current_memory = current_memory
        self.parent_view.current_summary_raw = current_summary_raw

        await self.parent_view.source_interaction.edit_original_response(
            content=_build_memory_view_content(target_user_id, current_summary_raw),
            view=self.parent_view,
        )
        await interaction.response.send_message("切换成功", ephemeral=True)


class MemoryCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    # @app_commands.command(name="查看并修改记忆", description="查看并编辑你在这个服务器的个人记忆")
    # async def view_and_edit_memory(self, interaction: discord.Interaction):
    #     user_id = interaction.user.id
        
        try:
            current_summary_raw = await _get_personal_summary_raw(user_id)
            current_summary = _prepare_memory_summary_for_view(current_summary_raw)
                
    #         view = MemoryView(
    #             actor_user_id=user_id,
    #             target_user_id=user_id,
    #             current_memory=current_summary,
    #             source_interaction=interaction,
    #             current_summary_raw=current_summary_raw,
    #         )
    #         content = _build_memory_view_content(user_id, current_summary_raw)
    #         await interaction.response.send_message(content, view=view, ephemeral=True)
            
    #     except Exception as e:
    #         log.error(f"获取用户 {user_id} 记忆失败: {e}", exc_info=True)
    #         await interaction.response.send_message(f"无法获取记忆: {e}", ephemeral=True)

async def setup(bot):
    await bot.add_cog(MemoryCommands(bot))
