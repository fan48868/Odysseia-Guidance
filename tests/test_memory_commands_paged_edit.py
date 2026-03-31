import asyncio
import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _install_memory_command_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    sqlalchemy_stub = ModuleType("sqlalchemy")
    sqlalchemy_stub.select = MagicMock(name="select")
    sqlalchemy_stub.update = MagicMock(name="update")
    monkeypatch.setitem(sys.modules, "sqlalchemy", sqlalchemy_stub)

    vector_service_stub = ModuleType(
        "src.chat.features.personal_memory.services.personal_memory_vector_service"
    )
    vector_service_stub.personal_memory_vector_service = SimpleNamespace(
        sync_vectors_for_user_strict=AsyncMock(),
        sync_vectors_for_user=AsyncMock(),
    )
    monkeypatch.setitem(
        sys.modules,
        "src.chat.features.personal_memory.services.personal_memory_vector_service",
        vector_service_stub,
    )

    database_stub = ModuleType("src.database.database")
    database_stub.AsyncSessionLocal = MagicMock(name="AsyncSessionLocal")
    monkeypatch.setitem(sys.modules, "src.database.database", database_stub)

    models_stub = ModuleType("src.database.models")

    class _DummyCommunityMemberProfile:
        discord_id = "discord_id"
        personal_summary = "personal_summary"

    models_stub.CommunityMemberProfile = _DummyCommunityMemberProfile
    monkeypatch.setitem(sys.modules, "src.database.models", models_stub)

    config_stub = ModuleType("src.config")
    config_stub.DEVELOPER_USER_IDS = set()
    monkeypatch.setitem(sys.modules, "src.config", config_stub)


@pytest.fixture
def memory_commands(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    _install_memory_command_stubs(monkeypatch)
    sys.modules.pop("src.chat.cogs.memory_commands", None)
    importlib.invalidate_caches()
    return importlib.import_module("src.chat.cogs.memory_commands")


def _build_long_valid_summary(memory_commands, *, min_length: int) -> str:
    long_term_lines: list[str] = []
    recent_lines: list[str] = []

    index = 1
    summary = memory_commands._build_empty_personal_summary()
    while len(summary) <= min_length:
        long_term_lines.append(f"- 长期记忆条目{index:04d}：" + ("甲" * 32))
        recent_lines.append(f"- 近期动态条目{index:04d}：" + ("乙" * 16))
        long_term_text = "\n".join(long_term_lines)
        recent_text = "\n".join(recent_lines)
        summary = (
            f"{memory_commands._LONG_TERM_HEADER}\n"
            f"{long_term_text}\n"
            f"{memory_commands._RECENT_HEADER}\n"
            f"{recent_text}\n"
        )
        index += 1

    return summary


def _make_component_interaction(user_id: int):
    return SimpleNamespace(
        user=SimpleNamespace(id=user_id),
        response=SimpleNamespace(
            send_message=AsyncMock(),
            send_modal=AsyncMock(),
            edit_message=AsyncMock(),
            defer=AsyncMock(),
        ),
        followup=SimpleNamespace(send=AsyncMock()),
        original_response=AsyncMock(),
    )


def test_split_memory_summary_into_pages_preserves_text_and_line_boundaries(
    memory_commands,
):
    summary = _build_long_valid_summary(
        memory_commands,
        min_length=memory_commands._MEMORY_PAGE_CHAR_LIMIT + 800,
    )

    pages = memory_commands._split_memory_summary_into_pages(summary)

    assert len(pages) > 1
    assert "".join(pages) == summary
    assert all(
        len(page) <= memory_commands._MEMORY_PAGE_CHAR_LIMIT for page in pages
    )
    assert all(page.endswith("\n") for page in pages[:-1])


def test_split_memory_summary_into_pages_hard_splits_long_line(memory_commands):
    summary = (
        f"{memory_commands._LONG_TERM_HEADER}\n"
        f"- {'甲' * (memory_commands._MEMORY_PAGE_CHAR_LIMIT + 180)}\n"
        f"{memory_commands._RECENT_HEADER}\n"
        "- 最近记忆1\n"
    )

    pages = memory_commands._split_memory_summary_into_pages(summary)

    assert len(pages) > 1
    assert "".join(pages) == summary
    assert all(
        len(page) <= memory_commands._MEMORY_PAGE_CHAR_LIMIT for page in pages
    )
    assert any(not page.endswith("\n") for page in pages[:-1])


def test_replace_memory_page_only_updates_target_page(memory_commands):
    summary = _build_long_valid_summary(
        memory_commands,
        min_length=memory_commands._MEMORY_PAGE_CHAR_LIMIT * 2 + 500,
    )
    pages = memory_commands._split_memory_summary_into_pages(summary)
    replacement = pages[1].replace("长期记忆条目", "长期记忆已修改条目", 1)

    new_summary = memory_commands._replace_memory_page(summary, 1, replacement)

    expected = "".join([pages[0], replacement, *pages[2:]])
    assert new_summary == expected


def test_memory_view_uses_modal_for_short_summary(
    memory_commands,
    monkeypatch: pytest.MonkeyPatch,
):
    async def _run():
        summary = (
            f"{memory_commands._LONG_TERM_HEADER}\n"
            "- 条目1\n"
            f"{memory_commands._RECENT_HEADER}\n"
            "- 条目2\n"
        )
        interaction = _make_component_interaction(user_id=123)
        modal = object()

        monkeypatch.setattr(
            memory_commands,
            "_get_personal_summary_raw",
            AsyncMock(return_value=summary),
        )
        modal_factory = MagicMock(return_value=modal)
        monkeypatch.setattr(memory_commands, "UserEditMemoryModal", modal_factory)

        view = memory_commands.MemoryView(
            actor_user_id=123,
            target_user_id=123,
            current_memory=summary,
            source_interaction=SimpleNamespace(),
        )

        await view.edit_memory.callback(interaction)

        modal_factory.assert_called_once_with(123, 123, summary)
        interaction.response.send_modal.assert_awaited_once_with(modal)
        interaction.response.send_message.assert_not_called()

    asyncio.run(_run())


def test_memory_view_uses_paged_preview_for_long_summary(
    memory_commands,
    monkeypatch: pytest.MonkeyPatch,
):
    async def _run():
        summary = _build_long_valid_summary(
            memory_commands,
            min_length=memory_commands._MEMORY_PAGE_CHAR_LIMIT + 1200,
        )
        preview_message = AsyncMock()
        interaction = _make_component_interaction(user_id=123)
        interaction.original_response = AsyncMock(return_value=preview_message)

        monkeypatch.setattr(
            memory_commands,
            "_get_personal_summary_raw",
            AsyncMock(return_value=summary),
        )

        view = memory_commands.MemoryView(
            actor_user_id=123,
            target_user_id=123,
            current_memory=summary,
            source_interaction=SimpleNamespace(),
        )

        await view.edit_memory.callback(interaction)

        interaction.response.send_modal.assert_not_called()
        send_kwargs = interaction.response.send_message.await_args.kwargs
        preview_view = send_kwargs["view"]

        assert send_kwargs["ephemeral"] is True
        assert isinstance(preview_view, memory_commands.PagedMemoryPreviewView)
        assert preview_view.message is preview_message
        assert [
            str(child.emoji) if child.emoji else child.label
            for child in preview_view.children
        ] == ["⏮️", "⏪", "编辑当前", "⏩", "⏭️"]

    asyncio.run(_run())


def test_edit_memory_page_modal_updates_only_current_page(
    memory_commands,
    monkeypatch: pytest.MonkeyPatch,
):
    async def _run():
        summary = _build_long_valid_summary(
            memory_commands,
            min_length=memory_commands._MEMORY_PAGE_CHAR_LIMIT * 2 + 500,
        )
        pages = memory_commands._split_memory_summary_into_pages(summary)
        replacement = pages[1].replace("长期记忆条目", "长期记忆条目已更新", 1)
        preview_view = memory_commands.PagedMemoryPreviewView(
            actor_user_id=1,
            target_user_id=1,
            summary_snapshot=summary,
        )
        preview_view.refresh_summary = AsyncMock()

        monkeypatch.setattr(
            memory_commands,
            "_get_personal_summary_raw",
            AsyncMock(return_value=summary),
        )
        apply_mock = AsyncMock()
        monkeypatch.setattr(
            memory_commands,
            "_apply_personal_summary_update_with_vector_sync",
            apply_mock,
        )

        modal = memory_commands.EditMemoryPageModal(
            preview_view=preview_view,
            page_index=1,
            summary_snapshot=summary,
            current_page_text=pages[1],
        )
        modal.page_input = SimpleNamespace(value=replacement)

        interaction = _make_component_interaction(user_id=1)

        await modal.on_submit(interaction)

        expected_summary = "".join([pages[0], replacement, *pages[2:]])
        apply_mock.assert_awaited_once_with(
            user_id=1,
            old_summary=summary,
            new_summary=expected_summary,
        )
        preview_view.refresh_summary.assert_awaited_once_with(expected_summary)
        assert "成功更新" in interaction.followup.send.await_args.args[0]

    asyncio.run(_run())


def test_edit_memory_page_modal_rejects_stale_summary(
    memory_commands,
    monkeypatch: pytest.MonkeyPatch,
):
    async def _run():
        summary = _build_long_valid_summary(
            memory_commands,
            min_length=memory_commands._MEMORY_PAGE_CHAR_LIMIT * 2 + 500,
        )
        pages = memory_commands._split_memory_summary_into_pages(summary)
        preview_view = memory_commands.PagedMemoryPreviewView(
            actor_user_id=1,
            target_user_id=1,
            summary_snapshot=summary,
        )

        monkeypatch.setattr(
            memory_commands,
            "_get_personal_summary_raw",
            AsyncMock(return_value=summary + "\n- 其他修改"),
        )
        apply_mock = AsyncMock()
        monkeypatch.setattr(
            memory_commands,
            "_apply_personal_summary_update_with_vector_sync",
            apply_mock,
        )

        modal = memory_commands.EditMemoryPageModal(
            preview_view=preview_view,
            page_index=1,
            summary_snapshot=summary,
            current_page_text=pages[1],
        )
        modal.page_input = SimpleNamespace(value=pages[1])

        interaction = _make_component_interaction(user_id=1)

        await modal.on_submit(interaction)

        apply_mock.assert_not_called()
        assert "已被其他操作改动" in interaction.followup.send.await_args.args[0]

    asyncio.run(_run())


def test_edit_memory_page_modal_returns_retry_view_on_validation_failure(
    memory_commands,
    monkeypatch: pytest.MonkeyPatch,
):
    async def _run():
        summary = _build_long_valid_summary(
            memory_commands,
            min_length=memory_commands._MEMORY_PAGE_CHAR_LIMIT * 2 + 500,
        )
        pages = memory_commands._split_memory_summary_into_pages(summary)
        preview_view = memory_commands.PagedMemoryPreviewView(
            actor_user_id=1,
            target_user_id=1,
            summary_snapshot=summary,
        )

        monkeypatch.setattr(
            memory_commands,
            "_get_personal_summary_raw",
            AsyncMock(return_value=summary),
        )
        apply_mock = AsyncMock()
        monkeypatch.setattr(
            memory_commands,
            "_apply_personal_summary_update_with_vector_sync",
            apply_mock,
        )

        modal = memory_commands.EditMemoryPageModal(
            preview_view=preview_view,
            page_index=0,
            summary_snapshot=summary,
            current_page_text=pages[0],
        )
        modal.page_input = SimpleNamespace(value="坏掉的第一页")

        interaction = _make_component_interaction(user_id=1)

        await modal.on_submit(interaction)

        apply_mock.assert_not_called()
        send_kwargs = interaction.followup.send.await_args.kwargs
        assert "会破坏整份记忆格式" in interaction.followup.send.await_args.args[0]
        assert isinstance(send_kwargs["view"], memory_commands.RetryEditMemoryPageView)

    asyncio.run(_run())


def test_paged_memory_preview_navigation_updates_button_state(memory_commands):
    async def _run():
        summary = _build_long_valid_summary(
            memory_commands,
            min_length=memory_commands._MEMORY_PAGE_CHAR_LIMIT * 2 + 500,
        )
        view = memory_commands.PagedMemoryPreviewView(
            actor_user_id=1,
            target_user_id=1,
            summary_snapshot=summary,
        )

        assert view.first_button.disabled is True
        assert view.prev_button.disabled is True

        interaction = _make_component_interaction(user_id=1)
        await view.go_to_next_page(interaction)

        assert view.current_page == 1
        assert view.first_button.disabled is False
        assert view.prev_button.disabled is False

        view.current_page = len(view.pages) - 1
        embed = view.build_embed()

        assert view.next_button.disabled is True
        assert view.last_button.disabled is True
        assert f"第 {len(view.pages)} / {len(view.pages)} 页" in embed.footer.text

    asyncio.run(_run())
