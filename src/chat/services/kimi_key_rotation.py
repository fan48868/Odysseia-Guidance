# -*- coding: utf-8 -*-

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

KIMI_TEMP_COOLDOWN_SECONDS = 120


class NoAvailableKimiKeyError(Exception):
    """当 Kimi Key 池中没有可用 Key 时抛出。"""

    pass


@dataclass
class KimiKeySlot:
    slot_id: str
    base_url: str
    api_key: str
    had_429_once: bool = False
    cooldown_until: float = 0.0
    daily_disabled_until: float = 0.0

    @property
    def key_tail(self) -> str:
        return self.api_key[-4:] if self.api_key else "????"

    def is_available(self, now_ts: float) -> bool:
        return now_ts >= self.cooldown_until and now_ts >= self.daily_disabled_until


class KimiKeyRotationService:
    """
    Kimi Key 路由服务（仅官方 Moonshot，纯内存，不持久化）：
    - 官方 key 按可用轮询；
    - 官方 key 保留 429 两阶段机制：
        1) 首次 429：冷却 120 秒；
        2) 2 分钟后再次 429：封禁至次日 00:00（Asia/Shanghai）；
    - 重启进程后状态重置，全部 key 会重新尝试。
    """

    def __init__(self, moonshot_base_url: Optional[str], moonshot_api_key: Optional[str]):
        self._lock = asyncio.Lock()
        self._slots: List[KimiKeySlot] = []
        self._slot_map: Dict[str, KimiKeySlot] = {}
        self._active_slot_id: Optional[str] = None
        self._rr_index: int = -1

        self._append_slots_if_valid(moonshot_base_url, moonshot_api_key)

        if self._slots:
            self._active_slot_id = self._slots[0].slot_id
            log.info(f"[KimiKeyRotation] 已初始化 {len(self._slots)} 个官方 key 槽位。")
        else:
            log.warning("[KimiKeyRotation] 未检测到任何可用的官方 Kimi Key 配置。")

    @property
    def has_configured_keys(self) -> bool:
        return bool(self._slots)

    @staticmethod
    def _split_api_keys(raw_api_keys: Optional[str]) -> List[str]:
        """
        支持以下格式：
        - 单个 key: "sk-xxx"
        - 逗号分隔: "sk-1,sk-2"
        - 中文逗号分隔: "sk-1，sk-2"
        - 多行分隔
        """
        raw = (raw_api_keys or "").strip()
        if not raw:
            return []

        parts = re.split(r"[,\n\r，]+", raw)
        normalized: List[str] = []
        seen = set()

        for part in parts:
            key = part.strip().strip('"').strip("'").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            normalized.append(key)

        return normalized

    def _append_slots_if_valid(self, base_url: Optional[str], api_keys: Optional[str]) -> None:
        url = (base_url or "").strip().rstrip("/")
        keys = self._split_api_keys(api_keys)
        if not (url and keys):
            return

        appended_count = 0
        for key in keys:
            slot_id = hashlib.sha256(f"moonshot|{url}|{key}".encode("utf-8")).hexdigest()[:16]
            if slot_id in self._slot_map:
                continue

            slot = KimiKeySlot(
                slot_id=slot_id,
                base_url=url,
                api_key=key,
            )
            self._slots.append(slot)
            self._slot_map[slot_id] = slot
            appended_count += 1

        if appended_count > 1:
            log.info(f"[KimiKeyRotation] 已加载 {appended_count} 个官方 key。")

    def _refresh_expired_states_locked(self) -> None:
        now_ts = time.time()
        for slot in self._slots:
            if slot.cooldown_until > 0 and now_ts >= slot.cooldown_until:
                slot.cooldown_until = 0.0

            if slot.daily_disabled_until > 0 and now_ts >= slot.daily_disabled_until:
                slot.daily_disabled_until = 0.0
                slot.had_429_once = False

    def _find_slot_index(self, slot_id: Optional[str]) -> int:
        if not slot_id:
            return -1
        for idx, slot in enumerate(self._slots):
            if slot.slot_id == slot_id:
                return idx
        return -1

    def _pick_next_available_locked(self, slot_id: Optional[str]) -> Optional[KimiKeySlot]:
        if not self._slots:
            return None

        now_ts = time.time()
        start_idx = self._find_slot_index(slot_id)
        if start_idx < 0:
            start_idx = -1

        for step in range(1, len(self._slots) + 1):
            idx = (start_idx + step) % len(self._slots)
            candidate = self._slots[idx]
            if candidate.is_available(now_ts):
                self._rr_index = idx
                return candidate

        return None

    def _pick_active_or_next_locked(self) -> Optional[KimiKeySlot]:
        if not self._slots:
            return None

        now_ts = time.time()
        active = self._slot_map.get(self._active_slot_id) if self._active_slot_id else None
        if active and active.is_available(now_ts):
            return active

        next_slot = self._pick_next_available_locked(self._active_slot_id)
        if next_slot:
            self._active_slot_id = next_slot.slot_id
            return next_slot

        return None

    def _next_beijing_midnight_ts(self) -> float:
        tz = ZoneInfo("Asia/Shanghai")
        now = datetime.now(tz)
        tomorrow = (now + timedelta(days=1)).date()
        next_midnight = datetime.combine(tomorrow, datetime.min.time(), tzinfo=tz)
        return next_midnight.timestamp()

    async def acquire_active_slot(self) -> KimiKeySlot:
        async with self._lock:
            if not self._slots:
                raise NoAvailableKimiKeyError("Kimi Key 未配置。")

            self._refresh_expired_states_locked()
            # 正常请求也轮询：每次都从上一个 active 的下一个开始找可用 key
            slot = self._pick_next_available_locked(self._active_slot_id)
            if not slot:
                raise NoAvailableKimiKeyError("所有 Kimi Key 当前不可用。")

            self._active_slot_id = slot.slot_id
            return slot

    async def report_success(self, slot_id: str) -> None:
        async with self._lock:
            slot = self._slot_map.get(slot_id)
            if not slot:
                return

            if slot.had_429_once:
                slot.had_429_once = False

            if self._active_slot_id != slot_id:
                self._active_slot_id = slot_id

    async def report_429(self, slot_id: str) -> Tuple[str, float]:
        """
        官方 key 的 429 双阶段惩罚：
        返回:
            (stage, until_ts)
            stage:
              - "temporary_cooldown"
              - "daily_disabled"
        """
        async with self._lock:
            slot = self._slot_map.get(slot_id)
            if not slot:
                return "temporary_cooldown", 0.0

            now_ts = time.time()
            if not slot.had_429_once:
                slot.had_429_once = True
                slot.cooldown_until = now_ts + KIMI_TEMP_COOLDOWN_SECONDS
                stage = "temporary_cooldown"
                until_ts = slot.cooldown_until
                log.warning(
                    f"[KimiKeyRotation] Key ...{slot.key_tail} 首次触发 429，冷却 {KIMI_TEMP_COOLDOWN_SECONDS} 秒。"
                )
            else:
                slot.had_429_once = False
                slot.cooldown_until = 0.0
                slot.daily_disabled_until = self._next_beijing_midnight_ts()
                stage = "daily_disabled"
                until_ts = slot.daily_disabled_until
                reset_dt = datetime.fromtimestamp(until_ts, ZoneInfo("Asia/Shanghai"))
                log.error(
                    f"[KimiKeyRotation] Key ...{slot.key_tail} 再次触发 429，已封禁至次日重置: "
                    f"{reset_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}。"
                )

            next_slot = self._pick_next_available_locked(slot_id)
            self._active_slot_id = next_slot.slot_id if next_slot else None

            return stage, until_ts

    async def report_tpd(self, slot_id: str) -> float:
        """
        命中 TPD 时直接封禁到次日重置。
        返回：until_ts
        """
        async with self._lock:
            slot = self._slot_map.get(slot_id)
            if not slot:
                return 0.0

            slot.had_429_once = False
            slot.cooldown_until = 0.0
            slot.daily_disabled_until = self._next_beijing_midnight_ts()
            until_ts = slot.daily_disabled_until

            next_slot = self._pick_next_available_locked(slot_id)
            self._active_slot_id = next_slot.slot_id if next_slot else None
            return until_ts

    async def get_pool_stats(self) -> Tuple[int, int]:
        """返回当前 key 池统计：(available_count, total_count)。"""
        async with self._lock:
            self._refresh_expired_states_locked()
            total = len(self._slots)
            now_ts = time.time()
            available = sum(1 for slot in self._slots if slot.is_available(now_ts))
            return available, total

    async def reset_all_penalties(self) -> None:
        """重置全部 Kimi key 的惩罚状态（冷却、次日封禁、429标记）。"""
        async with self._lock:
            for slot in self._slots:
                slot.had_429_once = False
                slot.cooldown_until = 0.0
                slot.daily_disabled_until = 0.0

            self._rr_index = -1
            self._active_slot_id = self._slots[0].slot_id if self._slots else None
            log.warning("[KimiKeyRotation] 已手动重置全部 key 惩罚状态。")
