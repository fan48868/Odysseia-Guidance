# -*- coding: utf-8 -*-

import asyncio
import hashlib
import logging
import random
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
    total_calls: int = 0

    @property
    def key_tail(self) -> str:
        return self.api_key[-4:] if self.api_key else "????"

    def is_available(self, now_ts: float) -> bool:
        return now_ts >= self.cooldown_until and now_ts >= self.daily_disabled_until


class KimiKeyRotationService:
    """
    Kimi Key 路由服务（仅官方 Moonshot，纯内存，不持久化）：
    - 为每个用户绑定专属 Key，实现对话连续性。
    - 在新会话开始时，通过负载均衡（绑定用户数最少）分配 Key。
    - 支持故障自动转移和用户会话超时（10分钟）自动释放。
    - 保留 429 两阶段惩罚机制。
    """
    USER_EXPIRATION_SECONDS = 600  # 10 minutes

    def __init__(self, moonshot_base_url: Optional[str], moonshot_api_key: Optional[str]):
        self._lock = asyncio.Lock()
        self._slots: List[KimiKeySlot] = []
        self._slot_map: Dict[str, KimiKeySlot] = {}
        
        # --- 用户绑定与负载均衡 ---
        self._user_bindings: Dict[int, str] = {}  # user_id -> slot_id
        self._binding_last_seen: Dict[int, float] = {}  # user_id -> last_seen_timestamp
        self._binding_user_count: Dict[str, int] = {}  # slot_id -> number of users
        self._cleanup_task: Optional[asyncio.Task] = None

        self._append_slots_if_valid(moonshot_base_url, moonshot_api_key)

        if self._slots:
            # 初始化负载计数
            for slot in self._slots:
                self._binding_user_count[slot.slot_id] = 0
            
            log.info(f"[KimiKeyRotation] 已初始化 {len(self._slots)} 个官方 key 槽位。后台清理任务将在首次使用时启动。")
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

    async def _periodic_cleanup(self):
        """定期清理超时的用户绑定。"""
        while True:
            await asyncio.sleep(60)  # 每分钟检查一次
            async with self._lock:
                now = time.time()
                expired_users = [
                    user_id for user_id, last_seen in self._binding_last_seen.items()
                    if now - last_seen > self.USER_EXPIRATION_SECONDS
                ]
                if not expired_users:
                    continue

                log.info(f"[KimiKeyRotation] 开始清理 {len(expired_users)} 个超时用户绑定...")
                for user_id in expired_users:
                    slot_id = self._user_bindings.pop(user_id, None)
                    self._binding_last_seen.pop(user_id, None)
                    if slot_id and slot_id in self._binding_user_count:
                        self._binding_user_count[slot_id] = max(0, self._binding_user_count.get(slot_id, 1) - 1)
                        log.debug(f"[KimiKeyRotation] 用户 {user_id} 的绑定已超时释放，Key ...{self._slot_map[slot_id].key_tail} 负载-1。")
                log.info(f"[KimiKeyRotation] 清理完成。")

    def _next_beijing_midnight_ts(self) -> float:
        tz = ZoneInfo("Asia/Shanghai")
        now = datetime.now(tz)
        tomorrow = (now + timedelta(days=1)).date()
        next_midnight = datetime.combine(tomorrow, datetime.min.time(), tzinfo=tz)
        return next_midnight.timestamp()

    async def acquire_active_slot(self, user_id: int) -> KimiKeySlot:
        """
        根据用户ID获取一个专属的、可用的API Key槽位。
        - 优先返回已绑定的健康Key。
        - 若绑定Key失效，则执行故障转移，重新分配。
        - 若无绑定，则执行负载均衡，分配一个负载最低的Key。
        """
        async with self._lock:
            # --- 首次使用时启动后台清理任务 ---
            if self._cleanup_task is None:
                self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
                log.info("[KimiKeyRotation] 首次调用，已成功启动后台绑定清理任务。")

            now = time.time()
            self._refresh_expired_states_locked()

            bound_slot_id = self._user_bindings.get(user_id)
            if bound_slot_id:
                slot = self._slot_map.get(bound_slot_id)
                if slot and slot.is_available(now):
                    self._binding_last_seen[user_id] = now
                    log.info(f"[KimiKeyRotation] 用户 {user_id} 使用已绑定的专属 Key ...{slot.key_tail}。")
                    return slot
                else:
                    # 绑定的key失效了，需要重新分配 (故障转移)
                    if slot:
                        log.warning(f"[KimiKeyRotation] 用户 {user_id} 绑定的 Key ...{slot.key_tail} 已失效，执行故障转移。")
                    if bound_slot_id in self._binding_user_count:
                        self._binding_user_count[bound_slot_id] = max(0, self._binding_user_count.get(bound_slot_id, 1) - 1)

            # --- 全局负载均衡逻辑 ---
            available_slots = [s for s in self._slots if s.is_available(now)]
            if not available_slots:
                all_slots_in_temp_cooldown = all(s.cooldown_until > now and s.daily_disabled_until < now for s in self._slots)
                if all_slots_in_temp_cooldown:
                    raise NoAvailableKimiKeyError("目前全部key都在冷却，请等待至少2分钟后再试")

                cooling_slots = sorted([s for s in self._slots if not s.is_available(now)], key=lambda s: max(s.cooldown_until, s.daily_disabled_until))
                if cooling_slots:
                    earliest_recovery_slot = cooling_slots[0]
                    recovery_time = max(earliest_recovery_slot.cooldown_until, earliest_recovery_slot.daily_disabled_until)
                    remaining_seconds = recovery_time - now
                    log.error(f"所有Kimi Key当前不可用。最早恢复的Key ...{earliest_recovery_slot.key_tail} 预计在 {remaining_seconds:.1f} 秒后可用。")
                else:
                    log.error("所有Kimi Key当前不可用，且没有冷却中的Key。")
                raise NoAvailableKimiKeyError("所有 Kimi Key 当前不可用。")

            # --- 新的分配逻辑 ---
            # 1. 按总调用次数和当前负载排序
            available_slots.sort(key=lambda s: (self._binding_user_count.get(s.slot_id, 0), s.total_calls))
            
            # 2. 找出负载最低且调用次数最少的一组
            if not available_slots:
                 raise NoAvailableKimiKeyError("代码逻辑错误：在有可用key的情况下找不到最佳key。") # Should not happen
            
            min_load = self._binding_user_count.get(available_slots[0].slot_id, 0)
            min_calls = available_slots[0].total_calls
            
            best_group = [
                s for s in available_slots 
                if self._binding_user_count.get(s.slot_id, 0) == min_load and s.total_calls == min_calls
            ]
            
            # 3. 从最佳组中随机选择一个
            best_slot = random.choice(best_group)

            # --- 绑定用户并更新计数 ---
            best_slot.total_calls += 1
            self._user_bindings[user_id] = best_slot.slot_id
            self._binding_last_seen[user_id] = now
            self._binding_user_count[best_slot.slot_id] = self._binding_user_count.get(best_slot.slot_id, 0) + 1

            log.info(f"[KimiKeyRotation] 为用户 {user_id} 分配新的专属 Key ...{best_slot.key_tail} (当前负载: {self._binding_user_count[best_slot.slot_id]}, 总调用: {best_slot.total_calls})。")
            return best_slot

    async def report_success(self, slot_id: str) -> None:
        """报告指定Key调用成功，重置其429观察状态。"""
        async with self._lock:
            slot = self._slot_map.get(slot_id)
            if not slot:
                return

            if slot.had_429_once:
                slot.had_429_once = False
                log.debug(f"[KimiKeyRotation] Key ...{slot.key_tail} 调用成功，已重置429观察状态。")

    async def report_429(self, slot_id: str) -> Tuple[str, float]:
        """
        报告 Key 触发了 429（非TPD），执行临时冷却。
        返回: ("temporary_cooldown", 冷却结束时间戳)
        """
        async with self._lock:
            slot = self._slot_map.get(slot_id)
            if not slot:
                return "temporary_cooldown", 0.0

            now_ts = time.time()
            slot.cooldown_until = now_ts + KIMI_TEMP_COOLDOWN_SECONDS
            
            log.warning(
                f"[KimiKeyRotation] Key ...{slot.key_tail} 触发 429 (无 TPD)，冷却 {KIMI_TEMP_COOLDOWN_SECONDS} 秒。"
            )
            return "temporary_cooldown", slot.cooldown_until

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
        """重置全部 Kimi key 的惩罚状态和用户绑定。"""
        async with self._lock:
            for slot in self._slots:
                slot.had_429_once = False
                slot.cooldown_until = 0.0
                slot.daily_disabled_until = 0.0
            
            self._user_bindings.clear()
            self._binding_last_seen.clear()
            for slot in self._slots:
                slot.total_calls = 0
            for slot_id in self._binding_user_count:
                self._binding_user_count[slot_id] = 0

            log.warning("[KimiKeyRotation] 已手动重置全部 key 惩罚状态、调用计数和用户绑定。")
