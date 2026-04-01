# -*- coding: utf-8 -*-

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderSelection:
    provider_name: str
    stage: str
    effective_score: float


@dataclass
class _ProviderState:
    provider_name: str
    current_unit_cost: Optional[float] = None
    warmup_total_unit_cost: float = 0.0
    warmup_sample_count: int = 0
    failure_penalty_seconds: float = 0.0
    explored: bool = False
    warmup_reserved: bool = False
    in_flight_count: int = 0
    last_output_units: int = 0
    last_elapsed_seconds: float = 0.0


class KimiGatewayProviderSelectorService:
    PROVIDER_NAMES: List[str] = [
        "moonshotai",
        "fireworks",
        "togetherai",
        "novita",
        "bedrock",
    ]
    PRIMARY_SELECTION_PROBABILITY = 0.7
    FAILURE_PENALTY_SECONDS = 2.0

    def __init__(self, *, rng: Optional[random.Random] = None) -> None:
        self._lock = asyncio.Lock()
        self._rng = rng or random.Random()
        self._warmup_results_logged = False
        self._provider_states: Dict[str, _ProviderState] = {
            provider_name: _ProviderState(provider_name=provider_name)
            for provider_name in self.PROVIDER_NAMES
        }

    async def select_provider(self) -> ProviderSelection:
        async with self._lock:
            warmup_candidates = [
                self._provider_states[provider_name]
                for provider_name in self.PROVIDER_NAMES
                if not self._provider_states[provider_name].explored
                and not self._provider_states[provider_name].warmup_reserved
            ]

            if warmup_candidates:
                selected_state = warmup_candidates[0]
                selected_state.warmup_reserved = True
            else:
                selected_state = self._select_steady_provider_locked()

            selected_state.in_flight_count += 1
            stage = "warmup" if not self._is_warmup_complete_locked() else "steady"
            return ProviderSelection(
                provider_name=selected_state.provider_name,
                stage=stage,
                effective_score=self._get_effective_score_locked(selected_state),
            )

    async def report_success(
        self, provider_name: str, *, elapsed_seconds: float, output_units: int
    ) -> None:
        async with self._lock:
            state = self._provider_states.get(provider_name)
            if state is None:
                return

            was_warmup_complete = self._is_warmup_complete_locked()
            state.explored = True
            state.warmup_reserved = False
            state.in_flight_count = max(0, state.in_flight_count - 1)
            state.failure_penalty_seconds = 0.0
            state.last_elapsed_seconds = max(float(elapsed_seconds), 0.0)
            state.last_output_units = max(int(output_units), 1)
            unit_cost = state.last_elapsed_seconds / state.last_output_units

            if not was_warmup_complete:
                state.warmup_total_unit_cost += unit_cost
                state.warmup_sample_count += 1
                state.current_unit_cost = (
                    state.warmup_total_unit_cost / state.warmup_sample_count
                )
                self._log_warmup_results_if_ready_locked()
                return

            state.current_unit_cost = unit_cost

    async def report_failure(self, provider_name: str) -> None:
        async with self._lock:
            state = self._provider_states.get(provider_name)
            if state is None:
                return

            state.explored = True
            state.warmup_reserved = False
            state.in_flight_count = max(0, state.in_flight_count - 1)
            state.failure_penalty_seconds += self.FAILURE_PENALTY_SECONDS
            if state.current_unit_cost is None:
                state.current_unit_cost = 0.0
            self._log_warmup_results_if_ready_locked()

    async def release_provider(self, provider_name: str) -> None:
        async with self._lock:
            state = self._provider_states.get(provider_name)
            if state is None:
                return

            state.warmup_reserved = False
            state.in_flight_count = max(0, state.in_flight_count - 1)

    def _select_steady_provider_locked(self) -> _ProviderState:
        ordered_states = [
            self._provider_states[provider_name] for provider_name in self.PROVIDER_NAMES
        ]
        ordered_states.sort(
            key=lambda state: (
                self._get_effective_score_locked(state),
                self.PROVIDER_NAMES.index(state.provider_name),
            )
        )

        primary_state = ordered_states[0]
        remaining_states = ordered_states[1:]

        if (
            not remaining_states
            or self._rng.random() < self.PRIMARY_SELECTION_PROBABILITY
        ):
            return primary_state

        return self._rng.choice(remaining_states)

    def _get_effective_score_locked(self, state: _ProviderState) -> float:
        return max(state.current_unit_cost or 0.0, 0.0) + max(
            state.failure_penalty_seconds, 0.0
        )

    def _is_warmup_complete_locked(self) -> bool:
        return all(state.explored for state in self._provider_states.values())

    def _log_warmup_results_if_ready_locked(self) -> None:
        if self._warmup_results_logged or not self._is_warmup_complete_locked():
            return

        self._warmup_results_logged = True
        scoreboard = " | ".join(self._build_scoreboard_entries_locked())
        log.info("[KimiGateway] 供应商预热完成，当前成绩榜：%s", scoreboard)

    def _build_scoreboard_entries_locked(self) -> List[str]:
        ordered_states = [
            self._provider_states[provider_name] for provider_name in self.PROVIDER_NAMES
        ]
        ordered_states.sort(
            key=lambda state: (
                self._get_effective_score_locked(state),
                self.PROVIDER_NAMES.index(state.provider_name),
            )
        )
        return [
            (
                f"{state.provider_name}:effective={self._get_effective_score_locked(state):.6f}"
                f",unit={float(state.current_unit_cost or 0.0):.6f}"
                f",penalty={state.failure_penalty_seconds:.2f}"
                f",samples={state.warmup_sample_count}"
            )
            for state in ordered_states
        ]
