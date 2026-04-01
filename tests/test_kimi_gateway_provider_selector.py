import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

import httpx
import pytest

from src.chat.services.kimi_gateway_provider_selector import (
    KimiGatewayProviderSelectorService,
)
from src.chat.services.openai_models.custom_model import CustomModelClient


class _DeterministicRandom:
    def __init__(
        self,
        *,
        random_values: Optional[List[float]] = None,
        choice_indexes: Optional[List[int]] = None,
    ) -> None:
        self._random_values = list(random_values or [])
        self._choice_indexes = list(choice_indexes or [])

    def random(self) -> float:
        if self._random_values:
            return self._random_values.pop(0)
        return 0.0

    def choice(self, items: List[Any]) -> Any:
        if not items:
            raise IndexError("Cannot choose from an empty sequence.")
        if self._choice_indexes:
            index = self._choice_indexes.pop(0)
        else:
            index = 0
        return items[index % len(items)]


class _MutableClock:
    def __init__(self, initial_value: float = 0.0) -> None:
        self.current = initial_value

    def __call__(self) -> float:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += seconds


def _build_runtime_config(
    *,
    base_url: str = "https://ai-gateway.vercel.sh/v1",
    model_name: str = "moonshotai/kimi-k2.5",
    gateway_provider_name: str = "",
    gateway_provider_timeout_ms: int = 4000,
) -> Dict[str, Any]:
    return {
        "base_url": base_url,
        "api_key": "test-key",
        "api_keys": ["test-key"],
        "api_key_source_type": "inline",
        "api_key_source_value": "test-key",
        "api_key_file_path": None,
        "api_key_error": None,
        "model_name": model_name,
        "enable_vision": False,
        "enable_video_input": False,
        "gateway_provider_timeout_ms": gateway_provider_timeout_ms,
        "gateway_provider_name": gateway_provider_name,
        "stream_idle_timeout_seconds": 5.0,
    }


def _build_sse_response(
    request: httpx.Request,
    *,
    model_name: str = "moonshotai/kimi-k2.5",
    content: str = "hello",
    completion_tokens: Optional[int] = 7,
) -> httpx.Response:
    chunks: List[Dict[str, Any]] = [
        {
            "id": "resp-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": None,
                }
            ],
        }
    ]
    final_chunk: Dict[str, Any] = {
        "id": "resp-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    if completion_tokens is not None:
        final_chunk["usage"] = {
            "prompt_tokens": 3,
            "completion_tokens": completion_tokens,
            "total_tokens": completion_tokens + 3,
        }
    chunks.append(final_chunk)

    body = "\n".join(
        [f"data: {json.dumps(chunk, ensure_ascii=False)}" for chunk in chunks]
        + ["data: [DONE]"]
    )
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text=body,
        request=request,
    )


def _build_json_response(
    request: httpx.Request,
    *,
    model_name: str,
    content: str = "ok",
) -> httpx.Response:
    payload = {
        "id": "resp-json-1",
        "object": "chat.completion",
        "created": 1,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }
    return httpx.Response(
        200,
        headers={"content-type": "application/json"},
        text=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        request=request,
    )


def _capture_request_payload(request: httpx.Request, captured_payloads: List[Dict[str, Any]]):
    captured_payloads.append(json.loads(request.content.decode("utf-8")))


@pytest.mark.asyncio
async def test_selector_warmup_claims_each_provider_once_under_concurrency():
    service = KimiGatewayProviderSelectorService()

    selections = await asyncio.gather(
        *(service.select_provider() for _ in service.PROVIDER_NAMES)
    )

    assert [selection.provider_name for selection in selections] == service.PROVIDER_NAMES
    assert all(selection.stage == "warmup" for selection in selections)


@pytest.mark.asyncio
async def test_selector_logs_scoreboard_when_warmup_completes(caplog: pytest.LogCaptureFixture):
    service = KimiGatewayProviderSelectorService()

    with caplog.at_level(
        logging.INFO, logger="src.chat.services.kimi_gateway_provider_selector"
    ):
        for index, provider_name in enumerate(service.PROVIDER_NAMES, start=1):
            await service.report_success(
                provider_name,
                elapsed_seconds=float(index),
                output_units=1,
            )

    assert "供应商预热完成" in caplog.text
    for provider_name in service.PROVIDER_NAMES:
        assert provider_name in caplog.text


@pytest.mark.asyncio
async def test_selector_uses_warmup_average_before_switching_to_latest_samples():
    service = KimiGatewayProviderSelectorService()

    await service.report_success("moonshotai", elapsed_seconds=0.2, output_units=2)
    await service.report_success("moonshotai", elapsed_seconds=0.3, output_units=3)

    moonshot_state = service._provider_states["moonshotai"]
    assert moonshot_state.current_unit_cost == pytest.approx(0.1)

    for provider_name in service.PROVIDER_NAMES:
        if provider_name == "moonshotai":
            continue
        await service.report_failure(provider_name)

    await service.report_success("moonshotai", elapsed_seconds=1.0, output_units=5)

    assert moonshot_state.current_unit_cost == pytest.approx(0.2)
    assert moonshot_state.failure_penalty_seconds == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_selector_prefers_fastest_provider_but_can_sample_others_and_penalize_failures():
    rng = _DeterministicRandom(random_values=[0.2, 0.95, 0.1], choice_indexes=[2])
    service = KimiGatewayProviderSelectorService(rng=rng)
    costs = {
        "novita": 0.03,
        "bedrock": 0.035,
        "moonshotai": 0.05,
        "fireworks": 0.025,
        "togetherai": 0.04,
    }

    for provider_name in service.PROVIDER_NAMES:
        await service.report_success(
            provider_name,
            elapsed_seconds=costs[provider_name] * 10,
            output_units=10,
        )

    primary_selection = await service.select_provider()
    assert primary_selection.provider_name == "fireworks"
    await service.release_provider(primary_selection.provider_name)

    exploratory_selection = await service.select_provider()
    assert exploratory_selection.provider_name == "togetherai"
    await service.release_provider(exploratory_selection.provider_name)

    await service.report_failure("fireworks")

    post_penalty_selection = await service.select_provider()
    assert post_penalty_selection.provider_name == "novita"


@pytest.mark.asyncio
async def test_selector_cools_down_slow_provider_and_reenables_after_20_minutes(
    caplog: pytest.LogCaptureFixture,
):
    clock = _MutableClock()
    service = KimiGatewayProviderSelectorService(
        time_fn=clock,
        rng=_DeterministicRandom(random_values=[0.0, 0.0]),
    )

    with caplog.at_level(
        logging.INFO, logger="src.chat.services.kimi_gateway_provider_selector"
    ):
        await service.report_success("novita", elapsed_seconds=0.21, output_units=1)
        for provider_name in service.PROVIDER_NAMES[1:]:
            await service.report_success(
                provider_name,
                elapsed_seconds=0.19,
                output_units=1,
            )

        for provider_name in service.PROVIDER_NAMES[1:]:
            await service.report_failure(provider_name)

        first_selection = await service.select_provider()
        await service.release_provider(first_selection.provider_name)

        clock.advance(service.SLOW_PROVIDER_COOLDOWN_SECONDS + 1)

        second_selection = await service.select_provider()
        await service.release_provider(second_selection.provider_name)

    assert first_selection.provider_name == "bedrock"
    assert second_selection.provider_name == "novita"
    assert "供应商测速过慢，进入冷却" in caplog.text
    assert "供应商冷却结束，重新参与选路" in caplog.text
    assert (
        service._provider_states["novita"].cooldown_until_monotonic
        == pytest.approx(0.0)
    )


def test_extract_output_units_uses_output_text_length_and_minimum_one():
    assert (
        CustomModelClient._extract_output_units_from_result(
            {
                "usage": {"completion_tokens": 11},
                "choices": [{"message": {"content": "ignored"}}],
            }
        )
        == len("ignored")
    )
    assert (
        CustomModelClient._extract_output_units_from_result(
            {
                "choices": [
                    {"message": {"content": [{"type": "text", "text": "abcd"}]}}
                ]
            }
        )
        == 4
    )
    assert (
        CustomModelClient._extract_output_units_from_result(
            {"choices": [{"message": {"content": ""}}]}
        )
        == 1
    )
    assert (
        CustomModelClient._extract_output_units_from_result(
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {"name": "test", "arguments": "{}"},
                                }
                            ],
                        }
                    }
                ]
            }
        )
        == 1
    )


@pytest.mark.asyncio
async def test_custom_model_send_injects_single_provider_options_for_kimi_gateway(
    caplog: pytest.LogCaptureFixture,
):
    client = CustomModelClient()
    client._kimi_gateway_provider_selector = KimiGatewayProviderSelectorService(
        rng=_DeterministicRandom(random_values=[0.0])
    )
    captured_payloads: List[Dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        _capture_request_payload(request, captured_payloads)
        return _build_sse_response(request, completion_tokens=7)

    runtime_config = _build_runtime_config()
    with caplog.at_level(
        logging.INFO, logger="src.chat.services.openai_models.custom_model"
    ):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            result = await client.send(
                http_client=http_client,
                payload={"model": "moonshotai/kimi-k2.5", "messages": []},
                runtime_config=runtime_config,
            )

    gateway_payload = captured_payloads[0]["providerOptions"]["gateway"]
    assert gateway_payload["only"] == ["novita"]
    assert gateway_payload["order"] == ["novita"]
    assert gateway_payload["providerTimeouts"]["byok"] == {"novita": 4000}
    assert result["selected_gateway_provider"] == "novita"
    assert result["selected_gateway_provider_stage"] == "warmup"

    novita_state = client._kimi_gateway_provider_selector._provider_states["novita"]
    assert novita_state.last_output_units == len("hello")
    assert novita_state.current_unit_cost is not None
    assert "已选择 Kimi Gateway 供应商" in caplog.text
    assert "Kimi Gateway 供应商测速结果" in caplog.text
    assert "供应商=novita" in caplog.text
    assert "输出字数=5" in caplog.text


@pytest.mark.asyncio
async def test_custom_model_send_keeps_non_kimi_gateway_behavior():
    client = CustomModelClient()
    captured_payloads: List[Dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        _capture_request_payload(request, captured_payloads)
        return _build_json_response(request, model_name="openai/gpt-4.1")

    runtime_config = _build_runtime_config(
        model_name="openai/gpt-4.1",
        gateway_provider_name="openai",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        result = await client.send(
            http_client=http_client,
            payload={"model": "openai/gpt-4.1", "messages": []},
            runtime_config=runtime_config,
        )

    gateway_payload = captured_payloads[0]["providerOptions"]["gateway"]
    assert "only" not in gateway_payload
    assert "order" not in gateway_payload
    assert gateway_payload["providerTimeouts"]["byok"] == {"openai": 4000}
    assert result["selected_gateway_provider"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [429, 503])
async def test_custom_model_send_penalizes_retryable_status_and_switches_provider_on_next_call(
    status_code: int,
):
    client = CustomModelClient()
    captured_payloads: List[Dict[str, Any]] = []

    def failing_handler(request: httpx.Request) -> httpx.Response:
        _capture_request_payload(request, captured_payloads)
        return httpx.Response(
            status_code,
            headers={"content-type": "application/json"},
            text='{"error":"retryable"}',
            request=request,
        )

    runtime_config = _build_runtime_config()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(failing_handler)
    ) as failing_client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.send(
                http_client=failing_client,
                payload={"model": "moonshotai/kimi-k2.5", "messages": []},
                runtime_config=runtime_config,
            )

    assert captured_payloads[0]["providerOptions"]["gateway"]["only"] == ["novita"]
    assert (
        client._kimi_gateway_provider_selector._provider_states[
            "novita"
        ].failure_penalty_seconds
        == pytest.approx(2.0)
    )

    next_payloads: List[Dict[str, Any]] = []

    def success_handler(request: httpx.Request) -> httpx.Response:
        _capture_request_payload(request, next_payloads)
        return _build_sse_response(request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(success_handler)
    ) as success_client:
        result = await client.send(
            http_client=success_client,
            payload={"model": "moonshotai/kimi-k2.5", "messages": []},
            runtime_config=runtime_config,
        )

    assert next_payloads[0]["providerOptions"]["gateway"]["only"] == ["bedrock"]
    assert result["selected_gateway_provider"] == "bedrock"


@pytest.mark.asyncio
async def test_custom_model_send_releases_provider_on_non_retryable_4xx():
    client = CustomModelClient()
    runtime_config = _build_runtime_config()

    def bad_request_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            headers={"content-type": "application/json"},
            text='{"error":"bad request"}',
            request=request,
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(bad_request_handler)
    ) as bad_client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.send(
                http_client=bad_client,
                payload={"model": "moonshotai/kimi-k2.5", "messages": []},
                runtime_config=runtime_config,
            )

    novita_state = client._kimi_gateway_provider_selector._provider_states["novita"]
    assert novita_state.explored is False
    assert novita_state.failure_penalty_seconds == pytest.approx(0.0)

    captured_payloads: List[Dict[str, Any]] = []

    def success_handler(request: httpx.Request) -> httpx.Response:
        _capture_request_payload(request, captured_payloads)
        return _build_sse_response(request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(success_handler)
    ) as success_client:
        result = await client.send(
            http_client=success_client,
            payload={"model": "moonshotai/kimi-k2.5", "messages": []},
            runtime_config=runtime_config,
        )

    assert captured_payloads[0]["providerOptions"]["gateway"]["only"] == ["novita"]
    assert result["selected_gateway_provider"] == "novita"


@pytest.mark.asyncio
async def test_custom_model_send_switches_provider_after_request_error():
    client = CustomModelClient()
    runtime_config = _build_runtime_config()

    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated timeout", request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(timeout_handler)
    ) as timeout_client:
        with pytest.raises(httpx.ReadTimeout):
            await client.send(
                http_client=timeout_client,
                payload={"model": "moonshotai/kimi-k2.5", "messages": []},
                runtime_config=runtime_config,
            )

    captured_payloads: List[Dict[str, Any]] = []

    def success_handler(request: httpx.Request) -> httpx.Response:
        _capture_request_payload(request, captured_payloads)
        return _build_sse_response(request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(success_handler)
    ) as success_client:
        result = await client.send(
            http_client=success_client,
            payload={"model": "moonshotai/kimi-k2.5", "messages": []},
            runtime_config=runtime_config,
        )

    assert captured_payloads[0]["providerOptions"]["gateway"]["only"] == ["bedrock"]
    assert result["selected_gateway_provider"] == "bedrock"


@pytest.mark.asyncio
async def test_custom_model_send_cools_down_slow_provider_after_slow_response(
    caplog: pytest.LogCaptureFixture,
):
    client = CustomModelClient()
    runtime_config = _build_runtime_config()
    captured_payloads: List[Dict[str, Any]] = []
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        _capture_request_payload(request, captured_payloads)
        if request_count == 1:
            time.sleep(0.35)
            return _build_sse_response(request, content="a")
        return _build_sse_response(request, content="hello")

    with caplog.at_level(logging.INFO):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            first_result = await client.send(
                http_client=http_client,
                payload={"model": "moonshotai/kimi-k2.5", "messages": []},
                runtime_config=runtime_config,
            )
            second_result = await client.send(
                http_client=http_client,
                payload={"model": "moonshotai/kimi-k2.5", "messages": []},
                runtime_config=runtime_config,
            )

    assert first_result["selected_gateway_provider"] == "novita"
    assert second_result["selected_gateway_provider"] == "bedrock"
    assert captured_payloads[0]["providerOptions"]["gateway"]["only"] == ["novita"]
    assert captured_payloads[1]["providerOptions"]["gateway"]["only"] == ["bedrock"]
    assert "供应商测速过慢，进入冷却" in caplog.text
