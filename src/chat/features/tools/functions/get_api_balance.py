import asyncio
import logging
import os
import re
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Any, Dict, List, Optional

import httpx

from src.chat.features.chat_settings.services.chat_settings_service import (
    chat_settings_service,
)
from src.chat.features.tools.tool_metadata import tool_metadata
from src.chat.utils.custom_model_api_keys import (
    get_custom_model_api_key_raw_value,
    resolve_custom_model_api_keys,
)

log = logging.getLogger(__name__)

DEEPSEEK_BALANCE_ENDPOINT = "https://api.deepseek.com/user/balance"
MOONSHOT_BALANCE_ENDPOINT = "https://api.moonshot.cn/v1/users/me/balance"
CUSTOM_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"
CUSTOM_GATEWAY_CREDITS_ENDPOINT = f"{CUSTOM_GATEWAY_BASE_URL}/credits"
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
SILICONFLOW_USER_INFO_ENDPOINT = f"{SILICONFLOW_BASE_URL}/user/info"


# ===== 通用辅助函数 =====


def _build_payload(
    ok: bool,
    provider: str,
    model: Optional[str],
    **extra: Any,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "ok": ok,
        "provider": provider,
        "model": model,
    }
    payload.update(extra)
    return payload


def _split_api_keys(raw_api_keys: str) -> List[str]:
    """
    支持以下格式：
    - 单 key: "sk-xxx"
    - 逗号分隔: "k1,k2"
    - 中文逗号分隔: "k1，k2"
    - 多行分隔
    """
    raw = (raw_api_keys or "").strip()
    if not raw:
        return []

    parts = re.split(r"[,\n\r，]+", raw)
    result: List[str] = []
    seen = set()

    for part in parts:
        key = part.strip().strip('"').strip("'").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(key)

    return result


def _key_tail(api_key: str) -> str:
    return api_key[-4:] if api_key else "????"


def _parse_json_response(response: httpx.Response) -> Any:
    try:
        return response.json()
    except Exception:
        return {"raw_text": (response.text or "")[:2000]}


def _truncate_decimal(value: Any, digits: int = 2) -> Any:
    """
    将数值/数值字符串截断到指定位数小数，不做四舍五入。
    """
    try:
        num = Decimal(str(value))
        quant = Decimal("1").scaleb(-digits)
        truncated = num.quantize(quant, rounding=ROUND_DOWN)
        return str(truncated)
    except (InvalidOperation, ValueError, TypeError):
        return value


def _parse_decimal(value: Any) -> Optional[Decimal]:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _truncate_custom_credits(data: Any) -> Any:
    """
    Custom(AI Gateway) 的 /credits 返回会透传给模型，这里直接截断原始 JSON。
    """
    if not isinstance(data, dict):
        return data

    if "balance" in data:
        data["balance"] = _truncate_decimal(data.get("balance"), digits=2)
    if "total_used" in data:
        data["total_used"] = _truncate_decimal(data.get("total_used"), digits=2)
    return data


def _sum_custom_gateway_balances(data_items: List[Any]) -> Optional[str]:
    total = Decimal("0")
    has_balance = False

    for data in data_items:
        if not isinstance(data, dict):
            continue

        balance = _parse_decimal(data.get("balance"))
        if balance is None:
            continue

        total += balance
        has_balance = True

    if not has_balance:
        return None

    return _truncate_decimal(total, digits=2)


def _get_custom_api_key() -> str:
    return get_custom_model_api_key_raw_value()


async def _get_current_model() -> str:
    return await chat_settings_service.get_current_ai_model()


# ===== DeepSeek 余额查询 =====


async def _get_deepseek_balance(
    current_model: str, log_detailed: bool = False
) -> Dict[str, Any]:
    provider = "deepseek"
    endpoint = DEEPSEEK_BALANCE_ENDPOINT
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        return _build_payload(
            ok=False,
            provider=provider,
            model=current_model,
            error="缺少环境变量 DEEPSEEK_API_KEY，无法查询余额。",
        )

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    if log_detailed:
        log.info(
            "--- [工具执行]: get_api_balance, model=%s, provider=%s, endpoint=%s ---",
            current_model,
            provider,
            endpoint,
        )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(endpoint, headers=headers)

        if response.status_code >= 400:
            body_preview = (response.text or "")[:800]
            log.warning(
                "get_api_balance 请求失败: provider=%s, status=%s, body=%s",
                provider,
                response.status_code,
                body_preview,
            )
            return _build_payload(
                ok=False,
                provider=provider,
                model=current_model,
                error=f"余额接口请求失败，HTTP {response.status_code}",
                response_preview=body_preview,
            )

        data = _parse_json_response(response)

        # DeepSeek 成功返回：返回当前 DeepSeek 账户的原始余额 JSON。
        return _build_payload(
            ok=True,
            provider=provider,
            model=current_model,
            data=data,
        )

    except httpx.TimeoutException:
        log.error("get_api_balance 请求超时: provider=%s, endpoint=%s", provider, endpoint)
        return _build_payload(
            ok=False,
            provider=provider,
            model=current_model,
            error="余额接口请求超时，请稍后重试。",
        )
    except Exception as exc:
        log.error("get_api_balance 调用异常。", exc_info=True)
        return _build_payload(
            ok=False,
            provider=provider,
            model=current_model,
            error=f"调用余额接口异常: {str(exc)}",
        )


# ===== Moonshot 余额查询 =====


async def _query_moonshot_balance_with_key(
    client: httpx.AsyncClient,
    endpoint: str,
    api_key: str,
    key_index: int,
) -> Dict[str, Any]:
    provider = "moonshot"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        response = await client.get(endpoint, headers=headers)

        if response.status_code >= 400:
            body_preview = (response.text or "")[:800]
            return {
                "ok": False,
                "provider": provider,
                "key_index": key_index,
                "key_tail": _key_tail(api_key),
                "status_code": response.status_code,
                "error": f"余额接口请求失败，HTTP {response.status_code}",
                "response_preview": body_preview,
            }

        data = _parse_json_response(response)
        return {
            "ok": True,
            "provider": provider,
            "key_index": key_index,
            "key_tail": _key_tail(api_key),
            "status_code": response.status_code,
            "data": data,
        }

    except httpx.TimeoutException:
        return {
            "ok": False,
            "provider": provider,
            "key_index": key_index,
            "key_tail": _key_tail(api_key),
            "error": "余额接口请求超时，请稍后重试。",
        }
    except Exception as exc:
        return {
            "ok": False,
            "provider": provider,
            "key_index": key_index,
            "key_tail": _key_tail(api_key),
            "error": f"调用余额接口异常: {str(exc)}",
        }


async def _get_moonshot_balance(
    current_model: str, log_detailed: bool = False
) -> Dict[str, Any]:
    provider = "moonshot"
    endpoint = MOONSHOT_BALANCE_ENDPOINT
    api_keys = _split_api_keys(os.getenv("MOONSHOT_API_KEY", ""))

    if not api_keys:
        return _build_payload(
            ok=False,
            provider=provider,
            model=current_model,
            error="缺少环境变量 MOONSHOT_API_KEY，无法查询余额。",
        )

    if log_detailed:
        log.info(
            "--- [工具执行]: get_api_balance, model=%s, provider=%s, endpoint=%s, key_count=%s ---",
            current_model,
            provider,
            endpoint,
            len(api_keys),
        )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = [
                _query_moonshot_balance_with_key(
                    client=client,
                    endpoint=endpoint,
                    api_key=api_key,
                    key_index=idx + 1,
                )
                for idx, api_key in enumerate(api_keys)
            ]
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_results):
            if isinstance(item, Exception):
                results.append(
                    {
                        "ok": False,
                        "provider": provider,
                        "key_index": idx + 1,
                        "key_tail": _key_tail(api_keys[idx]),
                        "error": f"并发请求异常: {type(item).__name__}: {str(item)}",
                    }
                )
            else:
                results.append(item)

        success_count = sum(1 for result in results if result.get("ok"))
        failed_count = len(results) - success_count

        payload = _build_payload(
            ok=success_count > 0,
            provider=provider,
            model=current_model,
            total_keys=len(api_keys),
            success_count=success_count,
            failed_count=failed_count,
            results=results,
        )

        if success_count == 0:
            payload["error"] = "所有 MOONSHOT_API_KEY 查询均失败。"

        # Moonshot 成功返回：返回所有 key 的查询结果汇总，而不是单个统一余额。
        return payload

    except Exception as exc:
        log.error("get_api_balance(Moonshot 多 key) 调用异常。", exc_info=True)
        return _build_payload(
            ok=False,
            provider=provider,
            model=current_model,
            error=f"调用余额接口异常: {str(exc)}",
        )


# ===== Custom(AI Gateway) 余额查询 =====


async def _get_custom_gateway_balance(
    current_model: str,
    custom_api_key: str,
    log_detailed: bool = False,
) -> Dict[str, Any]:
    provider = "custom"
    endpoint = CUSTOM_GATEWAY_CREDITS_ENDPOINT
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {custom_api_key}",
    }

    if log_detailed:
        log.info(
            "--- [工具执行]: get_api_balance, model=%s, provider=%s, endpoint=%s ---",
            current_model,
            provider,
            endpoint,
        )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(endpoint, headers=headers)

        if response.status_code >= 400:
            body_preview = (response.text or "")[:800]
            log.warning(
                "get_api_balance 请求失败: provider=%s, status=%s, body=%s",
                provider,
                response.status_code,
                body_preview,
            )
            return _build_payload(
                ok=False,
                provider=provider,
                model=current_model,
                error=f"余额接口请求失败，HTTP {response.status_code}",
                response_preview=body_preview,
                单位="美元",
            )

        data = _truncate_custom_credits(_parse_json_response(response))
        total_balance = _sum_custom_gateway_balances([data])

        # Custom 成功返回：返回 AI Gateway /credits 的原始 JSON，金额已截断为 2 位小数。
        return _build_payload(
            ok=True,
            provider=provider,
            model=current_model,
            data=data,
            total_balance=total_balance,
            单位="美元",
        )

    except httpx.TimeoutException:
        log.error("get_api_balance 请求超时: provider=%s, endpoint=%s", provider, endpoint)
        return _build_payload(
            ok=False,
            provider=provider,
            model=current_model,
            error="余额接口请求超时，请稍后重试。",
        )
    except Exception as exc:
        log.error("get_api_balance 调用异常。", exc_info=True)
        return _build_payload(
            ok=False,
            provider=provider,
            model=current_model,
            error=f"调用余额接口异常: {str(exc)}",
        )


async def _query_custom_gateway_balance_with_key(
    client: httpx.AsyncClient,
    endpoint: str,
    api_key: str,
    key_index: int,
) -> Dict[str, Any]:
    provider = "custom"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        response = await client.get(endpoint, headers=headers)

        if response.status_code >= 400:
            body_preview = (response.text or "")[:800]
            return {
                "ok": False,
                "provider": provider,
                "key_index": key_index,
                "key_tail": _key_tail(api_key),
                "status_code": response.status_code,
                "error": f"余额接口请求失败，HTTP {response.status_code}",
                "response_preview": body_preview,
                "单位": "美元",
            }

        data = _truncate_custom_credits(_parse_json_response(response))
        return {
            "ok": True,
            "provider": provider,
            "key_index": key_index,
            "key_tail": _key_tail(api_key),
            "status_code": response.status_code,
            "data": data,
            "单位": "美元",
        }

    except httpx.TimeoutException:
        return {
            "ok": False,
            "provider": provider,
            "key_index": key_index,
            "key_tail": _key_tail(api_key),
            "error": "余额接口请求超时，请稍后重试。",
            "单位": "美元",
        }
    except Exception as exc:
        return {
            "ok": False,
            "provider": provider,
            "key_index": key_index,
            "key_tail": _key_tail(api_key),
            "error": f"调用余额接口异常: {str(exc)}",
            "单位": "美元",
        }


async def _get_custom_gateway_balance_multi(
    current_model: str,
    api_keys: List[str],
    log_detailed: bool = False,
) -> Dict[str, Any]:
    provider = "custom"
    endpoint = CUSTOM_GATEWAY_CREDITS_ENDPOINT

    if not api_keys:
        return _build_payload(
            ok=False,
            provider=provider,
            model=current_model,
            error="缺少有效的 CUSTOM_MODEL_API_KEY，无法查询余额。",
        )

    if log_detailed:
        log.info(
            "--- [工具执行]: get_api_balance, model=%s, provider=%s, endpoint=%s, key_count=%s ---",
            current_model,
            provider,
            endpoint,
            len(api_keys),
        )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = [
                _query_custom_gateway_balance_with_key(
                    client=client,
                    endpoint=endpoint,
                    api_key=api_key,
                    key_index=idx + 1,
                )
                for idx, api_key in enumerate(api_keys)
            ]
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_results):
            if isinstance(item, Exception):
                results.append(
                    {
                        "ok": False,
                        "provider": provider,
                        "key_index": idx + 1,
                        "key_tail": _key_tail(api_keys[idx]),
                        "error": f"并发请求异常: {type(item).__name__}: {str(item)}",
                        "单位": "美元",
                    }
                )
            else:
                results.append(item)

        success_count = sum(1 for result in results if result.get("ok"))
        failed_count = len(results) - success_count
        total_balance = _sum_custom_gateway_balances(
            [result.get("data") for result in results if result.get("ok")]
        )

        payload = _build_payload(
            ok=success_count > 0,
            provider=provider,
            model=current_model,
            total_keys=len(api_keys),
            success_count=success_count,
            failed_count=failed_count,
            results=results,
            total_balance=total_balance,
            单位="美元",
        )

        if success_count == 0:
            payload["error"] = "所有 CUSTOM_MODEL_API_KEY 查询均失败。"

        return payload

    except Exception as exc:
        log.error("get_api_balance(Custom Gateway 多 key) 调用异常。", exc_info=True)
        return _build_payload(
            ok=False,
            provider=provider,
            model=current_model,
            error=f"调用余额接口异常: {str(exc)}",
        )


async def _get_custom_siliconflow_balance(
    current_model: str,
    custom_api_key: str,
    log_detailed: bool = False,
) -> Dict[str, Any]:
    provider = "custom"
    endpoint = SILICONFLOW_USER_INFO_ENDPOINT

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {custom_api_key}",
    }

    if log_detailed:
        log.info(
            "--- [工具执行]: get_api_balance, model=%s, provider=%s, endpoint=%s ---",
            current_model,
            provider,
            endpoint,
        )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(endpoint, headers=headers)

        if response.status_code >= 400:
            body_preview = (response.text or "")[:800]
            log.warning(
                "get_api_balance 请求失败: provider=%s, status=%s, body=%s",
                provider,
                response.status_code,
                body_preview,
            )
            return _build_payload(
                ok=False,
                provider=provider,
                model=current_model,
                error=f"余额接口请求失败，HTTP {response.status_code}",
                response_preview=body_preview,
            )

        data = _parse_json_response(response)

        # Custom 成功返回：返回 SiliconFlow /user/info 的原始 JSON。
        return _build_payload(
            ok=True,
            provider=provider,
            model=current_model,
            data=data,
        )

    except httpx.TimeoutException:
        log.error("get_api_balance 请求超时: provider=%s, endpoint=%s", provider, endpoint)
        return _build_payload(
            ok=False,
            provider=provider,
            model=current_model,
            error="余额接口请求超时，请稍后重试。",
        )
    except Exception as exc:
        log.error("get_api_balance 调用异常。", exc_info=True)
        return _build_payload(
            ok=False,
            provider=provider,
            model=current_model,
            error=f"调用余额接口异常: {str(exc)}",
        )


# ===== Custom 余额查询分发 =====


async def _get_custom_balance(
    current_model: str, log_detailed: bool = False
) -> Dict[str, Any]:
    provider = "custom"
    custom_model_url = (os.getenv("CUSTOM_MODEL_URL") or "").strip()
    custom_api_key_source = _get_custom_api_key()

    if not custom_model_url or not custom_api_key_source:
        return _build_payload(
            ok=False,
            provider=provider,
            model=current_model,
            error="缺少环境变量 CUSTOM_MODEL_URL / CUSTOM_MODEL_API_KEY，无法查询余额。",
        )

    try:
        resolved_custom_api_keys = resolve_custom_model_api_keys(custom_api_key_source)
    except ValueError as exc:
        return _build_payload(
            ok=False,
            provider=provider,
            model=current_model,
            error=str(exc),
        )

    custom_api_keys = list(resolved_custom_api_keys.api_keys)
    if not custom_api_keys:
        return _build_payload(
            ok=False,
            provider=provider,
            model=current_model,
            error="缺少有效的 CUSTOM_MODEL_API_KEY，无法查询余额。",
        )

    if custom_model_url == CUSTOM_GATEWAY_BASE_URL:
        if len(custom_api_keys) > 1:
            return await _get_custom_gateway_balance_multi(
                current_model=current_model,
                api_keys=custom_api_keys,
                log_detailed=log_detailed,
            )
        return await _get_custom_gateway_balance(
            current_model=current_model,
            custom_api_key=custom_api_keys[0],
            log_detailed=log_detailed,
        )

    if custom_model_url == SILICONFLOW_BASE_URL:
        return await _get_custom_siliconflow_balance(
            current_model=current_model,
            custom_api_key=custom_api_keys[0],
            log_detailed=log_detailed,
        )

    return _build_payload(
        ok=False,
        provider=provider,
        model=current_model,
        error=(
            f"CUSTOM_MODEL_URL='{custom_model_url}' 暂不支持查询 custom 余额。"
            f"仅支持 '{CUSTOM_GATEWAY_BASE_URL}' 或 '{SILICONFLOW_BASE_URL}'。"
        ),
    )


# ===== 工具入口 =====


@tool_metadata(
    name="查询API余额",
    description="根据当前启用模型，查询 DeepSeek / Kimi(Moonshot) / Custom(AI Gateway) 账户余额。",
    emoji="💰",
    category="系统",
)
async def get_api_balance(log_detailed: bool = False, **kwargs) -> Dict[str, Any]:
    """
    查询模型供应商账户余额（DeepSeek / Kimi / Custom）。

    [调用指南]
    - 仅当用户明确询问剩余 api 余额时使用该工具。
    - 如果用户要求查看 Ta 的余额，绝对禁止使用该工具，而是使用 get_user_profile。
    - 该工具会读取当前全局启用模型，并自动分发到对应供应商的余额接口。
    - Moonshot 支持多 key 并发查询
    """
    try:
        current_model = await _get_current_model()
    except Exception as exc:
        log.error("读取当前 AI 模型失败。", exc_info=True)
        return _build_payload(
            ok=False,
            provider="unknown",
            model=None,
            error=f"读取当前模型失败: {str(exc)}",
        )

    model_lower = (current_model or "").lower().strip()

    if "deepseek" in model_lower:
        return await _get_deepseek_balance(
            current_model=current_model,
            log_detailed=log_detailed,
        )

    if "kimi" in model_lower or "moonshot" in model_lower:
        return await _get_moonshot_balance(
            current_model=current_model,
            log_detailed=log_detailed,
        )

    if model_lower == "custom":
        return await _get_custom_balance(
            current_model=current_model,
            log_detailed=log_detailed,
        )

    return _build_payload(
        ok=False,
        provider="unknown",
        model=current_model,
        error=f"当前模型 '{current_model}' 暂不支持余额查询。",
    )
