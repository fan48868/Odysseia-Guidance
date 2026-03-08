import asyncio
import logging
import os
import re
from typing import Any, Dict, List

import httpx

from src.chat.features.chat_settings.services.chat_settings_service import (
    chat_settings_service,
)
from src.chat.features.tools.tool_metadata import tool_metadata

log = logging.getLogger(__name__)


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


async def _query_balance_with_key(
    client: httpx.AsyncClient,
    endpoint: str,
    api_key: str,
    provider: str,
    key_index: int,
) -> Dict[str, Any]:
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

        try:
            data = response.json()
        except Exception:
            data = {"raw_text": (response.text or "")[:2000]}

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
    except Exception as e:
        return {
            "ok": False,
            "provider": provider,
            "key_index": key_index,
            "key_tail": _key_tail(api_key),
            "error": f"调用余额接口异常: {str(e)}",
        }


@tool_metadata(
    name="查询API余额",
    description="根据当前启用模型，查询 DeepSeek 或 Kimi(Moonshot) 账户余额。",
    emoji="💰",
    category="系统",
)
async def get_balance(log_detailed: bool = False, **kwargs) -> Dict[str, Any]:
    """
    查询模型供应商账户余额（DeepSeek / Kimi）。

    [调用指南]
    - 仅当用户明确询问剩余余额时使用该工具。
    - 该工具会读取当前全局启用模型，并自动选择对应余额接口。
    - 当是 Kimi/Moonshot 时，会拆分 MOONSHOT_API_KEY 中的多 key 并并发查询，返回所有结果。

    Returns:
        统一结构的字典结果，包含:
        - ok: 是否成功（至少一个 key 查询成功则为 true）
        - provider: deepseek / moonshot / unknown
        - model: 当前模型名
        - endpoint: 调用地址
        - data: 上游返回的 JSON（DeepSeek 成功时）
        - results: 多 key 结果列表（Moonshot 时）
        - error: 错误信息（失败时）
    """
    try:
        current_model = await chat_settings_service.get_current_ai_model()
    except Exception as e:
        log.error("读取当前 AI 模型失败。", exc_info=True)
        return {
            "ok": False,
            "provider": "unknown",
            "model": None,
            "endpoint": None,
            "error": f"读取当前模型失败: {str(e)}",
        }

    model_lower = (current_model or "").lower()

    provider = "unknown"
    endpoint = None

    if "deepseek" in model_lower:
        provider = "deepseek"
        endpoint = "https://api.deepseek.com/user/balance"
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return {
                "ok": False,
                "provider": provider,
                "model": current_model,
                "endpoint": endpoint,
                "error": "缺少环境变量 DEEPSEEK_API_KEY，无法查询余额。",
            }

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        if log_detailed:
            log.info(
                f"--- [工具执行]: get_balance, model={current_model}, provider={provider}, endpoint={endpoint} ---"
            )

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(endpoint, headers=headers)

            if response.status_code >= 400:
                body_preview = (response.text or "")[:800]
                log.warning(
                    f"get_balance 请求失败: provider={provider}, status={response.status_code}, body={body_preview}"
                )
                return {
                    "ok": False,
                    "provider": provider,
                    "model": current_model,
                    "endpoint": endpoint,
                    "error": f"余额接口请求失败，HTTP {response.status_code}",
                    "response_preview": body_preview,
                }

            try:
                data = response.json()
            except Exception:
                data = {"raw_text": (response.text or "")[:2000]}

            return {
                "ok": True,
                "provider": provider,
                "model": current_model,
                "endpoint": endpoint,
                "data": data,
            }

        except httpx.TimeoutException:
            log.error(f"get_balance 请求超时: provider={provider}, endpoint={endpoint}")
            return {
                "ok": False,
                "provider": provider,
                "model": current_model,
                "endpoint": endpoint,
                "error": "余额接口请求超时，请稍后重试。",
            }
        except Exception as e:
            log.error("get_balance 调用异常。", exc_info=True)
            return {
                "ok": False,
                "provider": provider,
                "model": current_model,
                "endpoint": endpoint,
                "error": f"调用余额接口异常: {str(e)}",
            }

    if "kimi" in model_lower or "moonshot" in model_lower:
        provider = "moonshot"
        endpoint = "https://api.moonshot.cn/v1/users/me/balance"
        raw_keys = os.getenv("MOONSHOT_API_KEY", "")
        api_keys = _split_api_keys(raw_keys)

        if not api_keys:
            return {
                "ok": False,
                "provider": provider,
                "model": current_model,
                "endpoint": endpoint,
                "error": "缺少环境变量 MOONSHOT_API_KEY，无法查询余额。",
            }

        if log_detailed:
            log.info(
                f"--- [工具执行]: get_balance, model={current_model}, provider={provider}, endpoint={endpoint}, key_count={len(api_keys)} ---"
            )

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                tasks = [
                    _query_balance_with_key(
                        client=client,
                        endpoint=endpoint,
                        api_key=api_key,
                        provider=provider,
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

            success_count = sum(1 for r in results if r.get("ok"))
            failed_count = len(results) - success_count

            payload: Dict[str, Any] = {
                "ok": success_count > 0,
                "provider": provider,
                "model": current_model,
                "endpoint": endpoint,
                "total_keys": len(api_keys),
                "success_count": success_count,
                "failed_count": failed_count,
                "results": results,
            }

            if success_count == 0:
                payload["error"] = "所有 MOONSHOT_API_KEY 查询均失败。"

            return payload

        except Exception as e:
            log.error("get_balance(Moonshot 多 key) 调用异常。", exc_info=True)
            return {
                "ok": False,
                "provider": provider,
                "model": current_model,
                "endpoint": endpoint,
                "error": f"调用余额接口异常: {str(e)}",
            }

    return {
        "ok": False,
        "provider": "unknown",
        "model": current_model,
        "endpoint": None,
        "error": f"当前模型 '{current_model}' 暂不支持余额查询。",
    }