#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
点反应测试脚本（本地 Ollama + qwen3:0.6b + 非思考模式）

需求实现：
1. 使用 qwen3:0.6b
2. enable_thinking=False
3. 使用项目当前“原提示词”（chat_config.REACTION_AI["prompt"]）
4. 使用 coin_cog 同款 emoji 提取正则
5. 输入一段文字，输出“模型完整结果（原文）”和“提取到的表情”
"""

import asyncio
import json
import os
import re
import sys

import httpx

# 确保可以导入 src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.chat.config import chat_config

# 与 coin_cog 一致的提取规则
EMOJI_RE = re.compile(
    r"((?:[\U0001F1E6-\U0001F1FF]{2})|(?:[\U0001F300-\U0001FAFF\u2600-\u27BF])(?:\uFE0F|\uFE0E)?(?:[\U0001F3FB-\U0001F3FF])?(?:\u200D(?:[\U0001F300-\U0001FAFF\u2600-\u27BF])(?:\uFE0F|\uFE0E)?(?:[\U0001F3FB-\U0001F3FF])?)*)"
)

OLLAMA_BASE_URL = str(
    chat_config.REACTION_AI.get("url", "http://host.docker.internal:11434/v1")
).rstrip("/")
MODEL_NAME = "qwen3.5:0.8b"
SYSTEM_PROMPT = chat_config.REACTION_AI.get("prompt", "")
# 这里给测试脚本更宽松超时，避免模型首轮加载时触发 ReadTimeout
TIMEOUT_SECONDS = max(int(chat_config.REACTION_AI.get("timeout", 8)), 30)

# --- 为“卡思维链”增加硬限制 ---
MAX_OUTPUT_TOKENS = 16
STRICT_STOPS = ["\n", "<think>", "</think>"]
REQUEST_TEMPERATURE = 0.2

DEFAULT_INPUT = "来文爱"


def pick_emoji(raw: str):
    if not raw:
        return None
    match = EMOJI_RE.search(str(raw))
    if not match:
        return None
    return match.group(0)


def extract_content(data: dict) -> str:
    try:
        return str(data["choices"][0]["message"]["content"]).strip()
    except Exception:
        return ""


async def request_chat(payload: dict):
    api_url = f"{OLLAMA_BASE_URL}/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            resp = await client.post(api_url, json=payload)
            data = {}
            try:
                data = resp.json()
            except Exception:
                pass
            return resp.status_code, data
    except Exception as e:
        return -1, {"error": f"{type(e).__name__}: {e}"}


async def main():
    print("=== 点反应模型测试 ===")
    print(f"URL: {OLLAMA_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print("模式: enable_thinking=False")

    user_input = input(f"请输入测试文本（回车默认：{DEFAULT_INPUT}）：").strip()
    if not user_input:
        user_input = DEFAULT_INPUT

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
                + "\n补充硬规则：禁止输出任何思考过程。只输出1个emoji，立刻结束。",
            },
            {"role": "user", "content": user_input[:500]},
        ],
        "stream": False,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
        # OpenAI兼容参数硬限制
        "temperature": REQUEST_TEMPERATURE,
        "top_p": 0.8,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "stop": STRICT_STOPS,
        # Ollama 原生 options（双保险）
        "options": {
            "temperature": REQUEST_TEMPERATURE,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0,
            "num_predict": MAX_OUTPUT_TOKENS,
            "stop": STRICT_STOPS,
        },
    }

    # 兼容回退：尽量保留“禁思考 + 短输出”
    fallback_payload = {
        "model": MODEL_NAME,
        "messages": payload["messages"],
        "stream": False,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": REQUEST_TEMPERATURE,
        "top_p": 0.8,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "stop": STRICT_STOPS,
    }

    status, data = await request_chat(payload)
    if status == -1:
        print("\n请求异常（网络/超时）：")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    if status >= 400:
        print(f"\n首轮请求失败（status={status}），尝试兼容回退参数...")
        if data:
            print(json.dumps(data, ensure_ascii=False, indent=2))
        status, data = await request_chat(fallback_payload)

    if status == -1:
        print("\n回退请求异常（网络/超时）：")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    if status >= 400:
        print(f"\n请求失败（status={status}）")
        if data:
            print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    raw_output = extract_content(data)
    emoji = pick_emoji(raw_output)

    print("\n--- 模型完整结果（原文）---")
    print(raw_output if raw_output else "(空输出)")

    print("\n--- 提取到的表情 ---")
    print(emoji if emoji else "未提取到 emoji")


if __name__ == "__main__":
    asyncio.run(main())