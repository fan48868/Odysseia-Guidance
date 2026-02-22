#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG 查询生成测试脚本
用于验证 summarize_for_rag 方法是否能正确将用户的对话转化为独立的搜索查询
"""

import sys
import os
import asyncio
import logging

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 尝试加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("提示: 未找到 dotenv 模块或 .env 文件，将依赖系统环境变量。")

from src.chat.services.gemini_service import gemini_service

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def main():
    print("=" * 60)
    print("RAG 查询重写/生成测试工具")
    print("=" * 60)

    if not gemini_service.is_available():
        print("❌ 错误: Gemini 服务未就绪，请检查 GOOGLE_API_KEYS_LIST 环境变量配置。")
        return

    # --- 预定义测试用例 ---
    test_cases = [
        {
            "description": "无上下文的直接查询",
            "query": "萌猫语模式是什么？",
            "history": []
        },
        {
            "description": "带有上下文的指代消解 (它 -> 艾尔登法环)",
            "query": "它好玩吗？",
            "history": [
                {"role": "user", "parts": ["我想了解一下《艾尔登法环》。"]},
                {"role": "model", "parts": ["《艾尔登法环》是一款由FromSoftware开发的魂系开放世界游戏。"]}
            ]
        },
        {
            "description": "指令型查询 (尝试提取核心意图)",
            "query": "请帮我开启这个模式",
            "history": [
                {"role": "user", "parts": ["听说有一个沉浸式扮演模式？"]},
                {"role": "model", "parts": ["是的，这是一个实验性功能。"]}
            ]
        }
    ]

    print(f"正在运行 {len(test_cases)} 个预定义测试用例...\n")

    for i, case in enumerate(test_cases, 1):
        print(f"测试 {i}: {case['description']}")
        print(f"原始输入: '{case['query']}'")
        if case['history']:
            last_msg = case['history'][-1]['parts'][0]
            print(f"上文语境: ...{last_msg[-20:]} (共 {len(case['history'])} 条历史)")
        
        result = await gemini_service.summarize_for_rag(
            latest_query=case['query'],
            user_name="TestUser",
            conversation_history=case['history']
        )
        print(f"👉 生成查询: '{result}'")
        print("-" * 40)

    # --- 交互模式 ---
    print("\n进入交互模式 (输入 'q' 或 'exit' 退出):")
    while True:
        user_input = input("\n请输入模拟查询: ").strip()
        if user_input.lower() in ('q', 'quit', 'exit'):
            break
        
        if not user_input:
            continue

        print("正在生成 RAG 查询...")
        try:
            # 交互模式默认不带历史记录，主要测试关键词提取能力
            result = await gemini_service.summarize_for_rag(
                latest_query=user_input,
                user_name="InteractiveUser",
                conversation_history=[] 
            )
            print(f"👉 生成结果: '{result}'")
        except Exception as e:
            print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    asyncio.run(main())