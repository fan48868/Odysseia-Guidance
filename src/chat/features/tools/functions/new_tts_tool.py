# -*- coding: utf-8 -*-

import os
import logging
import io
import discord
import re
import asyncio
from pydantic import BaseModel, Field
from typing import Optional
from gradio_client import Client
from src.chat.features.tools.tool_metadata import tool_metadata

log = logging.getLogger(__name__)

# ==========================================
# Gradio API 地址不要硬编码，请在项目根目录的 .env 中配置：
#   GRADIO_API_URL=https://xxxx.ngrok-free.dev
# ==========================================
def _get_gradio_api_url() -> str:
    url = os.getenv("GRADIO_API_URL")
    if not url:
        raise RuntimeError(
            "GRADIO_API_URL 未在环境变量中设置。请在项目根目录的 .env 文件中配置 "
            "GRADIO_API_URL=<你的 gradio/ngrok 地址>。"
        )
    return url

class NewTTSParams(BaseModel):
    text: str = Field(..., description="要转换成语音的文字内容。")
    emotion_design: str = Field(
        ..., 
        description="针对这段话的具体语气、情感和语调变化描述。请详细描写情绪起伏、语速快慢、停顿和尾音处理。"
    )
    filename: str = Field(..., description="音频文件的名字。请根据内容生成一个10字以内的标题，如'傲娇抱怨'、'开心问候'等。无特殊要求中文名。")


async def _background_generate_and_send(channel: discord.abc.Messageable, params: NewTTSParams, status_message: discord.Message = None, user_id: str = None):
    """
    后台生成并发送语音的防断线任务。
    使用 asyncio.to_thread 将阻塞式的 Gradio 网络请求放入单独的线程执行，防止阻塞 Discord 机器人的心跳。
    """
    try:
        # 拼接固定基础音色特征与 AI 动态生成的具体语气
        base_design = (
            "年轻女性声线，19岁少女音色。明亮清澈，带有自然的元气感。语速中等偏快，活泼但不急促。"
            "情感表达：开心时音调上扬，尾音带俏皮感；害羞时语速放慢，声音略微发颤；傲娇时语气先硬后软，转折自然。"
        )
        final_design = f"{base_design} 当前具体表现要求：{params.emotion_design}"

        # 定义阻塞的 API 调用函数
        def call_gradio_api():
            client = Client(_get_gradio_api_url())
            return client.predict(
                text=params.text,
                lang_disp="Auto",      # 让模型自动识别语种
                design=final_design,   # 组合后的音色提示词
                api_name="/run_voice_design"
            )

        # 在独立线程中运行 API，避免阻塞主事件循环
        result = await asyncio.to_thread(call_gradio_api)
        
        filepath = result[0] # Gradio 返回的元组中，索引 0 是文件绝对路径
        
        # 处理文件名
        raw_name = params.filename or "狮子娘的语音消息"
        clean_name = re.sub(r'[\\/:*?"<>|]', '', raw_name)[:10]
        display_filename = f"{clean_name}.wav" # Qwen3 通常输出 wav

        # 读取文件到内存并发送到 Discord
        with open(filepath, "rb") as f:
            audio_bytes = f.read()

        file = discord.File(fp=io.BytesIO(audio_bytes), filename=display_filename)
        # 构造消息内容，如果有user_id则艾特发起人
        mention = f"<@{user_id}> " if user_id else ""
        await channel.send(content=f"{mention}🎵 {clean_name} 生成好啦！", file=file)
        log.info(f"已发送🎵{clean_name}.wav至频道")
        
        # 成功发送音频后删除状态消息
        if status_message:
            try:
                await status_message.delete()
                log.info("已删除TTS生成状态消息")
            except Exception as del_e:
                log.warning(f"删除状态消息失败: {del_e}")

    except Exception as e:
        log.error(f"新版 TTS 后台生成异常: {e}")
        await channel.send(content=f"❌ 呜呜，语音生成失败了，可能是 API 断开或者超时了：{e}")
        # 发生异常时也尝试删除状态消息
        if status_message:
            try:
                await status_message.delete()
            except Exception as del_e:
                log.warning(f"删除状态消息失败: {del_e}")


@tool_metadata(
    name="新版tts",
    description="使用先进的 Qwen3 模型生成带有细腻情感的高质量语音并发送到当前频道。",
    emoji="🎙️",
    category="工具",
)
async def new_tts_tool(
    params: NewTTSParams,
    **kwargs,
) -> str:
    """
    [工具说明]
    这是一个高拟真的情感文字转语音 (TTS) 工具。
    当用户明确提及"调用TTS/发语音"或你想要用更生动的方式回应时，请使用此工具。

    [你的声音人设（基础特征已内置，无需你操心）]

    [参数 emotion_design 生成指南]
    这是本工具最重要的参数！你需要根据当前对话语境，为你说的这句 `text` 编写“配音导演指令”。
    请参考以下范例来撰写 `emotion_design` 参数：
    
    范例（傲娇不满）：
    "整体情绪：表面抱怨、实则撒娇，带着'被你打败了'的无奈感。语调变化：开头'主人～！'尾调上扬，带点嗔怪；'真的很过分诶'语速略快，音量稍提，强调委屈；'不过——'这里停顿半拍，语气软下来；'勉为其难'四个字放慢、轻读，假装不情愿；结尾'啦！'尾音自然上扬，藏不住的小开心。"

    [执行逻辑与你的回复策略]
    1. 由于高质量模型生成速度较慢（大约需要 30 秒）。
    2. 工具被调用后，会立刻向你返回“成功触发”的状态，而不会让你一直干等。
    3. 音频将在后台悄悄生成，生成完毕后会自动发送到频道里。
    4. **非常重要**：收到本工具的成功返回后，你可以在接下来的文字回复中，顺便用自然的语气告诉用户：“语音正在努力生成中哦，大概需要等三十秒左右～”
    """
    channel = kwargs.get("channel")
    if not channel or not isinstance(channel, discord.abc.Messageable):
        return "错误：找不到有效的消息频道。"

    if not isinstance(params, NewTTSParams):
        try:
            params = NewTTSParams(**params)
        except Exception as e:
            return f"参数解析失败: {e}"

    # 获取用户ID用于艾特发起人
    user_id = kwargs.get("user_id")

    # 先发送状态消息到频道
    status_message = None
    try:
        status_message = await channel.send("已调用tts，正在生成中🎵...")
    except Exception as e:
        log.warning(f"发送TTS状态消息失败: {e}")

    # 立即抛出后台任务，不阻塞当前流程
    asyncio.create_task(_background_generate_and_send(channel, params, status_message, user_id))

    # 立即返回成功信息给 AI 侧（此时文件还没生成完）
    return (
        "成功：已成功向服务器发送语音生成请求。由于生成大约需要 30 秒，"
        "工具已转入后台运行。请在接下来的文字回复中，用符合人设的自然语气提醒用户稍等大约半分钟。"
    )