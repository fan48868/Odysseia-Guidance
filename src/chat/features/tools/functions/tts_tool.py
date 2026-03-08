# -*- coding: utf-8 -*-

import logging
import io
import discord
import re
from pydantic import BaseModel, Field
from typing import Optional
import edge_tts
from src.chat.features.tools.tool_metadata import tool_metadata

log = logging.getLogger(__name__)

def detect_language(text: str) -> str:
    """
    根据文本内容检测语言。
    - 如果包含日语假名，则认为是日语。
    - 如果包含韩语谚文，则认为是韩语。
    - 否则，默认为中文。
    """
    if re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text):
        return "ja-JP"
    if re.search(r'[\uAC00-\uD7A3]', text):
        return "ko-KR"
    return "zh-CN"

def adjust_pitch_hz(pitch: str, delta_hz: int) -> str:
    """
    在原有 pitch 基础上增减 Hz。
    输入格式要求为 '[+/-]NHz'，如 '+25Hz'、'-10Hz'。
    若格式不匹配则返回原值。
    """
    match = re.fullmatch(r'([+-]?)(\d+)Hz', pitch.strip())
    if not match:
        return pitch

    sign, value_str = match.groups()
    base = int(value_str)
    if sign == "-":
        base = -base

    adjusted = base + delta_hz
    return f"{adjusted:+d}Hz"

class TTSParams(BaseModel):
    text: str = Field(..., description="要转换成语音的文字内容。")
    filename: Optional[str] = Field(None, description="音频文件的名字。请根据内容生成一个10字以内的标题，如'余额播报'、'主人亲启'等。")
    rate: str = Field("+15%", description="语速调整字符串。格式为 '[+/-]N%'，例如 '+20%' 表示加快，'-10%' 表示减慢。")
    pitch: str = Field("+25Hz", description="音调调整字符串。格式为 '[+/-]NHz'，例如 '+35Hz' 表示变高，'+10Hz' 表示变低。")

@tool_metadata(
    name="文字转语音",
    description="将文字转换为语音音频文件并发送到当前频道。",
    emoji="🎙️",
    category="工具",
)
async def tts_tool(
    params: TTSParams,
    **kwargs,
) -> str:
    """
    [工具说明]
    这是一个多语言文字转语音 (TTS) 工具，支持中文、英文、日语和韩语。
    当用户明确提及"调用TTS/发语音"时，你**必须**使用此工具，禁止使用文字发送。
    自由调用：此工具可以日常随意选择调用。鼓励多调用此工具，来增加日常互动的惊喜感。

    [语言使用规则]
    1.  **中文和英文**：可以混合在一条消息里发送。
    2.  **日语/韩语**：如果要发送日语或韩语语音，则文本**必须**只包含该语言的字符，**不准**夹杂中文、英文或其他语言。

    [参数利用指南]
    - **rate (语速)**: 范围在 -50% 到 +100% 之间。增加数值会让表达更急促、活泼；减少数值会让表达更沉稳、迟缓。
    - **pitch (音调)**: 范围建议在 +10Hz 到 +50Hz 之间。增加数值会让声音更清脆、年轻；减少数值会让声音更雄浑、成熟。
    - 推荐使用: rate +15% ,pitch +25Hz。
    - 你可以根据当前的对话语境，自主组合这两个参数以达到最自然的表达效果。

    [情绪输出指南]
    - 表示**害羞/迟疑**情绪时，调低 rate (+0%)，调高 pitch (+35Hz)，并在文字中大量使用 `...`。
    - 表示**傲娇/生气**: 调高 rate (+30%)，语气要短促，多用 `！`。
    - 表示**撒娇/元气**:多用 `~~` 或语气词。

    [执行逻辑]
    - 工具会根据文本内容自动选择语音引擎。
    - 生成一个 .mp3 音频流并直接发送至 Discord 频道。
    """
    channel = kwargs.get("channel")
    if not channel or not isinstance(channel, discord.abc.Messageable):
        return "错误：找不到有效的消息频道。"

    if not isinstance(params, TTSParams):
        try:
            params = TTSParams(**params)
        except Exception as e:
            return f"参数解析失败: {e}"
    # 文件名处理：移除非法字符并限制长度
    raw_name = params.filename or "神所娘的语音消息"
    clean_name = re.sub(r'[\\/:*?"<>|]', '', raw_name)[:10]
    display_filename = f"{clean_name}.mp3"

    try:
        # 根据文本内容自动选择语音
        lang = detect_language(params.text)
        speech_pitch = params.pitch
        if lang == "ja-JP":
            voice = "ja-JP-NanamiNeural"
            speech_pitch = adjust_pitch_hz(params.pitch, 20)
        elif lang == "ko-KR":
            voice = "ko-KR-SunHiNeural"
        else:
            voice = "zh-CN-XiaoyiNeural"

        communicate = edge_tts.Communicate(
            text=params.text,
            voice=voice,
            rate=params.rate,
            pitch=speech_pitch
        )

        audio_data = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.write(chunk["data"])
        
        audio_data.seek(0)

        # 发送到 Discord
        file = discord.File(fp=audio_data, filename=display_filename)
        await channel.send(file=file)

        return "成功：语音文件已发送。"

    except Exception as e:
        log.error(f"TTS 运行异常: {e}")
        return f"错误：生成语音时发生故障: {e}"