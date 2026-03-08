import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import discord
from discord.http import Route
from src.chat.utils.time_utils import BEIJING_TZ
from src.chat.features.tools.tool_metadata import tool_metadata

log = logging.getLogger(__name__)

# ================= 辅助函数 =================

def _format_search_results(messages: List[Dict]) -> List[Dict[str, Any]]:
    """格式化 Discord API 返回的搜索结果"""
    results = []
    for item in messages:
        if isinstance(item, list):
            for message_data in item:
                _process_message_data(message_data, results)
        elif isinstance(item, dict):
             _process_message_data(item, results)
    return results

def _process_message_data(message_data: Dict, results: List[Dict[str, Any]]):
    """处理单条消息数据"""
    if message_data.get("hit", True): 
        author_data = message_data.get("author", {})
        timestamp_str = message_data.get("timestamp")
        
        if timestamp_str:
            utc_dt = datetime.fromisoformat(timestamp_str)
            beijing_dt = utc_dt.astimezone(BEIJING_TZ)
            timestamp_fmt = beijing_dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            timestamp_fmt = "Unknown"
            utc_dt = datetime.min.replace(tzinfo=BEIJING_TZ)

        results.append(
            {
                "id": str(message_data.get("id")),
                "author": f"{author_data.get('username', 'N/A')}#{author_data.get('discriminator', '0000')}",
                "content": message_data.get("content", ""),
                "timestamp": timestamp_fmt,
                "timestamp_obj": utc_dt, # 用于后续混合排序
            }
        )

# ================= 核心工具入口 =================

@tool_metadata(
    name="历史消息",
    description="搜索历史消息。优先全服检索，保底检索当前频道。支持按关键词搜索，或按用户ID查看其近期发言。",
    emoji="📜",
    category="查询",
)
async def search_channel_history(
    query: Optional[str] = None,
    author_id: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    在服务器中搜索历史消息。优先使用全服检索，当触发限制时保底检索当前频道。
    
    [调用指南]
    - **极严限制**: 每次对话**最多只能调用 1 次**！切勿重复或多次调用，否则会导致服务器速率受限(Rate Limit)！
    - **使用场景**: 这不是一个常用的总结工具。**仅当用户明确要求搜索**特定关键词、或查看某人特定历史发言时才调用。
    - **功能区分**: 如果用户要求总结(Summarize)最近的聊天记录或频道活跃情况，**绝对不要**使用此工具，请改用专门的总结工具（如 summarize_channel 等）。
    - **按需提供参数**:
      - 若只想看某人的所有近期发言，仅提供 `author_id`，留空 `query`。
      - 若想搜全频道提及某个词的消息，仅提供 `query`，留空 `author_id`。
      - 若想优先搜特定用户说过的特定词，同时提供 `author_id` 和 `query`。
    - **关键词规范**: `query` 必须是单个最核心的词，严禁带空格的多个词（请自行拆分为最核心的单字/词）。

    Args:
        query (str, optional): 搜索的核心关键词。
        author_id (str, optional): 目标用户的 Discord 数字ID。
    """
    bot = kwargs.get("bot")
    channel = kwargs.get("channel")
    
    if not bot or not channel or not channel.guild:
        log.error("机器人实例、频道或服务器对象在上下文中不可用。")
        return {"results": [], "count": 0}

    if not query and not author_id:
        return {"results": [], "count": 0}

    # 如果 query 包含空格，尝试只取第一部分
    if query and " " in query:
        original_query = query
        parts = query.split()
        query = max(parts, key=len)
        log.warning(f"检测到复合关键词 '{original_query}'，已自动截取为 '{query}' 进行搜索。")

    tasks = []
    
    # Task 1: 搜索特定作者发布的，含关键词的内容
    if author_id and query:
        tasks.append(_execute_combined_search(bot, channel, query=query, author_id=author_id))
    else:
        tasks.append(asyncio.sleep(0, result=[]))

    # Task 2: 所有人发布的，含关键词的内容
    if query:
        tasks.append(_execute_combined_search(bot, channel, query=query, author_id=None))
    else:
        tasks.append(asyncio.sleep(0, result=[]))

    # Task 3: 特定作者发布的，全部内容
    # 仅当没有关键词时，才搜索该作者的全部内容，避免结果被不相关发言刷屏
    if author_id and not query:
        tasks.append(_execute_combined_search(bot, channel, query=None, author_id=author_id))
    else:
        tasks.append(asyncio.sleep(0, result=[]))

    # 并发执行三大分支
    results_groups = await asyncio.gather(*tasks)
    
    author_query_results = results_groups[0]
    all_query_results = results_groups[1]
    author_all_results = results_groups[2]

    # 合并与去重逻辑
    final_results = []
    seen_ids = set()
    seen_content = set()

    def add_unique(items):
        for item in items:
            msg_id = item["id"]
            msg_content = item.get("content", "").strip()
            msg_author = item.get("author", "")
            
            content_key = f"{msg_author}:{msg_content}"

            if msg_id not in seen_ids and content_key not in seen_content:
                if "timestamp_obj" in item:
                    del item["timestamp_obj"]
                final_results.append(item)
                seen_ids.add(msg_id)
                seen_content.add(content_key)

    # 严格按照优先级合并
    add_unique(author_query_results)
    add_unique(all_query_results)
    add_unique(author_all_results)
    
    # 限制返回数量，防止 Token 爆炸
    MAX_RETURN = 300
    final_results = final_results[:MAX_RETURN]
    
    log.info(f"双擎搜索完成。Query: {query}, Author: {author_id}, 去重后结果 {len(final_results)} 条。")
    return {
        "results": final_results,
        "count": len(final_results)
    }

# ================= 搜索引擎 =================

async def _execute_combined_search(bot, channel: discord.TextChannel, query: Optional[str] = None, author_id: Optional[str] = None) -> List[Dict]:
    """优先遍历当前频道历史记录，若结果不足再降级调用全服 API 补充"""
    # 1. 优先使用本地历史搜索 (安全、稳定，基于 channel.history)
    history_results = await _execute_history_search(channel, query, author_id)
    
    # 2. 判断是否需要降级调用 API (当本地搜到的结果极少时，比如 < 20 条)
    # 这样可以大幅减少触发 API 限制和 403 错误的概率
    if len(history_results) < 20:
        log.info(f"本地频道搜索仅找到 {len(history_results)} 条结果，尝试调用全服 API 补充数据...")
        # API 搜索传 guild_id (搜全服)
        api_results = await _execute_api_search(bot, channel.guild.id, query, author_id)
    else:
        api_results = []
    
    combined = api_results + history_results
    
    # 因为存在两个来源的结果合并，必须重新基于时间排序去重
    # (外部的 search_channel_history 会处理按内容/ID去重，这里只需时间降序)
    combined.sort(key=lambda x: x["timestamp_obj"], reverse=True)
    return combined

async def _execute_api_search(bot, guild_id: int, query: Optional[str], author_id: Optional[str]) -> List[Dict]:
    """官方隐藏的 Search API (全服范围)"""
    try:
        # 直接使用全服路由
        route = Route("GET", "/guilds/{guild_id}/messages/search", guild_id=guild_id)
        params = {}
        if query: params["content"] = query
        if author_id: params["author_id"] = author_id
        
        data = await bot.http.request(route, params=params)
        messages = data.get("messages", [])
        
        # 拿到什么直接格式化什么，不再剔除非当前频道的消息
        return _format_search_results(messages)

    except discord.Forbidden:
        log.info(f"全服API搜索被拦截 (403)，将部分依赖 History 引擎。服务器: {guild_id}")
        return []
    except Exception as e:
        log.warning(f"全服 API 搜索发生错误: {e}")
        return []

async def _execute_history_search(channel: discord.TextChannel, query: Optional[str], author_id: Optional[str]) -> List[Dict]:
    """合法的 History 遍历 API (保底机制 - 仅限当前频道)"""
    results = []
    try:
        # limit 可根据需求调整。因为只看当前频道，稍微调大一点也没关系。
        async for msg in channel.history(limit=1000):
            match = True
            if author_id and str(msg.author.id) != str(author_id):
                match = False
            if match and query:
                if query.lower() not in msg.content.lower():
                    match = False
                    
            if match:
                utc_dt = msg.created_at
                if utc_dt.tzinfo is None:
                    utc_dt = utc_dt.replace(tzinfo=discord.utils.utcnow().tzinfo)
                beijing_dt = utc_dt.astimezone(BEIJING_TZ)
                
                disc = getattr(msg.author, 'discriminator', '0000')
                author_name = f"{msg.author.name}#{disc}" if disc != '0' else msg.author.name

                results.append({
                    "id": str(msg.id),
                    "author": author_name,
                    "content": msg.content,
                    "timestamp": beijing_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "timestamp_obj": utc_dt
                })
                
                # 熔断：本地查到 300 条就收手，防止卡死
                if len(results) >= 300: 
                    break
    except discord.Forbidden:
        log.error(f"没有读取当前频道 {channel.id} 历史记录的权限！")
    except Exception as e:
        log.warning(f"当前频道 History 搜索发生错误: {e}")
        
    return results

# Metadata 必须保留给工具加载器
# 虽然使用了 @tool_metadata 装饰器，但为了兼容旧的加载机制（如果有），保留此骨架。
# 具体的参数描述已被移至函数的 docstring 中供 AI 识别。
SEARCH_CHANNEL_HISTORY_TOOL = {
    "type": "function",
    "function": {
        "name": "search_channel_history",
        "description": "搜索聊天记录。支持按关键词跨频道搜索，或直接查看指定用户的近期发言。",
    },
}
