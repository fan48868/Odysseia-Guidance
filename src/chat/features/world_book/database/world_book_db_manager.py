# src/chat/features/world_book/database/world_book_db_manager.py

import aiosqlite
import logging
import yaml
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

# 设置日志记录器
log = logging.getLogger(__name__)

# 定义数据库文件的路径
DB_PATH = Path("data/world_book.sqlite3")
# 这里假设你把所有数据都放这一个文件里，或者你可以把成员数据放单独的文件
KNOWLEDGE_YAML_PATH = Path("src/chat/features/world_book/data/knowledge.yml")

# 定义数据库结构
SQL_SCHEMA = """
-- 通用设定类表
CREATE TABLE IF NOT EXISTS "categories" ("id" INTEGER PRIMARY KEY AUTOINCREMENT, "name" TEXT NOT NULL UNIQUE);
CREATE TABLE IF NOT EXISTS "general_knowledge" ("id" TEXT PRIMARY KEY, "title" TEXT, "name" TEXT, "content_json" TEXT, "category_id" INTEGER, "contributor_id" INTEGER, "created_at" TEXT DEFAULT CURRENT_TIMESTAMP, "status" TEXT DEFAULT 'approved', FOREIGN KEY ("category_id") REFERENCES "categories"("id"));
CREATE TABLE IF NOT EXISTS "aliases" ("id" INTEGER PRIMARY KEY AUTOINCREMENT, "entry_id" TEXT NOT NULL, "alias" TEXT NOT NULL, FOREIGN KEY ("entry_id") REFERENCES "general_knowledge"("id"));
CREATE TABLE IF NOT EXISTS "knowledge_refers_to" ("id" INTEGER PRIMARY KEY AUTOINCREMENT, "entry_id" TEXT NOT NULL, "reference" TEXT NOT NULL, FOREIGN KEY ("entry_id") REFERENCES "general_knowledge"("id"));

-- 社区成员类表
CREATE TABLE IF NOT EXISTS "community_members" ("id" TEXT PRIMARY KEY, "title" TEXT, "discord_id" TEXT, "discord_number_id" TEXT, "history" TEXT, "content_json" TEXT, "status" TEXT DEFAULT 'approved');
CREATE TABLE IF NOT EXISTS "member_discord_nicknames" ("id" INTEGER PRIMARY KEY AUTOINCREMENT, "member_id" TEXT NOT NULL, "nickname" TEXT NOT NULL, FOREIGN KEY ("member_id") REFERENCES "community_members"("id"));

-- 待审核表
CREATE TABLE IF NOT EXISTS "pending_entries" ("id" INTEGER PRIMARY KEY AUTOINCREMENT, "entry_type" TEXT NOT NULL, "data_json" TEXT NOT NULL, "message_id" INTEGER, "channel_id" INTEGER NOT NULL, "guild_id" INTEGER NOT NULL, "proposer_id" INTEGER NOT NULL, "status" TEXT NOT NULL DEFAULT 'pending', "created_at" DATETIME DEFAULT CURRENT_TIMESTAMP, "expires_at" TEXT NOT NULL);

-- 索引
CREATE INDEX IF NOT EXISTS "idx_community_members_discord_id" ON "community_members" ("discord_number_id");
CREATE INDEX IF NOT EXISTS "idx_pending_entries_status_expires" ON "pending_entries" ("status", "expires_at");
"""

class WorldBookDBManager:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # 缓存结构: List[Tuple(keyword, entry_id, type)]
        # type: 'general' | 'member'
        self._keyword_cache = []

    async def init_async(self):
        log.info(f"正在初始化 World Book 数据库: {self.db_path}...")
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.executescript(SQL_SCHEMA)
                await self._import_knowledge_yaml(db)
                await db.commit()
            
            await self._refresh_keyword_cache()
            log.info("World Book 数据库准备就绪。")
        except Exception as e:
            log.error(f"World Book DB Init Error: {e}", exc_info=True)

    async def _import_knowledge_yaml(self, db):
        """
        智能导入：支持列表格式，自动区分社区成员和通用设定
        """
        if not KNOWLEDGE_YAML_PATH.exists(): return
        try:
            with open(KNOWLEDGE_YAML_PATH, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if not data: return

            # 统一转为列表处理 (兼容根节点是 list 或 dict)
            items = []
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                # 兼容旧格式：如果有 entries 键
                items.extend(data.get('entries', []))
                # 如果有 members 键
                items.extend(data.get('members', []))

            count_general = 0
            count_member = 0

            for item in items:
                entry_id = item.get('id')
                if not entry_id: continue

                # 判断类型：通过 metadata.category 或 字段特征
                metadata = item.get('metadata', {})
                category = metadata.get('category', '')
                
                # --- 分支 A: 社区成员 ---
                if category == "社区成员" or 'discord_nickname' in item:
                    content_json = json.dumps(item.get('content', {}), ensure_ascii=False)
                    
                    # 插入成员主表
                    await db.execute(
                        """
                        INSERT OR IGNORE INTO community_members 
                        (id, title, discord_id, discord_number_id, history, content_json, status) 
                        VALUES (?, ?, ?, ?, ?, ?, 'approved')
                        """,
                        (
                            entry_id, 
                            item.get('title'), 
                            item.get('discord_id', ''), 
                            item.get('discord_number_id', ''),
                            item.get('history', ''),
                            content_json
                        )
                    )
                    
                    # 插入昵称
                    nicknames = item.get('discord_nickname', [])
                    for nick in nicknames:
                        if nick:
                            await db.execute(
                                "INSERT OR IGNORE INTO member_discord_nicknames (member_id, nickname) VALUES (?, ?)",
                                (entry_id, nick)
                            )
                    count_member += 1

                # --- 分支 B: 通用设定 ---
                else:
                    # 处理分类 ID
                    cat_id = None
                    cat_name = item.get('category') or category # 兼容两种写法
                    if cat_name:
                        await db.execute("INSERT OR IGNORE INTO categories (name) VALUES (?)", (cat_name,))
                        async with db.execute("SELECT id FROM categories WHERE name=?", (cat_name,)) as c:
                            row = await c.fetchone()
                            if row: cat_id = row[0]
                    
                    content_json = json.dumps(item.get('content', {}), ensure_ascii=False)
                    
                    # 插入通用表
                    await db.execute(
                        """
                        INSERT OR IGNORE INTO general_knowledge 
                        (id, title, name, content_json, category_id, status) 
                        VALUES (?, ?, ?, ?, ?, 'approved')
                        """,
                        (
                            entry_id, 
                            item.get('title'), 
                            item.get('name'), 
                            content_json, 
                            cat_id
                        )
                    )
                    
                    # 插入别名
                    aliases = item.get('aliases', [])
                    for alias in aliases:
                        await db.execute(
                            "INSERT OR IGNORE INTO aliases (entry_id, alias) VALUES (?, ?)",
                            (entry_id, alias)
                        )
                    count_general += 1
            
            log.info(f"YAML 导入完成: 成员 {count_member} 个, 设定 {count_general} 个。")
        except Exception as e:
            log.error(f"YAML Import Error: {e}", exc_info=True)

    async def _refresh_keyword_cache(self):
        """同时缓存 通用设定(general) 和 社区成员(member) 的关键词"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                self._keyword_cache = []
                
                # 1. 加载通用设定
                async with db.execute("SELECT id, title, name FROM general_knowledge WHERE status='approved'") as c:
                    async for row in c:
                        eid, title, name = row
                        if title: self._keyword_cache.append((title.lower(), eid, 'general'))
                        if name: self._keyword_cache.append((name.lower(), eid, 'general'))
                
                async with db.execute("SELECT entry_id, alias FROM aliases") as c:
                    async for row in c:
                        eid, alias = row
                        if alias: self._keyword_cache.append((alias.lower(), eid, 'general'))

                # 2. 加载社区成员
                async with db.execute("SELECT id, title FROM community_members WHERE status='approved'") as c:
                    async for row in c:
                        eid, title = row
                        # title 通常是 "社区成员：梦星"，我们可能只想匹配 "梦星"？
                        # 这里先全部缓存，依赖昵称表进行精确匹配
                        if title: self._keyword_cache.append((title.lower(), eid, 'member'))

                async with db.execute("SELECT member_id, nickname FROM member_discord_nicknames") as c:
                    async for row in c:
                        eid, nick = row
                        if nick: self._keyword_cache.append((nick.lower(), eid, 'member'))
            
            log.info(f"已缓存 {len(self._keyword_cache)} 个世界书索引 (含成员与设定)。")
        except Exception as e:
            log.error(f"Cache Refresh Error: {e}")

    async def search_entries_in_message(self, message: str) -> List[Dict[str, Any]]:
        """
        检查消息中是否包含关键词，并返回统一格式的条目
        """
        if not message or not self._keyword_cache:
            return []
            
        matched_items = set() # Set[Tuple(id, type)]
        message_lower = message.lower()
        
        # 1. 关键词匹配
        for keyword, entry_id, source_type in self._keyword_cache:
            if keyword in message_lower:
                matched_items.add((entry_id, source_type))
        
        if not matched_items:
            return []
            
        results = []
        try:
            async with aiosqlite.connect(self.db_path) as db:
                for entry_id, source_type in matched_items:
                    
                    if source_type == 'general':
                        async with db.execute(
                            "SELECT id, title, name, content_json FROM general_knowledge WHERE id = ?", 
                            (entry_id,)
                        ) as cursor:
                            row = await cursor.fetchone()
                            if row:
                                eid, title, name, content_str = row
                                try:
                                    content = json.loads(content_str)
                                    results.append({
                                        "id": eid,
                                        "type": "设定",
                                        "title": title or name,
                                        "content": content
                                    })
                                except: pass

                    elif source_type == 'member':
                        async with db.execute(
                            "SELECT id, title, content_json FROM community_members WHERE id = ?", 
                            (entry_id,)
                        ) as cursor:
                            row = await cursor.fetchone()
                            if row:
                                eid, title, content_str = row
                                try:
                                    content = json.loads(content_str)
                                    results.append({
                                        "id": eid,
                                        "type": "社区成员",
                                        "title": title,
                                        "content": content
                                    })
                                except: pass
        except Exception as e:
            log.error(f"Search Error: {e}")
            
        if results:
            log.info(f"触发世界书/成员条目: {[r['title'] for r in results]}")
        
        return results

world_book_db_manager = WorldBookDBManager(DB_PATH)