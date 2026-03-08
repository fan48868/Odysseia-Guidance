# src/chat/features/world_book/database/world_book_db_manager.py

import aiosqlite
import logging
from pathlib import Path

# 设置日志记录器
log = logging.getLogger(__name__)

# 定义数据库文件的路径
DB_PATH = Path("data/world_book.sqlite3")

# 定义数据库的完整结构 (Schema) - 从 initialize_world_book_db.py 合并而来
# 使用 CREATE TABLE IF NOT EXISTS 来确保数据安全
SQL_SCHEMA = """
-- 类别表
CREATE TABLE IF NOT EXISTS "categories" (
    "id"    INTEGER PRIMARY KEY AUTOINCREMENT,
    "name"  TEXT NOT NULL UNIQUE
);

-- 社区成员档案表
CREATE TABLE IF NOT EXISTS "community_members" (
    "id"    TEXT PRIMARY KEY,
    "title" TEXT,
    "discord_id" TEXT,
    "discord_number_id" TEXT,
    "history" TEXT,
    "content_json" TEXT,
    "status" TEXT DEFAULT 'approved'
);

-- 成员 Discord 昵称表 (一对多)
CREATE TABLE IF NOT EXISTS "member_discord_nicknames" (
    "id" INTEGER PRIMARY KEY AUTOINCREMENT,
    "member_id" TEXT NOT NULL,
    "nickname" TEXT NOT NULL,
    FOREIGN KEY ("member_id") REFERENCES "community_members"("id")
);

-- 通用知识条目表
CREATE TABLE IF NOT EXISTS "general_knowledge" (
    "id" TEXT PRIMARY KEY,
    "title" TEXT,
    "name" TEXT,
    "content_json" TEXT,
    "category_id" INTEGER,
    "contributor_id" INTEGER,
    "created_at" TEXT,
    "status" TEXT DEFAULT 'approved',
    FOREIGN KEY ("category_id") REFERENCES "categories"("id")
);

-- 别名表 (通用知识的一对多)
CREATE TABLE IF NOT EXISTS "aliases" (
    "id" INTEGER PRIMARY KEY AUTOINCREMENT,
    "entry_id" TEXT NOT NULL,
    "alias" TEXT NOT NULL,
    FOREIGN KEY ("entry_id") REFERENCES "general_knowledge"("id")
);

-- 引用关系表 (通用知识的一对多)
CREATE TABLE IF NOT EXISTS "knowledge_refers_to" (
    "id" INTEGER PRIMARY KEY AUTOINCREMENT,
    "entry_id" TEXT NOT NULL,
    "reference" TEXT NOT NULL,
    FOREIGN KEY ("entry_id") REFERENCES "general_knowledge"("id")
);

-- 待审核条目表
CREATE TABLE IF NOT EXISTS "pending_entries" (
    "id"    INTEGER PRIMARY KEY AUTOINCREMENT,
    "entry_type"    TEXT NOT NULL,
    "data_json" TEXT NOT NULL,
    "message_id"    INTEGER,
    "channel_id"    INTEGER NOT NULL,
    "guild_id"  INTEGER NOT NULL,
    "proposer_id"   INTEGER NOT NULL,
    "status"    TEXT NOT NULL DEFAULT 'pending',
    "created_at" DATETIME DEFAULT CURRENT_TIMESTAMP,
    "expires_at"    TEXT NOT NULL
);

-- 为常用查询字段创建索引
CREATE INDEX IF NOT EXISTS "idx_community_members_discord_id" ON "community_members" ("discord_number_id");
CREATE INDEX IF NOT EXISTS "idx_pending_entries_status_expires" ON "pending_entries" ("status", "expires_at");
"""

class WorldBookDBManager:
    """
    管理 world_book.sqlite3 数据库的初始化和连接。
    """
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def init_async(self):
        """
        在机器人启动时安全地初始化数据库。
        """
        log.info(f"正在检查并初始化 World Book 数据库: {self.db_path}...")
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.executescript(SQL_SCHEMA)
                await db.commit()
            log.info(f"World Book 数据库 '{self.db_path}' 检查完毕，所有表结构已确保存在。")
        except Exception as e:
            log.error(f"初始化 World Book 数据库时发生严重错误: {e}", exc_info=True)
            raise

# 创建一个单例，方便在项目其他地方导入和使用
world_book_db_manager = WorldBookDBManager(DB_PATH)