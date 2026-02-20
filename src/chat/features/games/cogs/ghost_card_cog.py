# src/chat/features/games/cogs/ghost_card_cog.py
# -*- coding: utf-8 -*-

import discord
from discord.ext import commands
from discord import app_commands
from discord.ui import View, Button, Modal, TextInput
import logging
import random
import asyncio
# 假设这是你的金币服务路径，请根据实际情况调整
from src.chat.features.odysseia_coin.service.coin_service import coin_service

log = logging.getLogger(__name__)

# --- 资源配置 ---
CASINO_ICON_URL = "https://cdn.discordapp.com/emojis/1471788864749305962.png" # 替换为你的图标
SUITS = ['♠️', '♥️', '♣️', '♦️']
RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
JOKER = "🃏"
BACK_CARD = "🎴"

EMOJI_URLS = {
    "happy": "https://cdn.discordapp.com/emojis/1471766600205205655.png",
    "sad": "https://cdn.discordapp.com/emojis/1471766631981121568.png",
    "cute": "https://cdn.discordapp.com/emojis/1471766428565639281.png",
    "angry": "https://cdn.discordapp.com/emojis/1471766571926945998.png",
    "invite": "https://cdn.discordapp.com/emojis/1471766484895137803.png",
    "tsundere": "https://cdn.discordapp.com/emojis/1471766459020480522.png",
    "snicker": "https://cdn.discordapp.com/emojis/1471930795412426843.png",
}

DIALOGUE_EMOTIONS = {
    "start": "invite",
    "ai_draw_joker": "sad",
    "ai_draw_safe": "happy",
    "ai_draw_safe_with_joker": "tsundere",
    "win": "tsundere",
    "lose": "sad",
    "confirm_proud": "snicker",
    "confirm_annoyed": "angry",
    "cancel_proud": "tsundere",
    "cancel_annoyed": "sad",
    "confirm_safe_idle": "cute",
    "selected_ghost_fake": "cute",
    "selected_safe_fake": "angry",
    "safe_idle": "cute"
}

DIFFICULTY_SETTINGS = {"LOW": 0.05, "MEDIUM": 0.25, "HIGH": 0.50, "SCHRODINGER": "DYNAMIC"}

DIALOGUES = {
    "start": ["哼，居然敢挑战我？做好输得哭鼻子的准备了吗？", "既然你这么想送死，那我就勉为其难陪你玩玩吧~", "杂鱼~ 杂鱼~ 准备好接受惩罚了吗？♥"],
    "ai_draw_joker": [
        "怎么会这样？！我算错了吗？",
        "不——！我的完美计划！",
        "呜...我居然抽到了这张牌...",
    ],
    "ai_draw_safe": ["嘿嘿，安全上垒！", "哦耶!抽到安全牌啦!", "哼哼,一切都在计划之中!"],
    "ai_draw_safe_with_joker": [
        "嘿嘿，好牌get！接下来轮到你了，快点把那张讨厌的牌拿走！",
        "呼... 喂，下一轮你可要好好选哦，有个大惊喜等着你呢~",
        "这张牌归我啦~ 至于那张鬼牌嘛... 它是为你准备的哦~",
        "不错不错~ 接下来就是见证你倒霉的时刻了！"
    ],
    "win": ["输了吧？杂鱼~ 快点承认原本就不如我吧！♥", "哼，赢得毫无悬念，真无聊。", "下次再练练吧，虽然练了也是输~"],
    "lose": ["怎... 怎么可能？！我居然输给了杂鱼？！", "这局不算！肯定是你作弊了！💢", "呜呜... 下次绝对不会输给你了！"],
    
    # === 结果反馈台词 ===
    "confirm_proud": [
        "太好了！你终于把它抽走了！",
        "嘿嘿，这张烫手的山芋现在是你的了！祝你好运哦～",
        "**计划通！**",
    ],
    "confirm_annoyed": [
        "什...什么？！你居然看穿了我的计谋！",
        "不可能！我的演技应该天衣无缝才对！",
        "呃...好吧，算你厉害...",
        "呜...被你看穿了...",
        "失败了...",
        "哼，别得意！",
    ],
    "cancel_proud": [
        "改变主意了？太好了。",
        "嗯，谨慎一点总是好的。",
        "好的，你决定不抽这张了。",
    ],
    "cancel_annoyed": [
        "哎，怎么不抽了？就差一点了...",
        "真可惜...",
        "改变主意了吗？再考虑一下嘛。",
    ],
    "confirm_safe_idle": [
        "拿走吧，反正对我没影响。",
        "切，随便你抽。",
        "哦，那张啊，送你了。",
        "拿去拿去，反正不是鬼牌。"
    ],

    # === 欺诈/反应台词库 ===
    "selected_ghost_fake": [
        "嘿嘿!对,就抽这个!",
        "嘿嘿，这个“好运”现在是你的了！",
        "**计划通！**",
        "快抽吧快抽吧！那张牌对我来说是个大麻烦！",
        "相信我，这张牌绝对不是你想看到的那张！",
        "嘿嘿，别犹豫了，就是它！",
    ],
    "selected_safe_fake": [
        "不！别抽这个...！",
        "呜呜呜...不许拿这张啦...",
        "别...别抽那张！",
        "可恶,不许抽这张!",
        "这张不可以啦!",
    ],
    
    # === 新增：摆烂台词 (当AI没鬼牌时) ===
    "safe_idle": ["哼，反正我手里没有鬼牌，随便你拿。", "快点抽啦，磨磨蹭蹭的...", "看上哪张了？拿走拿走！", "这张？给你给你，反正对我没影响。"]
}

SHAME_TEMPLATES = [
    "{user} 小杂鱼在赌场输掉了 **{bet}** 类脑币~♥",
    "大家快来看呀！{user} 又来送钱啦，输掉了 **{bet}** 类脑币！",
    "噗... {user} 居然输了 **{bet}**，真是不自量力呢~",
    "感谢 {user} 赞助的 **{bet}** 类脑币，今晚可以加餐了~",
    "悲报：{user} 的 **{bet}** 类脑币瞬间蒸发了！好惨哦~",
    "赢我的概率和 {user} 找到对象的概率一样低呢~ (输掉 **{bet}**)",
    "哎呀，{user} 你的钱包还好吗？刚少了 **{bet}** 块呢~",
    "是哪位小笨蛋在赌场里输掉了**{bet}**类脑币呀？"
]

class GhostGameSession:
    def __init__(self, user_id, bet):
        self.user_id = user_id
        self.bet = bet
        self.player_hand = []
        self.ai_hand = []
        self.turn = "player"
        self.attempts_left = 3 # 每回合剩余选择次数
        self.selected_card_index = -1
        self.is_deceiving = False
        self.deception_type = None
        self.card_strategies = []

        # 初始化牌组
        deck = [f"{suit}{rank}" for suit in SUITS for rank in RANKS] + [JOKER]
        random.shuffle(deck)
        
        # [修复] 随机发牌顺序，消除因牌数奇数(53张)导致的概率偏差(51% vs 49%)
        hands = [self.player_hand, self.ai_hand]; random.shuffle(hands)
        while deck:
            if deck: hands[0].append(deck.pop())
            if deck: hands[1].append(deck.pop())
        self.remove_pairs(self.player_hand)
        self.remove_pairs(self.ai_hand)
        self.generate_strategies()
        self.turn = "player" if len(self.player_hand) > len(self.ai_hand) else "ai"
        self.attempts_left = min(3, len(self.ai_hand))

    @property
    def deception_prob(self):
        return random.choice([0.25, 0.50, 0.75])

    def generate_strategies(self):
        self.card_strategies = []
        prob = self.deception_prob
        for _ in range(len(self.ai_hand)):
            self.card_strategies.append(random.random() < prob)

    def remove_pairs(self, hand):
        new_hand = []
        counts = {}
        for card in hand:
            rank = "JOKER" if card == JOKER else card.replace('♠️','').replace('♥️','').replace('♣️','').replace('♦️','')
            if rank not in counts: counts[rank] = []
            counts[rank].append(card)
        for rank, cards in counts.items():
            while len(cards) >= 2: cards.pop(); cards.pop()
            new_hand.extend(cards)
        hand.clear(); hand.extend(new_hand); random.shuffle(hand)

class GhostBetModal(Modal, title="输入下注金额"):
    amount = TextInput(label="金额", placeholder="请输入正整数...", min_length=1)
    def __init__(self, view):
        super().__init__()
        self.view = view
    async def on_submit(self, interaction: discord.Interaction):
        try:
            bet_amount = int(self.amount.value)
            await self.view.process_bet(interaction, bet_amount)
        except ValueError:
            await interaction.response.send_message("请输入有效的数字！", ephemeral=True)

class GhostBettingView(View):
    def __init__(self, cog, user_id, balance, disp_msg, mode="integrated"):
        super().__init__(timeout=60)
        self.cog = cog
        self.user_id = user_id
        self.balance = balance
        self.disp_msg = disp_msg
        self.mode = mode
        
        self.small_bet = 100
        self.mid_bet = max(1, int(balance * 0.25))
        self.big_bet = max(1, int(balance * 0.5))
        
        self.add_item(self.create_button(f"小 ({self.small_bet})", self.small_bet, discord.ButtonStyle.secondary))
        self.add_item(self.create_button(f"中 ({self.mid_bet})", self.mid_bet, discord.ButtonStyle.primary))
        self.add_item(self.create_button(f"大 ({self.big_bet})", self.big_bet, discord.ButtonStyle.success))
        self.add_item(self.create_button(f"梭哈 ({self.balance})", self.balance, discord.ButtonStyle.danger))
        
        custom_btn = Button(label="自定义", style=discord.ButtonStyle.secondary, emoji="✏️")
        custom_btn.callback = self.custom_bet
        self.add_item(custom_btn)

    def create_button(self, label, amount, style):
        btn = Button(label=label, style=style)
        async def callback(it: discord.Interaction):
            await self.process_bet(it, amount)
        btn.callback = callback
        return btn

    async def custom_bet(self, it: discord.Interaction):
        await it.response.send_modal(GhostBetModal(self))

    async def process_bet(self, it: discord.Interaction, amount: int):
        if amount <= 0: return await it.response.send_message("金额必须大于0", ephemeral=True)
        if await coin_service.get_balance(self.user_id) < amount:
            return await it.response.send_message("余额不足！", ephemeral=True)
        
        await coin_service.remove_coins(self.user_id, amount, "鬼牌赌注")
        
        game = GhostGameSession(self.user_id, amount)
        controller = GameControlView(self.cog, game, self.disp_msg, mode=self.mode)
        
        key = "start"
        emotion = DIALOGUE_EMOTIONS.get(key)
        
        # 初始更新：如果是合体模式，update_display 会直接挂载 controller
        await self.cog.update_display(self.disp_msg, game, dialogue=random.choice(DIALOGUES[key]), emotion=emotion, mode=self.mode, controller_view=controller)
        
        if self.mode == "separated":
            await it.response.edit_message(content="🎮 **操控手柄** (仅你可见)", view=controller)
            controller.controller_msg = await it.original_response()
        else:
            # 合体模式：下注视图被替换为游戏视图，直接 defer 即可
            await it.response.defer()
            controller.controller_msg = self.disp_msg
        
        if game.turn == "ai":
             asyncio.create_task(self.cog.handle_ai_turn(self.disp_msg, game, controller))

class GhostMainView(View):
    def __init__(self, cog, game, display_msg):
        super().__init__(timeout=3600)
        self.cog = cog
        self.game = game
        self.display_msg = display_msg

    @discord.ui.button(label="🎮 恢复手柄", style=discord.ButtonStyle.primary, custom_id="ghost_refresh")
    async def refresh_btn(self, it: discord.Interaction, btn: discord.ui.Button):
        if it.user.id != self.game.user_id:
            return await it.response.send_message("这不是你的牌局！", ephemeral=True)
        
        view = GameControlView(self.cog, self.game, self.display_msg, mode="separated")
        await it.response.send_message("🎮 **操控手柄** (已刷新)", view=view, ephemeral=True)
        view.controller_msg = await it.original_response()

class GhostReplayView(View):
    def __init__(self, cog, user_id, display_msg, mode="integrated", shame_msg=None):
        super().__init__(timeout=600)
        self.cog = cog
        self.user_id = user_id
        self.display_msg = display_msg
        self.mode = mode
        self.shame_msg = shame_msg
        
        # 动态添加按钮
        self.add_item(self.replay_button())
        if self.mode == "separated":
            self.add_item(self.delete_button())
        if self.shame_msg:
            self.add_item(self.shame_button())

    def replay_button(self):
        btn = Button(label="🔄 再来一局", style=discord.ButtonStyle.success)
        async def callback(it: discord.Interaction):
            if it.user.id != self.user_id: return await it.response.send_message("这不是你的牌局！", ephemeral=True)
            bal = await coin_service.get_balance(self.user_id)
            if bal <= 0: return await it.response.send_message("你已经破产了！快去打工或者签到吧！", ephemeral=True)
            
            if self.mode == "separated":
                try: await self.display_msg.delete()
                except: pass
                embed = discord.Embed(title="🃏 准备对局", description="请选择下注金额...", color=discord.Color.purple())
                await it.response.send_message(embed=embed)
                new_msg = await it.original_response()
                view = GhostBettingView(self.cog, self.user_id, bal, new_msg, mode="separated")
                await it.followup.send("💵 **下注台**", view=view, ephemeral=True)
            else:
                embed = discord.Embed(title="🃏 准备对局", description="请选择下注金额...", color=discord.Color.purple())
                view = GhostBettingView(self.cog, self.user_id, bal, self.display_msg, mode="integrated")
                await it.response.edit_message(embed=embed, view=view)
        btn.callback = callback
        return btn

    def delete_button(self):
        btn = Button(label="🗑️ 删除", style=discord.ButtonStyle.danger)
        async def callback(it: discord.Interaction):
            if it.user.id != self.user_id: return await it.response.send_message("这不是你的牌局！", ephemeral=True)
            try: await it.message.delete()
            except: await it.response.send_message("删除失败。", ephemeral=True)
        btn.callback = callback
        return btn

    def shame_button(self):
        btn = Button(label="😈我是杂鱼 (-100币)", style=discord.ButtonStyle.primary)
        async def callback(it: discord.Interaction):
            if it.user.id != self.user_id: return await it.response.send_message("这不是你的牌局！", ephemeral=True)
            
            bal = await coin_service.get_balance(self.user_id)
            if bal < 100: return await it.response.send_message("余额不足 100 类脑币，无法删除黑历史！", ephemeral=True)
            
            await coin_service.remove_coins(self.user_id, 100, "删除嘲讽消息")
            try:
                await self.shame_msg.delete()
            except discord.NotFound:
                pass # 已经被删除了，算成功
            except Exception:
                await coin_service.add_coins(self.user_id, 100, "删除嘲讽消息失败退款")
                return await it.response.send_message("删除失败，可能已经被删除了。", ephemeral=True)

            self.shame_msg = None
            self.remove_item(btn)
            try:
                if self.mode == "separated": await self.display_msg.edit(view=self)
                else: await it.message.edit(view=self)
            except: pass
            await it.response.send_message("黑历史已删除... 呜呜...", ephemeral=True)
        btn.callback = callback
        return btn

class GameControlView(View):
    def __init__(self, cog, game, display_msg, mode="integrated"):
        super().__init__(timeout=600)
        self.cog = cog
        self.game = game
        self.display_msg = display_msg
        self.mode = mode
        self.controller_msg = None
        self.setup_buttons()

    def setup_buttons(self):
        self.clear_items()
        
        if self.game.turn == "ai":
            # AI 回合禁用所有按钮
            self.add_item(Button(label="*类脑娘正在小心翼翼地抽牌... *", style=discord.ButtonStyle.success, disabled=True))
            return

        # 玩家回合：如果还未选牌
        if self.game.selected_card_index == -1:
            for i in range(len(self.game.ai_hand)):
                # 使用不同的颜色增加趣味性，或者全部灰色保持神秘
                btn = Button(label=f"{i+1}", emoji=BACK_CARD, style=discord.ButtonStyle.secondary, custom_id=f"select_{i}")
                btn.callback = self.make_callback(i)
                self.add_item(btn)
        else:
            # 玩家回合：已选牌，等待确认
            # 这里的显示文本至关重要，它代表了 AI 的反应
            self.add_item(Button(label=f"你摸到了第 {self.game.selected_card_index + 1} 张...", style=discord.ButtonStyle.primary, disabled=True))
            
            confirm_btn = Button(label="🔥 就要这张！", style=discord.ButtonStyle.success)
            confirm_btn.callback = self.confirm_cb
            self.add_item(confirm_btn)
            
            # 只有当剩余次数大于1时才显示取消按钮 (最后一次必须确认)
            if self.game.attempts_left > 1:
                cancel_btn = Button(label="🤔 再想想...", style=discord.ButtonStyle.danger)
                cancel_btn.callback = self.cancel_cb
                self.add_item(cancel_btn)
            
            self.add_item(Button(label=f"剩余次数: {self.game.attempts_left}", style=discord.ButtonStyle.secondary, disabled=True))

    def make_callback(self, i):
        async def cb(it: discord.Interaction):
            if self.game.turn == "ai":
                await it.response.send_message("别急！现在不是你的回合。", ephemeral=True)
                return

            self.game.selected_card_index = i
            target = self.game.ai_hand[i]
            ai_has_joker = JOKER in self.game.ai_hand
            
            self.game.is_deceiving = False
            self.game.deception_type = None
            
            # === AI 反应逻辑 (博弈核心) ===
            key = "safe_idle"
            if ai_has_joker:
                # 薛定谔的难度计算
                should_deceive = self.game.card_strategies[i]
                
                if should_deceive:
                    self.game.is_deceiving = True
                    if target == JOKER:
                        # 选中鬼牌 -> 伪装成安全 (诱导)
                        self.game.deception_type = "ghost_fake"
                        key = "selected_ghost_fake"
                    else:
                        # 选中安全牌 -> 伪装成鬼牌 (恐吓)
                        self.game.deception_type = "safe_fake"
                        key = "selected_safe_fake"
                else:
                    # 不欺骗，说真话
                    if target == JOKER:
                        # 选中鬼牌 -> 真的惊慌
                        key = "selected_safe_fake" # 复用恐吓台词表示惊慌
                    else:
                        # 选中安全牌 -> 真的很平淡
                        key = "selected_ghost_fake" # 复用诱导台词表示平淡
            
            dlg = random.choice(DIALOGUES[key])
            emotion = DIALOGUE_EMOTIONS.get(key)

            # 先更新按钮状态 (变为确认/取消)
            self.setup_buttons()
            # 更新大屏对话，并刷新手柄出确认按钮
            await self.cog.update_display(self.display_msg, self.game, dialogue=dlg, emotion=emotion, mode=self.mode, controller_view=self)
            
            if self.mode == "separated":
                await it.response.edit_message(view=self)
            else:
                await it.response.defer()
            
        return cb

    async def confirm_cb(self, it: discord.Interaction):
        if self.game.turn == "ai": return

        # 检查 AI 是否持有鬼牌 (用于判断是否触发懊恼台词)
        ai_has_joker = JOKER in self.game.ai_hand

        # 执行抽牌
        card = self.game.ai_hand.pop(self.game.selected_card_index)
        self.game.player_hand.append(card)
        self.game.selected_card_index = -1
        
        # 结算台词
        key = ""
        if card == JOKER:
            # 场景 6, 7: 用户拿走鬼牌 -> AI 得意
            key = "confirm_proud"
        else:
            # 场景 5, 8: 用户拿走安全牌
            if ai_has_joker:
                # AI 有鬼牌 -> 懊恼 (计谋被识破)
                key = "confirm_annoyed"
            else:
                # AI 无鬼牌 -> 摆烂 (反正全是安全牌)
                key = "confirm_safe_idle"

        dlg = random.choice(DIALOGUES[key])
        emotion = DIALOGUE_EMOTIONS.get(key)
        self.game.remove_pairs(self.game.player_hand)
        self.game.turn = "ai"
        
        # 1. 先更新按钮状态 (变为 AI 思考中/禁用)
        self.setup_buttons()
        
        # 2. 更新大屏显示 (带上新的禁用按钮视图)
        await self.cog.update_display(self.display_msg, self.game, dialogue=dlg, log_msg=f"你抽到了 **{card}**！", emotion=emotion, mode=self.mode, controller_view=self)
        
        # 3. 处理交互响应
        if self.mode == "separated":
            await it.response.edit_message(view=self)
            self.controller_msg = await it.original_response()
        else:
            await it.response.defer()

        # 胜负判定
        if not self.game.player_hand:
            await self.end(it, "player")
            return
        if not self.game.ai_hand:
            await self.end(it, "ai")
            return

        # 异步启动 AI 回合
        asyncio.create_task(self.cog.handle_ai_turn(self.display_msg, self.game, self))

    async def cancel_cb(self, it: discord.Interaction):
        # 减少尝试次数
        self.game.attempts_left -= 1
        
        # 判断逻辑：
        # 场景 4: 不是鬼牌 + 没欺骗(说真话"是安全") + 放弃 -> AI 得意 (笑你胆小)
        # 其他场景 (1, 2, 3): 都是 AI 懊恼 (要么骗失败了，要么真话没被信导致鬼牌没送出去)
        target_card = self.game.ai_hand[self.game.selected_card_index]
        is_joker = target_card == JOKER
        
        key = ""
        if not is_joker and not self.game.is_deceiving:
            key = "cancel_proud"
        else:
            key = "cancel_annoyed"
            
        dlg = random.choice(DIALOGUES[key])
        emotion = DIALOGUE_EMOTIONS.get(key)
        self.game.selected_card_index = -1
        
        # 1. 先更新按钮 (恢复为选牌状态)
        self.setup_buttons()
        # 2. 更新显示
        await self.cog.update_display(self.display_msg, self.game, dialogue=dlg, emotion=emotion, mode=self.mode, controller_view=self)
        
        if self.mode == "separated":
            await it.response.edit_message(view=self)
        else:
            await it.response.defer()

    async def end(self, it, winner):
        self.clear_items()
        if winner == "player":
            win = int(self.game.bet * 2)
            key = "lose" # AI lose
            await coin_service.add_coins(self.game.user_id, win, "鬼牌获胜")
            res = f"太好啦！**你赢了！** 类脑娘抽到了鬼牌呢～ (获得 {win} 类脑币)"
            shame_msg = None
        else:
            key = "win" # AI win
            res = f"啊呀，**类脑娘赢了！** 你抽到了鬼牌... (失去了 {self.game.bet} 类脑币)"
            
            # 发送嘲讽消息
            shame_msg = None
            try:
                if self.display_msg.channel:
                    template = random.choice(SHAME_TEMPLATES)
                    txt = template.format(user=f"<@{self.game.user_id}>", bet=self.game.bet)
                    shame_msg = await self.display_msg.channel.send(txt)
            except: pass
        
        dlg = random.choice(DIALOGUES[key])
        emotion = DIALOGUE_EMOTIONS.get(key)
        
        # 更新主显示 (挂载 ReplayView)
        await self.cog.update_display(self.display_msg, self.game, dialogue=dlg, log_msg=res, over=True, emotion=emotion, mode=self.mode, shame_msg=shame_msg)
        
        # 仅在分离模式下清除手柄消息，合体模式下主显示就是手柄，不能清除 view
        if self.mode == "separated":
            try:
                if not it.response.is_done():
                    await it.response.edit_message(content=f"🏁 对局结束\n{res}", view=None)
                else:
                    await it.edit_original_response(content=f"🏁 对局结束\n{res}", view=None)
            except: pass
            
        self.stop()

class GhostCardCog(commands.Cog):
    def __init__(self, bot): self.bot = bot

    async def update_display(self, msg, game, dialogue="...", log_msg="", over=False, emotion=None, mode="integrated", controller_view=None, shame_msg=None):
        # 根据局势改变颜色：AI 有鬼牌(紫色/危险)，AI 无鬼牌(绿色/安全)，结束(金色)
        color = discord.Color.purple()
        if over: 
            color = discord.Color.gold() if not game.player_hand else discord.Color.dark_grey()
        elif JOKER in game.player_hand:
            color = discord.Color.green() # 玩家持有鬼牌，相对安全（去抽牌进货）

        # 提示鬼牌位置
        hint = ""
        if not over: hint = " (鬼牌在你手里哦)" if JOKER in game.player_hand else " (鬼牌在我手里哦)"

        embed = discord.Embed(color=color)
        embed.set_author(name="🃏 类脑娘的抽鬼牌对决", icon_url=CASINO_ICON_URL)
        
        # 设置表情图片 (Thumbnail 在桌面端显示在右上角，移动端显示在上方)
        if emotion and emotion in EMOJI_URLS:
            embed.set_thumbnail(url=EMOJI_URLS[emotion])

        # 增加台词的排版
        embed.description = f"🗣️ **类脑娘**{hint}\n# 「 {dialogue} 」\n━━━━━━━━━━━━━━━━━━━━━━"
        
        # 手牌显示优化
        ai_hand_display = " ".join([f"`{c}`" for c in game.ai_hand]) if over else f"{BACK_CARD} " * len(game.ai_hand)
        p_hand_display = " ".join([f"`{c}`" for c in game.player_hand]) if game.player_hand else "*(已清空)*"
        
        embed.add_field(name=f"🤖 她的手牌: {len(game.ai_hand)}", value=ai_hand_display or "(空)", inline=False)
        embed.add_field(name=f"👤 你的手牌: {len(game.player_hand)}", value=p_hand_display, inline=False)
        
        if log_msg:
            embed.add_field(name="📜 战况", value=log_msg, inline=False)
        
        state_str = "🏁 游戏结束" if over else ("⏳ 思考中..." if game.turn == 'ai' else "👉 轮到你了")
        embed.set_footer(text=f"状态: {state_str}")
        
        # 视图选择逻辑
        if over:
            view = GhostReplayView(self, game.user_id, msg, mode=mode, shame_msg=shame_msg)
        elif mode == "integrated":
            view = controller_view # 合体模式：直接挂载手柄
        else:
            view = GhostMainView(self, game, msg) # 分离模式：挂载恢复按钮
            
        await msg.edit(embed=embed, view=view)

    async def handle_ai_turn(self, msg, game, view):
        # 思考时间，增加紧张感
        await asyncio.sleep(random.uniform(2.0, 3.5))
        
        # AI 简单策略：随机抽
        idx = random.randint(0, len(game.player_hand)-1)
        card = game.player_hand.pop(idx)
        game.ai_hand.append(card)
        
        # 简单的反应
        key = ""
        if card == JOKER:
            key = "ai_draw_joker"
        else:
            if JOKER in game.ai_hand:
                key = "ai_draw_safe_with_joker"
            else:
                key = "ai_draw_safe"
        
        reaction = random.choice(DIALOGUES[key])
        emotion = DIALOGUE_EMOTIONS.get(key)
        game.remove_pairs(game.ai_hand)
        game.generate_strategies()
        
        # AI 胜利 (手牌清空)
        if not game.ai_hand:
            key_win = "win"
            
            # 发送嘲讽消息
            shame_msg = None
            try:
                if msg.channel:
                    template = random.choice(SHAME_TEMPLATES)
                    txt = template.format(user=f"<@{game.user_id}>", bet=game.bet)
                    shame_msg = await msg.channel.send(txt)
            except: pass

            await self.update_display(msg, game, dialogue=random.choice(DIALOGUES[key_win]), log_msg="啊呀，**类脑娘赢了！** 你抽到了鬼牌...", over=True, emotion=DIALOGUE_EMOTIONS.get(key_win), mode=view.mode, shame_msg=shame_msg)
            view.stop()
            if view.mode == "separated" and view.controller_msg:
                try: await view.controller_msg.edit(content="🏁 对局已结束", view=None)
                except: pass
            return
        
        # 玩家胜利 (被抽光)
        if not game.player_hand:
            win = int(game.bet * 2)
            await coin_service.add_coins(game.user_id, win, "鬼牌获胜")
            res = f"🎉 胜利！牌被抽光了！获得 {win} 币"
            key_lose = "lose"
            dlg = random.choice(DIALOGUES[key_lose])
            await self.update_display(msg, game, dialogue=dlg, log_msg=res, over=True, emotion=DIALOGUE_EMOTIONS.get(key_lose), mode=view.mode)
            view.stop()
            if view.mode == "separated" and view.controller_msg:
                try: await view.controller_msg.edit(content="🏁 对局已结束", view=None)
                except: pass
            return

        game.turn = "player"
        game.attempts_left = min(3, len(game.ai_hand)) # 重置玩家选择次数
        
        # 刷新玩家控制面板
        view.setup_buttons()
        
        # 更新显示 (如果是合体模式，这里会把 view 挂上去)
        await self.update_display(msg, game, dialogue=reaction, log_msg=f"类脑娘抽到了: {card}", emotion=emotion, mode=view.mode, controller_view=view)
        
        if view.mode == "separated":
            if view.controller_msg:
                try:
                    await view.controller_msg.edit(content="又轮到你了！加油哦～", view=view)
                except discord.NotFound:
                    pass 
                except Exception as e:
                    log.error(f"Controller update failed: {e}")

    @app_commands.command(name="抽鬼牌", description="和类脑娘玩抽鬼牌！")
    @app_commands.describe(mode="游戏模式 (默认: 合体模式)")
    @app_commands.choices(mode=[
        app_commands.Choice(name="合体模式 (仅自己可见)", value="integrated"),
        app_commands.Choice(name="分离模式 (公开战况+私密手柄)", value="separated")
    ])
    async def joker(self, it: discord.Interaction, mode: app_commands.Choice[str] = None):
        bal = await coin_service.get_balance(it.user.id)
        if bal <= 0: return await it.response.send_message("余额不足...", ephemeral=True)
        
        mode_val = mode.value if mode else "integrated"
        embed = discord.Embed(title="🃏 准备对局", description="请选择下注金额...", color=discord.Color.purple())
        
        # 合体模式使用 ephemeral，分离模式公开
        await it.response.send_message(embed=embed, ephemeral=(mode_val == "integrated"))
        msg = await it.original_response()
        
        view = GhostBettingView(self, it.user.id, bal, msg, mode=mode_val)
        
        if mode_val == "separated":
            await it.followup.send("💵 **下注台**", view=view, ephemeral=True)
        else:
            # 合体模式直接在原消息上挂载下注视图
            await it.edit_original_response(view=view)

async def setup(bot):
    await bot.add_cog(GhostCardCog(bot))