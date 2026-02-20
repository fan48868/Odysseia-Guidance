# src/chat/features/games/cogs/blackjack_cog.py
# -*- coding: utf-8 -*-

import discord
from discord.ext import commands
from discord import app_commands
from discord.ui import Modal, TextInput, Button
import logging
import random
from src.chat.features.odysseia_coin.service.coin_service import coin_service

log = logging.getLogger(__name__)

# --- 资源配置 ---
CASINO_ICON_URL = "https://cdn.discordapp.com/emojis/1471788864749305962.png"

EMOTIONS = {
    "default": "<:ao_jiao:1471766459020480522>",
    "happy": "<:kai_xin:1471766600205205655>",
    "shock": "<:hao_qi:1471766512858697748>",
    "angry": "<:sheng_qi:1471766571926945998>",
    "smug": "<:pu_ke:1471788864749305962>",
    "pity": "<:tan_qi:1471766292301086826>",
    "love": "<:zhu_fu:1471766393945849929>"
}

DIALOGUES = {
    "welcome": ["哟，带钱来了吗？别输光了哭鼻子哦~♥", "今天想输多少？我随时奉陪。", "又是你？哼，这次可不会让你轻易溜走。"],
    "hit": ["还要？小心爆掉哦~ 杂鱼~", "真贪心呢...", "好呀，这就给你一张~"],
    "stand": ["这就怕了？那轮到我了！", "哦？对自己很有信心嘛。", "看我不瞬间秒杀你！"],
    "player_win": ["切... 居然让你赢了，真不爽！💢", "下次... 下次绝对不会让你赢了！", "仅仅是运气好而已！别得意！"],
    "dealer_win": ["嘻嘻~ 赢你这种杂鱼真是太轻松了~♥", "哎呀，你的钱归我啦~ 谢谢款待~", "啧啧啧，这就是实力的差距呀~"],
    "tie": ["切，真没劲，居然平手。", "算你运气好，这次没输。", "哼，平局而已，再来！"],
    "bust": ["噗... 爆牌了？真是笨蛋呢~♥", "哈哈哈！贪心不足蛇吞象哦~", "哎呀呀，直接自爆了，笑死我了~"]
}

SUITS = ['♠️', '♥️', '♣️', '♦️']
RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
VALUES = {'A': 11, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 10, 'Q': 10, 'K': 10}

class Deck:
    def __init__(self):
        self.cards = [(rank, suit) for suit in SUITS for rank in RANKS]
        random.shuffle(self.cards)

    def draw(self):
        return self.cards.pop()

def calculate_score(hand):
    score = sum(VALUES[card[0]] for card in hand)
    num_aces = sum(1 for card in hand if card[0] == 'A')
    while score > 21 and num_aces:
        score -= 10
        num_aces -= 1
    return score

def format_hand(hand, hide_second=False):
    if hide_second:
        return f"`{hand[0][0]:>2}{hand[0][1]}` ` ?? ` "
    return " ".join([f"`{card[0]:>2}{card[1]}`" for card in hand])

class CustomBetModal(Modal, title="输入下注金额"):
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

# --- 下注阶段视图 ---
class BettingView(discord.ui.View):
    def __init__(self, bot, user_id, balance):
        super().__init__(timeout=60)
        self.bot = bot
        self.user_id = user_id
        self.balance = balance
        
        self.small_bet = 100
        self.mid_bet = max(1, int(balance * 0.25))
        self.big_bet = max(1, int(balance * 0.5))
        
        self.children[0].label = f"小 ({self.small_bet})"
        self.children[1].label = f"中 ({self.mid_bet})"
        self.children[2].label = f"大 ({self.big_bet})"
        self.children[3].label = f"梭哈 ({self.balance})"

    async def process_bet(self, interaction: discord.Interaction, amount: int):
        if amount <= 0:
            return await interaction.response.send_message("赌注必须大于0！", ephemeral=True)
        
        current_balance = await coin_service.get_balance(self.user_id)
        if current_balance < amount:
            return await interaction.response.send_message(f"没钱还想玩？你的余额不够支付 {amount}！", ephemeral=True)
            
        success = await coin_service.remove_coins(self.user_id, amount, "Blackjack 赌注")
        if success is None:
            return await interaction.response.send_message("扣款失败，请稍后再试。", ephemeral=True)

        game_view = BlackjackView(self.bot, self.user_id, amount)
        await game_view.start_game(interaction)

    @discord.ui.button(style=discord.ButtonStyle.secondary)
    async def bet_small(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id: return
        await self.process_bet(interaction, self.small_bet)

    @discord.ui.button(style=discord.ButtonStyle.primary)
    async def bet_mid(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id: return
        await self.process_bet(interaction, self.mid_bet)

    @discord.ui.button(style=discord.ButtonStyle.success)
    async def bet_big(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id: return
        await self.process_bet(interaction, self.big_bet)

    @discord.ui.button(label="梭哈", style=discord.ButtonStyle.danger, emoji="🔥")
    async def bet_all(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id: return
        await self.process_bet(interaction, self.balance)

    @discord.ui.button(label="自定义", style=discord.ButtonStyle.secondary, emoji="✏️")
    async def bet_custom(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id: return
        await interaction.response.send_modal(CustomBetModal(self))

# --- 游戏阶段视图 ---
class BlackjackView(discord.ui.View):
    def __init__(self, bot, user_id: int, bet: int):
        super().__init__(timeout=120)
        self.bot = bot
        self.user_id = user_id
        self.bet = bet
        self.deck = Deck()
        self.player_hand = [self.deck.draw(), self.deck.draw()]
        self.dealer_hand = [self.deck.draw(), self.deck.draw()]
        self.game_over = False
        self.current_dialogue = random.choice(DIALOGUES["welcome"])
        self.current_emotion = EMOTIONS["default"]
        self.shame_message = None

    async def start_game(self, interaction: discord.Interaction):
        player_score = calculate_score(self.player_hand)
        dealer_score = calculate_score(self.dealer_hand)
        
        if player_score == 21:
            if dealer_score == 21:
                await self.end_game(interaction, "双双 BlackJack！平局！", 0, initial=True)
            else:
                await self.end_game(interaction, "✨ BlackJack！天胡！", 1.5, initial=True)
        else:
            await self.update_message(interaction, initial=True)

    async def update_message(self, interaction: discord.Interaction, result_type=None, extra_msg="", color=discord.Color.from_rgb(46, 204, 113), initial=False):
        player_score = calculate_score(self.player_hand)
        
        if result_type == "win":
            self.current_emotion = EMOTIONS["angry"]
            self.current_dialogue = random.choice(DIALOGUES["player_win"])
        elif result_type == "lose":
            self.current_emotion = EMOTIONS["happy"]
            self.current_dialogue = random.choice(DIALOGUES["dealer_win"])
            if "爆牌" in extra_msg:
                 self.current_emotion = EMOTIONS["smug"]
                 self.current_dialogue = random.choice(DIALOGUES["bust"])
        elif result_type == "tie":
            self.current_emotion = EMOTIONS["pity"]
            self.current_dialogue = random.choice(DIALOGUES["tie"])

        embed = discord.Embed(color=color)
        embed.set_author(name="Blackjack 赌场", icon_url=CASINO_ICON_URL)
        embed.description = f"## {self.current_emotion} {self.current_dialogue}\n━━━━━━━━━━━━━━━━━━━━━━"

        if self.game_over:
            dealer_score = calculate_score(self.dealer_hand)
            embed.add_field(name=f"😈 庄家 ({dealer_score}点)", value=f"{format_hand(self.dealer_hand)}\n", inline=False)
            embed.add_field(name=f"👤 玩家 ({player_score}点)", value=f"{format_hand(self.player_hand)}\n", inline=False)
            
            result_header = "🏁 结算"
            if result_type == "win": result_header = "🎉 胜利！"
            elif result_type == "lose": result_header = "💀 失败..."
            elif result_type == "tie": result_header = "🤝 平局"
            embed.add_field(name=result_header, value=f"**{extra_msg}**", inline=False)
        else:
            embed.add_field(name="😈 庄家", value=f"{format_hand(self.dealer_hand, hide_second=True)}\n", inline=False)
            embed.add_field(name=f"👤 玩家 ({player_score}点)", value=f"{format_hand(self.player_hand)}\n", inline=False)
            embed.add_field(name="💰 当前赌注", value=f"`{self.bet}` 类脑币", inline=False)

        if self.game_over:
            self.clear_items()
            
            play_again_btn = Button(label="🔄 再来一局", style=discord.ButtonStyle.success)
            
            # --- [重点修改] 所有的 Replay 逻辑都在这里 ---
            async def play_again_callback(bt_interaction: discord.Interaction):
                if bt_interaction.user.id != self.user_id: return
                
                # 1. 先 defer，因为数据库查询可能耗时，也为了确认交互
                await bt_interaction.response.defer() 
                
                new_balance = await coin_service.get_balance(self.user_id)
                if new_balance <= 0:
                     # 没钱了发送一条临时消息，不覆盖当前的游戏结果
                     await bt_interaction.followup.send("你已经破产了！快去打工或者签到吧！", ephemeral=True)
                     return

                # 2. 准备新的下注界面
                new_view = BettingView(self.bot, self.user_id, new_balance)
                embed = discord.Embed(description=f"你的当前余额: **{new_balance}** 类脑币\n请选择下注金额：", color=discord.Color.blue())
                embed.set_author(name="Blackjack 赌场", icon_url=CASINO_ICON_URL)
                
                # 3. 使用 edit_original_response 覆盖原来的消息，而不是 send 发送新消息
                await bt_interaction.edit_original_response(embed=embed, view=new_view)
            
            play_again_btn.callback = play_again_callback
            self.add_item(play_again_btn)

            # 添加“我是杂鱼”按钮 (交换位置到再来一局后面)
            if self.shame_message:
                async def shame_callback(it: discord.Interaction):
                    if it.user.id != self.user_id: return
                    bal = await coin_service.get_balance(self.user_id)
                    if bal < 100: return await it.response.send_message("余额不足 100 类脑币！", ephemeral=True)
                    
                    await coin_service.remove_coins(self.user_id, 100, "删除嘲讽消息")
                    try:
                        await self.shame_message.delete()
                        self.shame_message = None
                        self.remove_item(shame_btn)
                        await interaction.edit_original_response(view=self)
                        await it.response.send_message("黑历史已删除...", ephemeral=True)
                    except:
                        await it.response.send_message("删除失败，可能已经被删除了。", ephemeral=True)

                shame_btn = Button(label="😈我是杂鱼 (-100币)", style=discord.ButtonStyle.primary)
                shame_btn.callback = shame_callback
                self.add_item(shame_btn)
            
            # 游戏结束时的更新
            if initial:
                await interaction.response.edit_message(content=None, embed=embed, view=self)
            else:
                await interaction.response.edit_message(embed=embed, view=self)
        else:
            # 游戏进行中的更新
            # 动态禁用加倍按钮 (只能在首轮/2张牌时使用)
            for child in self.children:
                if isinstance(child, Button) and child.label == "加倍":
                    child.disabled = len(self.player_hand) != 2

            # 游戏进行中的更新
            if initial:
                 await interaction.response.edit_message(content=None, embed=embed, view=self)
            else:
                 await interaction.response.edit_message(embed=embed, view=self)

    async def end_game(self, interaction: discord.Interaction, result_msg: str, multiplier: float, initial=False):
        self.game_over = True
        result_type = "lose"
        msg = ""
        
        if multiplier > 0:
            result_type = "win"
            profit = int(self.bet * multiplier)
            total_payout = self.bet + profit
            
            await coin_service.add_coins(self.user_id, total_payout, f"Blackjack 赢钱")
            msg = f"你一共获得 {total_payout} 类脑币"
        elif multiplier == 0:
            result_type = "tie"
            payout = self.bet
            await coin_service.add_coins(self.user_id, payout, "Blackjack 平局退款")
            msg = f"本金 {payout} 已全额退还。"
        else:
            result_type = "lose"
            msg = f"你输掉了 {self.bet} 类脑币。"
            
            try:
                if interaction.channel:
                    # --- [修改开始] 这里增加了随机嘲讽话术 ---
                    shame_templates = [
                        "{user} 小杂鱼在赌场输掉了 **{bet}** 类脑币~♥",
                        "大家快来看呀！{user} 又来送钱啦，输掉了 **{bet}** 类脑币！",
                        "噗... {user} 居然输了 **{bet}**，真是不自量力呢~",
                        "感谢 {user} 赞助的 **{bet}** 类脑币，今晚可以加餐了~",
                        "悲报：{user} 的 **{bet}** 类脑币瞬间蒸发了！好惨哦~",
                        "赢我的概率和 {user} 找到对象的概率一样低呢~ (输掉 **{bet}**)",
                        "哎呀，{user} 你的钱包还好吗？刚少了 **{bet}** 块呢~",
                        "是哪位小笨蛋在赌场里输掉了**{bet}**类脑币呀？"
                    ]
                    
                    # 随机选一句
                    template = random.choice(shame_templates)
                    # 填入变量
                    shame_msg = template.format(user=interaction.user.mention, bet=self.bet)
                    
                    self.shame_message = await interaction.channel.send(shame_msg)
                    # --- [修改结束] ---
            except Exception as e:
                log.error(f"公开处刑消息发送失败: {e}")

        color = discord.Color.gold() if result_type == "win" else (discord.Color.red() if result_type == "lose" else discord.Color.light_grey())
        await self.update_message(interaction, result_type, f"{result_msg}\n{msg}", color, initial=initial)

    @discord.ui.button(label="要牌", style=discord.ButtonStyle.primary, emoji="🃏")
    async def hit_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id: return
        self.current_dialogue = random.choice(DIALOGUES["hit"])
        self.player_hand.append(self.deck.draw())
        score = calculate_score(self.player_hand)
        
        if score > 21:
            await self.end_game(interaction, "💥 爆牌了！", -1)
        elif score == 21:
             await self.stand_logic(interaction)
        else:
            await self.update_message(interaction)

    @discord.ui.button(label="停牌", style=discord.ButtonStyle.secondary, emoji="🛑")
    async def stand_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id: return
        self.current_dialogue = random.choice(DIALOGUES["stand"])
        await self.stand_logic(interaction)

    async def stand_logic(self, interaction: discord.Interaction):
        while calculate_score(self.dealer_hand) < 17:
            self.dealer_hand.append(self.deck.draw())
        
        player_score = calculate_score(self.player_hand)
        dealer_score = calculate_score(self.dealer_hand)
        
        if dealer_score > 21:
            await self.end_game(interaction, "💥 庄家爆牌！", 1)
        elif player_score > dealer_score:
            await self.end_game(interaction, "🏆 你点数更大！", 1)
        elif player_score == dealer_score:
            await self.end_game(interaction, "🤝 点数相同。", 0)
        else:
            await self.end_game(interaction, "💔 庄家点数更大。", -1)

    @discord.ui.button(label="加倍", style=discord.ButtonStyle.primary, emoji="💰")
    async def double_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id: return
        
        if len(self.player_hand) != 2:
            return await interaction.response.send_message("只能在首轮加倍！", ephemeral=True)
            
        balance = await coin_service.get_balance(self.user_id)
        if balance < self.bet:
            return await interaction.response.send_message("余额不足，无法加倍！", ephemeral=True)
            
        if not await coin_service.remove_coins(self.user_id, self.bet, "Blackjack 加倍"):
             return await interaction.response.send_message("扣款失败，请稍后再试。", ephemeral=True)
             
        self.bet *= 2
        self.current_dialogue = "加倍？很有胆量嘛！小心爆掉哦~"
        self.player_hand.append(self.deck.draw())
        
        score = calculate_score(self.player_hand)
        if score > 21:
            await self.end_game(interaction, "💥 加倍后爆牌！", -1)
        else:
            # 加倍后强制停牌
            await self.stand_logic(interaction)

class BlackjackCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="blackjack", description="挑战类脑娘的Blackjack！")
    async def blackjack(self, interaction: discord.Interaction):
        balance = await coin_service.get_balance(interaction.user.id)
        
        if balance <= 0:
             return await interaction.response.send_message("你已经破产了！快去打工或者签到吧！", ephemeral=True)

        view = BettingView(self.bot, interaction.user.id, balance)
        
        embed = discord.Embed(description=f"你的当前余额: **{balance}** 类脑币\n请选择下注金额：", color=discord.Color.blue())
        embed.set_author(name="Blackjack 赌场", icon_url=CASINO_ICON_URL)
        
        # 初始发送使用 ephemeral=True，之后的 edit_original_response 会保持这个属性
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)

async def setup(bot: commands.Bot):
    await bot.add_cog(BlackjackCog(bot))