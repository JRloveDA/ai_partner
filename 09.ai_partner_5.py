###############################################################################
# 代码运行思路与布局说明
#
# 【整体架构】
#   本应用采用面向对象设计，所有功能封装在 AIPartner 类中，分为三大模块：
#   1. 初始化模块    - 设置页面配置、初始化会话状态、创建AI客户端
#   2. 会话管理模块  - 保存/加载/删除聊天会话，支持多会话切换
#   3. UI渲染模块    - 渲染侧边栏控制面板和主聊天界面
#
# 【运行流程】
#   程序入口 (if __name__ == "__main__")
#        ↓
#   创建 AIPartner 实例 (__init__)
#        ↓
#   初始化会话状态 (_init_session_state)
#        ↓
#   配置页面外观 (_setup_page)
#        ↓
#   调用 run() 方法启动主循环：
#        ├─ render_chat()        渲染聊天消息界面
#        ├─ render_sidebar()     渲染侧边栏（伴侣信息 + 会话控制）
#        └─ handle_chat_input()  处理用户输入并调用AI生成回复
#        ↓
#   Streamlit 每次用户操作后重新执行整个脚本，形成交互循环
#
# 【页面布局】
#   ┌─────────────────────────────────────────────────────┐
#   │  🤖 AI智能伴侣              [Logo]                  │  ← 页面标题
#   ├──────────────────────┬──────────────────────────────┤
#   │                      │                              │
#   │   侧边栏 (Sidebar)   │      主聊天区域 (Main)       │
#   │                      │                              │
#   │  ┌────────────────┐  │  当前会话标识: 2026-11-11... │
#   │  │ 伴侣信息        │  │                              │
#   │  │ ├ 昵称输入框    │  │  ┌────────────────────────┐  │
#   │  │ └ 性格描述框    │  │  │ 用户: 你好             │  │
#   │  ├────────────────┤  │  └────────────────────────┘  │
#   │  │ AI控制面板      │  │  ┌────────────────────────┐  │
#   │  │ ├ [新建会话]   │  │  │ AI: 你好呀！❤️          │  │
#   │  │ └ 会话历史列表  │  │  └────────────────────────┘  │
#   │  │   ├ 2026-11-11 │  │                              │
#   │  │   └ 2026-11-10 │  │  [请输入你的问题...]         │
#   │  └────────────────┘  │                              │
#   │                      │                              │
#   └──────────────────────┴──────────────────────────────┘
#
# 【数据流向】
#   用户输入 → handle_chat_input() → 调用DeepSeek API → 流式输出AI回复
#                                                ↓
#                                       更新 session_state.messages
#                                                ↓
#                                       save_session() 保存到 JSON 文件
#
# 【会话存储】
#   所有会话数据保存在 sessions/ 目录下，每个会话一个 JSON 文件，
#   文件名格式为: YYYY-MM-DD_HH-MM-SS.json
#
# 【技术栈】
#   - Streamlit: Web UI 框架
#   - OpenAI SDK: 调用 DeepSeek 大语言模型 API
#   - JSON: 会话数据持久化
###############################################################################

import json
import os
from datetime import datetime
import streamlit as st
from openai import OpenAI

class AIPartner:
    """AI智能伴侣主类"""
    def __init__(self):
        """初始化AI智能伴侣应用
        
        功能：
            - 创建OpenAI客户端，用于调用DeepSeek大语言模型API
            - 初始化Streamlit会话状态，设置默认值
            - 配置Streamlit页面布局和外观
        
        参数：
            无
        
        返回值：
            无
        """
        self.client = OpenAI(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com"
        )
        self._init_session_state()
        self._setup_page()

    def _init_session_state(self):
        """初始化Streamlit会话状态
        
        功能：
            设置应用所需的默认会话状态变量，包括聊天消息列表、伴侣昵称、
            性格描述和当前会话标识。仅在变量不存在时设置，避免覆盖用户数据。
        
        参数：
            无
        
        返回值：
            无
        
        设置的默认值：
            - messages: 空列表，用于存储聊天消息
            - nick_name: "小甜甜"，伴侣的默认昵称
            - nature: "活泼开朗的东北姑娘"，伴侣的默认性格描述
            - current_session: 当前时间生成的会话标识符
        """
        defaults = {
            "messages": [],
            "nick_name": "小甜甜",
            "nature": "活泼开朗的东北姑娘",
            "current_session": self._generate_session_id()
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def _setup_page(self):
        """配置Streamlit页面外观和布局
        
        功能：
            设置页面的标题、图标、布局方式、侧边栏状态等，
            并添加页面主标题和Logo图片。
        
        参数：
            无
        
        返回值：
            无
        
        页面配置：
            - page_title: 浏览器标签页显示的标题
            - page_icon: 页面图标（机器人表情）
            - layout: 使用宽屏布局
            - initial_sidebar_state: 侧边栏默认展开
            - menu_items: 隐藏Streamlit默认菜单
        """
        st.set_page_config(
            page_title="AI智能伴侣",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={}
        )
        st.title("AI智能伴侣")
        st.logo("resources/logo.png")

    def _generate_session_id(self):
        """生成会话标识符
        
        功能：
            基于当前时间生成唯一的会话标识符，用于区分不同的聊天会话。
        
        参数：
            无
        
        返回值：
            str: 格式为"YYYY-MM-DD_HH-MM-SS"的字符串，例如"2026-11-11_17-30-00"
        """
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ==================== 会话管理 ====================

    def save_session(self):
        """保存当前会话到JSON文件
        
        功能：
            将当前会话的所有数据（昵称、性格、会话标识、聊天消息）
            序列化为JSON格式，保存到sessions目录下。
        
        参数：
            无
        
        返回值：
            无
        
        保存内容：
            - nick_name: 伴侣昵称
            - nature: 伴侣性格描述
            - current_session: 当前会话标识
            - messages: 聊天消息列表
        
        文件路径：
            sessions/{current_session}.json
        """
        if not st.session_state.current_session:
            return
        session_data = {
            "nick_name": st.session_state.nick_name,
            "nature": st.session_state.nature,
            "current_session": st.session_state.current_session,
            "messages": st.session_state.messages
        }
        os.makedirs("sessions", exist_ok=True)
        with open(f"sessions/{st.session_state.current_session}.json","w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)

    def load_sessions(self):
        """加载所有会话列表
        
        功能：
            扫描sessions目录下的所有JSON文件，提取会话标识符，
            并按时间倒序排列（最新的会话在前）。
        
        参数：
            无
        
        返回值：
            list: 会话标识符字符串列表，按时间倒序排列
                  例如：["2026-11-11_17-30-00", "2026-11-11_16-20-00"]
        """
        if not os.path.exists("sessions"):
            return []
        session_list = [f[:-5] for f in os.listdir("sessions") if f.endswith(".json")]
        session_list.sort(reverse=True)
        return session_list

    def load_session(self, session_id):
        """加载指定会话的数据到当前状态
        
        功能：
            从JSON文件中读取指定会话的数据，并更新Streamlit会话状态，
            使界面显示该会话的聊天内容。
        
        参数：
            session_id (str): 要加载的会话标识符，例如"2026-11-11_17-30-00"
        
        返回值：
            无
        
        异常处理：
            如果文件读取失败，会在界面上显示错误提示
        """
        try:
            with open(f"sessions/{session_id}.json", "r",encoding="utf-8") as f:
                data = json.load(f)
                st.session_state.update({
                    "nick_name": data["nick_name"],
                    "nature": data["nature"],
                    "current_session": data["current_session"],
                    "messages": data["messages"]
                })
        except Exception:
            st.error("加载会话信息失败!")

    def delete_session(self, session_id):
        """删除指定会话
        
        功能：
            删除sessions目录下的指定会话文件。如果删除的是当前会话，
            则清空消息列表并生成新的会话标识。
        
        参数：
            session_id (str): 要删除的会话标识符
        
        返回值：
            无
        
        异常处理：
            如果文件删除失败，会在界面上显示错误提示
        """
        try:
            if os.path.exists(f"sessions/{session_id}.json"):
                os.remove(f"sessions/{session_id}.json")
                if session_id == st.session_state.current_session:
                    st.session_state.messages = []
                    st.session_state.current_session = self._generate_session_id()
        except Exception:
            st.error("删除会话信息失败!")

    # ==================== UI 渲染 ====================

    def render_sidebar(self):
        """渲染侧边栏界面
        
        功能：
            在Streamlit侧边栏中渲染伴侣信息设置和会话控制面板，
            包括昵称输入、性格描述、新建会话按钮和历史会话列表。
        
        参数：
            无
        
        返回值：
            无
        """
        with st.sidebar:
            self._render_partner_info()
            st.divider()
            self._render_session_control()

    def _render_partner_info(self):
        """渲染伴侣信息设置区域
        
        功能：
            在侧边栏中显示昵称和性格描述的输入控件，
            允许用户自定义AI伴侣的身份特征。
        
        参数：
            无
        
        返回值：
            无
        
        界面元素：
            - 昵称输入框：单行文本输入
            - 性格描述框：多行文本输入
        """
        st.subheader("伴侣信息")
        nick_name = st.text_input("昵称", placeholder="请输入昵称",
                                  value=st.session_state.nick_name)
        if nick_name:
            st.session_state.nick_name = nick_name

        nature = st.text_area("性格描述", placeholder="请输入性格描述",value=st.session_state.nature)
        if nature:
            st.session_state.nature = nature

    def _get_empty_session(self):
        """查找第一个空会话的标识符
        
        功能：
            遍历所有已保存的会话文件，查找消息列表为空的会话，
            返回第一个找到的空会话标识符。
        
        参数：
            无
        
        返回值：
            str or None: 空会话的标识符字符串，如果不存在空会话则返回None
        
        空会话定义：
            会话文件中messages字段为空列表的会话
        """
        sessions = self.load_sessions()
        for session_id in sessions:
            try:
                with open(f"sessions/{session_id}.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # 如果会话的messages为空，则认为是一个空会话
                    if not data.get("messages"):
                        return session_id
            except Exception:
                continue
        return None

    def _render_session_control(self):
        """渲染会话控制面板
        
        功能：
            在侧边栏中显示会话管理界面，包括新建会话按钮和历史会话列表。
            每个历史会话都有加载和删除按钮。
        
        参数：
            无
        
        返回值：
            无
        
        新建会话逻辑：
            1. 检查是否存在空会话，如果存在则激活该空会话
            2. 保存当前会话数据
            3. 如果当前会话有消息，则创建新的空会话
        
        界面元素：
            - 新建会话按钮：创建新对话或激活空会话
            - 会话历史列表：显示所有已保存的会话
            - 加载按钮：切换到指定会话
            - 删除按钮：删除指定会话
        """
        st.subheader("AI控制面板")

        if st.button("新建会话", width="stretch", icon="✏️"):
            # 检查是否存在空会话
            empty_session = self._get_empty_session()
            if empty_session:
                # 激活该空会话
                self.load_session(empty_session)
                st.rerun()
            
            # 先保存当前会话
            self.save_session()
            # 如果当前会话有消息，才创建新会话
            if st.session_state.messages:
                st.session_state.messages = []
                st.session_state.current_session = self._generate_session_id()
                self.save_session()
                st.rerun()

        st.text("会话历史")
        for session in self.load_sessions():
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(session, width="stretch", icon="📄",
                           key=f"load_{session}",
                           type="primary" if session == st.session_state.current_session else "secondary"):
                    self.load_session(session)
                    st.rerun()
            with col2:
                if st.button("", width="stretch", icon="🗑️",key=f"del_{session}"):
                    self.delete_session(session)
                    st.rerun()

    def render_chat(self):
        """渲染聊天消息界面
        
        功能：
            在主界面显示当前会话标识和所有聊天消息，
            区分用户消息和AI助手消息的显示样式。
        
        参数：
            无
        
        返回值：
            无
        
        界面元素：
            - 会话标识文本：显示当前会话的唯一标识
            - 聊天消息列表：按时间顺序显示所有对话内容
        """
        st.text(f"当前会话标识: {st.session_state.current_session}")
        for message in st.session_state.messages:
            st.chat_message(message["role"]).write(message["content"])

    def handle_chat_input(self):
        """处理用户聊天输入并调用AI生成回复
        
        功能：
            1. 显示聊天输入框，等待用户输入
            2. 将用户消息添加到消息列表并显示
            3. 构建包含角色设定的系统提示词
            4. 调用DeepSeek大语言模型API生成AI回复
            5. 以流式方式实时显示AI回复内容
            6. 保存完整对话到会话文件
        
        参数：
            无
        
        返回值：
            无
        
        AI模型配置：
            - model: deepseek-flash
            - stream: 启用流式输出
            - reasoning_effort: 低推理强度
            - thinking: 禁用思考模式
        """
        prompt = st.chat_input("请输入你的问题")
        if not prompt:
            return

        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content":prompt})

        # 构建系统提示词
        system_prompt = self._build_system_prompt()

        # 调用AI
        response = self.client.chat.completions.create(
            model="deepseek-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                *st.session_state.messages
            ],
            stream=True,
            reasoning_effort="low",
            extra_body={"thinking": {"type": "disabled"}}
        )

        # 流式输出
        response_message = st.empty()
        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                response_message.chat_message("assistant").write(full_response)

        st.session_state.messages.append({"role": "assistant","content": full_response})
        self.save_session()

    def _build_system_prompt(self):
        """构建AI角色扮演的系统提示词
        
        功能：
            根据当前设置的伴侣昵称和性格描述，生成用于指导AI行为的
            系统提示词，使AI能够以伴侣身份与用户对话。
        
        参数：
            无
        
        返回值：
            str: 包含角色设定和行为规则的系统提示词字符串
        
        提示词内容：
            - 角色身份：使用设置的昵称
            - 行为规则：7条对话规则（单条回复、无场景描述等）
            - 性格特征：使用设置的性格描述
        """
        return f"""

你叫{st.session_state.nick_name}，现在是用户的真实伴侣，请完全代入伴侣
角色。
    规则：
        1.每次只回1条信息
        2.禁止任何场景或状态描述性文字
        3.匹配用户的语言
        4.回复简短，像微信聊天一样
        5.有需要的话可以用❤️🌸等emoji表情
        6.用符合伴侣性格的方式对话
        7.回复的内容，要充分体现伴侣的性格特征
    伴侣性格：
        - {st.session_state.nature}
    你必须严格遵守上述规则来回复用户。
"""

    # ==================== 主运行方法 ====================

    def run(self):
        """运行AI智能伴侣应用的主入口
        
        功能：
            按顺序执行应用的三个主要功能：
            1. 渲染聊天消息界面
            2. 渲染侧边栏控制面板
            3. 处理用户输入并生成AI回复
        
        参数：
            无
        
        返回值：
            无
        
        执行顺序：
            render_chat() → render_sidebar() → handle_chat_input()
        """
        self.render_chat()
        self.render_sidebar()
        self.handle_chat_input()
# 入口点
if __name__ == "__main__":
    app = AIPartner()
    app.run()