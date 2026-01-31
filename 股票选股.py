import streamlit as st
import pandas as pd
import numpy as np
import base64
import requests
import time
import json
from datetime import datetime, timedelta
from io import BytesIO
from PIL import Image

# ==========================================
# 0. 全局配置
# ==========================================
apiKey = "AIzaSyCZUo71aX1jjk1B8AfDo__nOSRNQ6LF0Rg" 
# 已固定用户提供的 Tushare Token
tsToken = "f194e03b8127d27094934651740603868fd4f7e64ed732ea803c0150"

# ==========================================
# 1. 数据驱动引擎 (Tushare API 核心)
# ==========================================
class TushareEngine:
    @staticmethod
    def get_data(api_name, token, params, fields=""):
        """调用 Tushare HTTP 接口获取实时/历史行情"""
        url = "http://api.tushare.pro"
        payload = {
            "api_name": api_name,
            "token": token,
            "params": params,
            "fields": fields
        }
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                res = response.json()
                if res.get("code") == 0:
                    data = res.get("data")
                    return pd.DataFrame(data["items"], columns=data["fields"])
                else:
                    st.error(f"Tushare 错误: {res.get('msg')}")
        except Exception as e:
            st.error(f"连接 Tushare 失败: {str(e)}")
        return None

    @staticmethod
    def format_code(code):
        """自动补全 A 股代码后缀"""
        code = code.strip()
        if not code: return ""
        if "." in code: return code
        if code.startswith("6"): return f"{code}.SH"
        if code.startswith("0") or code.startswith("3"): return f"{code}.SZ"
        if code.startswith("8") or code.startswith("4"): return f"{code}.BJ"
        return code

# ==========================================
# 2. 核心 AI 诊断引擎
# ==========================================
class GeminiAnalyst:
    @staticmethod
    def process_images(uploaded_files):
        """处理并压缩图片"""
        processed_images = []
        for uploaded_file in uploaded_files:
            try:
                img = Image.open(uploaded_file)
                if img.mode != "RGB": img = img.convert("RGB")
                img.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
                buffered = BytesIO()
                img.save(buffered, format="PNG") 
                processed_images.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))
            except Exception as e:
                st.error(f"图片处理失败: {str(e)}")
        return processed_images

    @staticmethod
    def analyze_stock(prompt, images_base64=None, use_search=True, persona="平衡派"):
        """调用 Gemini 2.5 进行全维度诊断或对话"""
        model_id = "gemini-2.5-flash-preview-09-2025"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={apiKey}"
        
        parts = [{"text": prompt}]
        if images_base64:
            for b64 in images_base64:
                parts.append({"inlineData": {"mimeType": "image/png", "data": b64}})
            
        persona_prompts = {
            "价值派": "你是一位极其看重估值和财务确定性的投资大师，言辞严谨，注重风险边际。",
            "技术派": "你是一位专注于趋势和筹码博弈的短线专家，注重爆发力和止损位置。",
            "平衡派": "你是一位公募基金经理，平衡考虑公司的基本面品质与技术面的买入时机。"
        }

        system_instruction = f"""你是一位拥有 20 年经验的顶级基金经理。当前风格：{persona_prompts.get(persona)}。
你的任务是结合用户提供的数据和图片：
1. 视觉识别：图片中的均线位置、成交量异常、K线形态。
2. 数据解读：财务指标（PE, ROE等）是否健康。
3. 联网搜索：通过 Google Search 寻找该股最近 48 小时的突发新闻。
4. 明确决策：给出具体的投资建议。"""

        payload = {
            "contents": [{"parts": parts}],
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "tools": [{"google_search": {}}] if use_search else [], 
            "generationConfig": {"temperature": 0.15, "maxOutputTokens": 3000}
        }
        
        try:
            response = requests.post(url, json=payload, timeout=100)
            if response.status_code == 200:
                result = response.json()
                text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "")
                sources = result.get('candidates', [{}])[0].get('groundingMetadata', {}).get('groundingAttributions', [])
                return text, sources
        except Exception as e:
            st.error(f"AI 调用出错: {str(e)}")
        return "诊断服务暂时无响应，请重试。", []

# ==========================================
# 3. UI 界面逻辑
# ==========================================
def main_app():
    st.set_page_config(page_title="Gemini 2.5 至尊量化决策系统", layout="wide")
    
    # 初始化会话状态
    if 'stock_data' not in st.session_state:
        st.session_state.stock_data = {"price": 0.0, "change": 0.0, "pe": 0.0, "pb": 0.0, "name": ""}
    if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0
    if 'last_report' not in st.session_state: st.session_state.last_report = ""
    if 'chat_history' not in st.session_state: st.session_state.chat_history = []

    st.title("🚀 Gemini 2.5 至尊量化决策系统 (Pro 增强版)")
    st.caption("核心能力：Tushare 数据实装 + 联网资讯搜索 + 深度追问对话")
    st.markdown("---")
    
    # --- 侧边栏：配置与风险管理 ---
    with st.sidebar:
        st.header("🔑 接口配置")
        # 直接使用固定好的 tsToken
        st.success("Gemini API: 已就绪")
        st.success("Tushare Token: 已固定")
        
        st.divider()
        stock_persona = st.radio("选择诊断风格：", ["平衡派", "价值派", "技术派"])
        
        st.divider()
        st.header("🧮 风险头寸助手")
        total_fund = st.number_input("账户总资金 (元)", value=100000)
        risk_per_trade = st.slider("单笔风险承受 (%)", 1.0, 5.0, 2.0)
        
        st.divider()
        if st.button("🔄 重置系统所有缓存"):
            st.session_state.stock_data = {"price": 0.0, "change": 0.0, "pe": 0.0, "pb": 0.0, "name": ""}
            st.session_state.last_report = ""
            st.session_state.chat_history = []
            st.session_state.uploader_key += 1
            st.rerun()

    # 主标签页
    tab_diag, tab_chat, tab_guide = st.tabs(["📊 综合诊断报告", "💬 深度追问模块", "📋 判定手册"])

    # --- Tab 1: 综合诊断 ---
    with tab_diag:
        # 数据同步行
        sync_col1, sync_col2 = st.columns([3, 1])
        with sync_col1:
            stock_code = st.text_input("股票代码 (如 600519)", placeholder="输入后点击补全")
        with sync_col2:
            st.write("") # 垂直对齐
            if st.button("🛰️ 自动补全数据"):
                if not stock_code:
                    st.warning("请输入代码")
                else:
                    with st.spinner("数据链同步中..."):
                        f_code = TushareEngine.format_code(stock_code)
                        df_daily = TushareEngine.get_data("daily", tsToken, {"ts_code": f_code, "limit": 1})
                        df_basic = TushareEngine.get_data("daily_basic", tsToken, {"ts_code": f_code, "limit": 1})
                        
                        if df_daily is not None and not df_daily.empty:
                            st.session_state.stock_data["price"] = float(df_daily.iloc[0]['close'])
                            st.session_state.stock_data["change"] = float(df_daily.iloc[0]['pct_chg'])
                        if df_basic is not None and not df_basic.empty:
                            st.session_state.stock_data["pe"] = float(df_basic.iloc[0]['pe_ttm'])
                            st.session_state.stock_data["pb"] = float(df_basic.iloc[0]['pb'])
                        st.success("Tushare 数据已同步！")
                        st.rerun()

        # 核心表单
        with st.form("main_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                display_name = st.text_input("股票/板块名称", value=stock_code if stock_code else "")
                price = st.number_input("最新价格", value=st.session_state.stock_data["price"], format="%.2f")
            with c2:
                change = st.number_input("今日涨跌 (%)", value=st.session_state.stock_data["change"], format="%.2f")
                pe_val = st.number_input("市盈率 PE", value=st.session_state.stock_data["pe"], format="%.2f")
            with c3:
                vol = st.selectbox("量能状态", ["由 AI 识别", "温和放量", "倍量拉升", "地量缩量", "天量滞涨"])
                ma = st.selectbox("均线趋势", ["由 AI 识别", "多头排列", "回踩20日线", "空头阴跌"])

            st.divider()
            f1, f2 = st.columns(2)
            with f1:
                roe = st.number_input("ROE (%)", value=15.0)
                industry = st.text_input("概念板块", placeholder="如: AI、半导体")
            with f2:
                enable_search = st.checkbox("开启 AI 实时联网搜索资讯", value=True)
                enable_radar = st.checkbox("输出 AI 五维能力图", value=True)

            st.divider()
            up_files = st.file_uploader(
                "📸 上传行情截图 (K线、成交量、资金流等)", 
                accept_multiple_files=True, 
                key=f"up_{st.session_state.uploader_key}",
                type=["png", "jpg", "jpeg"]
            )
            submit = st.form_submit_button(f"🔥 启动【{stock_persona}】深度诊断")

        # 图片清除逻辑
        if up_files and st.button("🗑️ 一键清空已选图片"):
            st.session_state.uploader_key += 1
            st.rerun()

        if submit:
            if not display_name: st.error("请确认股票名称")
            else:
                with st.spinner(f"【{stock_persona}】专家正在为您复核图表及基本面..."):
                    imgs = GeminiAnalyst.process_images(up_files) if up_files else None
                    prompt = f"诊断对象:{display_name}, 性格:{stock_persona}, 价格:{price}, 涨跌:{change}%, PE:{pe_val}, ROE:{roe}%, 行业:{industry}, 均线:{ma}, 量能:{vol}"
                    
                    report, sources = GeminiAnalyst.analyze_stock(prompt, imgs, enable_search, persona=stock_persona)
                    st.session_state.last_report = report # 保存研报以供追问
                    
                    st.divider()
                    st.success(f"📊 {display_name} 全维度诊断研报")
                    st.markdown(report)
                    if sources:
                        with st.expander("🔗 联网搜索参考来源"):
                            for s in sources: st.write(f"- [{s.get('title')}]({s.get('uri')})")

    # --- Tab 2: 深度追问 ---
    with tab_chat:
        st.header("💬 AI 专家深度追问")
        if not st.session_state.last_report:
            st.info("💡 请先在‘综合诊断报告’页生成一份研报，之后可以针对报告内容进行深度追问。")
        else:
            for chat in st.session_state.chat_history:
                with st.chat_message(chat["role"]):
                    st.markdown(chat["content"])

            if query := st.chat_input("问问 AI 专家：如‘当前止损位建议设在哪里？’"):
                st.session_state.chat_history.append({"role": "user", "content": query})
                with st.chat_message("user"):
                    st.markdown(query)
                
                with st.chat_message("assistant"):
                    with st.spinner("正在思考中..."):
                        follow_up_prompt = f"基于以下诊断研报：\n{st.session_state.last_report}\n\n请回答用户的新问题：{query}"
                        ans, _ = GeminiAnalyst.analyze_stock(follow_up_prompt, persona=stock_persona)
                        st.markdown(ans)
                        st.session_state.chat_history.append({"role": "assistant", "content": ans})

    # --- Tab 3: 手册 ---
    with tab_guide:
        st.header("📖 研报判定手册")
        st.markdown("""
        - **数据自动同步**：在顶部填入股票代码并点击按钮，程序已固定您的 Token，将自动同步行情数据。
        - **追问技巧**：研报生成后，切换至“深度追问”页。您可以针对当前诊断进行细节咨询。
        - **性格说明**：技术派重成交量和均线；价值派重估值和 ROE。
        """)

if __name__ == "__main__":
    main_app()