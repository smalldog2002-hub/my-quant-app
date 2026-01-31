import streamlit as st
import pandas as pd
import numpy as np
import base64
import requests
import time
import json
from datetime import datetime
from io import BytesIO
from PIL import Image

# ==========================================
# 0. 安全配置 (环境变量模式)
# ==========================================
try:
    SEC_GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "")
    SEC_TS_TOKEN = st.secrets.get("TUSHARE_TOKEN", "")
except Exception:
    SEC_GEMINI_KEY = ""
    SEC_TS_TOKEN = ""

# ==========================================
# 1. 注入自定义 CSS (隐藏密码框的“眼睛”图标)
# ==========================================
def hide_password_eye():
    st.markdown(
        """
        <style>
        /* 隐藏密码输入框右侧的显示/隐藏切换按钮（眼睛图标） */
        button[data-testid="stTextInputPasswordFieldVisibilityToggle"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ==========================================
# 2. 数据驱动引擎 (Tushare API)
# ==========================================
class TushareEngine:
    @staticmethod
    def get_data(api_name, token, params, fields=""):
        url = "http://api.tushare.pro"
        payload = {"api_name": api_name, "token": token, "params": params, "fields": fields}
        try:
            response = requests.post(url, json=payload, timeout=15)
            if response.status_code == 200:
                res = response.json()
                if res.get("code") == 0:
                    data = res.get("data")
                    return pd.DataFrame(data["items"], columns=data["fields"])
                else:
                    st.error(f"Tushare 错误: {res.get('msg')}")
        except Exception as e:
            st.error(f"Tushare 连接失败: {str(e)}")
        return None

    @staticmethod
    def format_code(code):
        code = code.strip()
        if not code: return ""
        if "." in code: return code
        if code.startswith("6"): return f"{code}.SH"
        if code.startswith("0") or code.startswith("3"): return f"{code}.SZ"
        if code.startswith("8") or code.startswith("4"): return f"{code}.BJ"
        return code

# ==========================================
# 3. 核心 AI 诊断引擎 (Gemini 2.5 Flash Preview)
# ==========================================
class GeminiAnalyst:
    @staticmethod
    def process_images(uploaded_files):
        processed_images = []
        for uploaded_file in uploaded_files:
            try:
                img = Image.open(uploaded_file)
                if img.mode != "RGB": img = img.convert("RGB")
                img.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
                buffered = BytesIO()
                img.save(buffered, format="JPEG", quality=90) 
                processed_images.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))
            except Exception as e:
                st.error(f"图片处理失败: {str(e)}")
        return processed_images

    @staticmethod
    def analyze_stock(prompt, api_key, images_base64=None, persona="平衡派", use_search=True, use_radar=True):
        if not api_key:
            return "❌ 未检测到 API Key。请在侧边栏高级设置中手动输入或在后台配置 Secrets。", []

        model_id = "gemini-2.5-flash-preview-09-2025" 
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
        
        parts = [{"text": prompt}]
        if images_base64:
            for b64 in images_base64:
                parts.append({"inlineData": {"mimeType": "image/jpeg", "data": b64}})
            
        system_instruction = f"""你是一位拥有 20 年实战经验的顶级基金经理。当前风格：{persona}。
任务：结合视觉图片（K线、指标）和数值数据，给出专业的操盘建议。
{'要求：必须使用 Google Search 工具核实最新消息。' if use_search else ''}
{'要求：在报告末尾，必须输出一个 [AI 五维能力综合评分表]，包含以下维度的 0-100 分打分：成长性、安全性、趋势性、资金面、热度。' if use_radar else ''}"""

        payload = {
            "contents": [{"parts": parts}],
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "tools": [{"google_search": {}}] if use_search else [], 
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 3000}
        }
        
        for i in range(3):
            try:
                response = requests.post(url, json=payload, timeout=120)
                if response.status_code == 200:
                    result = response.json()
                    candidate = result.get('candidates', [{}])[0]
                    text = candidate.get('content', {}).get('parts', [{}])[0].get('text', "")
                    sources = candidate.get('groundingMetadata', {}).get('groundingAttributions', [])
                    return text, sources
                else:
                    error_info = response.json().get('error', {}).get('message', '未知错误')
                    return f"❌ 诊断失败: {error_info}", []
            except:
                time.sleep(2)
        return "诊断服务暂时无法连接，请重试。", []

# ==========================================
# 4. UI 界面逻辑
# ==========================================
def main_app():
    st.set_page_config(page_title="Gemini 2.5 视觉量化系统", layout="wide", page_icon="📈")
    hide_password_eye() # 调用 CSS 隐藏函数
    
    # 初始化状态
    if 'stock_data' not in st.session_state:
        st.session_state.stock_data = {"price": 0.0, "change": 0.0, "pe": 0.0, "pb": 0.0}
    if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0
    if 'last_report' not in st.session_state: st.session_state.last_report = ""
    if 'chat_history' not in st.session_state: st.session_state.chat_history = []

    st.title("🚀 Gemini 2.5 视觉量化诊断系统")
    st.caption("核心能力：数据同步 | 2.5 Preview 引擎 | 联网搜索 | 研报导出 (GitHub 安全版)")
    st.markdown("---")
    
    with st.sidebar:
        # 将密钥输入隐藏在折叠器内
        with st.expander("🛠️ 高级接口设置 (已隐藏)", expanded=False):
            st.info("系统已自动加载云端 Secrets。如需覆盖，请在下方输入。")
            user_gemini_key = st.text_input("Gemini API Key", value=SEC_GEMINI_KEY, type="password", placeholder="请输入密钥")
            user_ts_token = st.text_input("Tushare Token", value=SEC_TS_TOKEN, type="password", placeholder="请输入 Token")
        
        if not user_gemini_key:
            st.error("⚠️ 未检测到 Gemini 密钥，请在上方高级设置中配置。")

        st.divider()
        persona = st.radio("专家诊断风格：", ["平衡派", "价值派", "技术派"], index=0)
        
        st.divider()
        st.header("🧮 风险管理")
        total_fund = st.number_input("账户总资金 (元)", value=100000)
        risk_per_trade = st.slider("单笔风险承受 (%)", 1.0, 5.0, 2.0)

        if st.button("🔄 重置系统状态"):
            st.session_state.stock_data = {"price": 0.0, "change": 0.0, "pe": 0.0, "pb": 0.0}
            st.session_state.uploader_key += 1
            st.session_state.last_report = ""
            st.session_state.chat_history = []
            st.rerun()

    tab_diag, tab_chat, tab_guide = st.tabs(["📊 诊断研报", "💬 深度追问", "📋 判定手册"])

    # --- Tab 1: 诊断研报模块 ---
    with tab_diag:
        sc1, sc2 = st.columns([3, 1])
        with sc1:
            stock_code = st.text_input("股票代码 (如 600519)", placeholder="输入后点击同步按钮")
        with sc2:
            st.write("")
            if st.button("🛰️ 同步数据"):
                if not user_ts_token: st.warning("请在高级设置中配置 Tushare Token")
                elif not stock_code: st.warning("请输入代码")
                else:
                    with st.spinner("从 Tushare 抓取数据中..."):
                        f_code = TushareEngine.format_code(stock_code)
                        d = TushareEngine.get_data("daily", user_ts_token, {"ts_code": f_code, "limit": 1})
                        b = TushareEngine.get_data("daily_basic", user_ts_token, {"ts_code": f_code, "limit": 1})
                        if d is not None and not d.empty:
                            st.session_state.stock_data["price"] = float(d.iloc[0]['close'])
                            st.session_state.stock_data["change"] = float(d.iloc[0]['pct_chg'])
                        if b is not None and not b.empty:
                            st.session_state.stock_data["pe"] = float(b.iloc[0]['pe_ttm'])
                            st.session_state.stock_data["pb"] = float(b.iloc[0]['pb'])
                        st.success("数据补全成功！")
                        st.rerun()

        with st.form("main_form"):
            st.subheader("1. 技术面 (TA)")
            c1, c2, c3 = st.columns(3)
            with c1:
                name_input = st.text_input("目标名称", value=stock_code if stock_code else "")
                price_input = st.number_input("价格", value=st.session_state.stock_data["price"], format="%.2f")
            with c2:
                chg_input = st.number_input("涨跌幅 (%)", value=st.session_state.stock_data["change"], format="%.2f")
                pe_input = st.number_input("PE (TTM)", value=st.session_state.stock_data["pe"], format="%.2f")
            with c3:
                vol_input = st.selectbox("成交量状态", ["由 AI 识别", "温和放量", "倍量拉升", "地量十字星", "天量滞涨"])
                ma_input = st.selectbox("均线排列特征", ["由 AI 识别", "多头排列", "回踩20日线", "粘合变盘"])
            
            st.divider()
            st.subheader("2. 基本面 (FA)")
            f1, f2 = st.columns(2)
            with f1:
                roe_input = st.number_input("净资产收益率 (%)", value=15.0)
                pb_input = st.number_input("市净率 (PB)", value=st.session_state.stock_data["pb"], format="%.2f")
                industry_input = st.text_input("概念板块", placeholder="如: AI、半导体")
            with f2:
                enable_search = st.checkbox("开启 AI 实时联网搜索资讯", value=True)
                enable_radar = st.checkbox("输出 AI 五维能力图表", value=True)
            
            submit_diagnosis = st.form_submit_button(f"🔥 启动 {persona} 深度诊断")

        st.divider()
        st.subheader("3. 视觉证据上传")
        up_files = st.file_uploader("📸 上传截图 (支持多选)", accept_multiple_files=True, key=f"up_{st.session_state.uploader_key}", type=["png", "jpg", "jpeg"])
        if up_files and st.button("🗑️ 一键清除图片"):
            st.session_state.uploader_key += 1
            st.rerun()

        if submit_diagnosis:
            if not user_gemini_key:
                st.error("❌ 密钥缺失。")
            elif not name_input:
                st.error("请输入目标名称。")
            else:
                with st.spinner("AI 专家正在扫描并执行联网搜索..."):
                    imgs_b64 = GeminiAnalyst.process_images(up_files) if up_files else None
                    prompt_text = f"目标:{name_input}, 价格:{price_input}, 涨跌:{chg_input}%, PE:{pe_input}, PB:{pb_input}, ROE:{roe_input}%, 行业:{industry_input}, 趋势:{ma_input}, 量能:{vol_input}"
                    res_text, src_links = GeminiAnalyst.analyze_stock(prompt_text, user_gemini_key, imgs_b64, persona=persona, use_search=enable_search, use_radar=enable_radar)
                    st.session_state.last_report = res_text
                    st.divider()
                    st.success(f"📈 {name_input} 投研诊断研报")
                    st.markdown(res_text)
                    
                    st.download_button(
                        label="📥 点击下载研报 (.md)",
                        data=res_text,
                        file_name=f"{name_input}_诊断研报_{datetime.now().strftime('%Y%m%d')}.md",
                        mime="text/markdown"
                    )

                    if src_links:
                        with st.expander("🔗 参考来源"):
                            for s in src_links: st.write(f"- [{s.get('title')}]({s.get('uri')})")

    # --- Tab 2: 深度追问 ---
    with tab_chat:
        st.header("💬 AI 专家深度追问")
        if not st.session_state.last_report:
            st.info("请先生成研报。")
        else:
            for chat in st.session_state.chat_history:
                with st.chat_message(chat["role"]): st.markdown(chat["content"])
            if query_input := st.chat_input("追问专家："):
                st.session_state.chat_history.append({"role": "user", "content": query_input})
                with st.chat_message("user"): st.markdown(query_input)
                with st.chat_message("assistant"):
                    with st.spinner("专家正在思考..."):
                        follow_up_prompt = f"基于报告：\n{st.session_state.last_report}\n\n回答：{query_input}"
                        ans_text, _ = GeminiAnalyst.analyze_stock(follow_up_prompt, user_gemini_key, persona=persona)
                        st.markdown(ans_text)
                        st.session_state.chat_history.append({"role": "assistant", "content": ans_text})

if __name__ == "__main__":
    main_app()
