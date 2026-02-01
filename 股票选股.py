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
# 1. 数据驱动引擎 (Tushare API)
# ==========================================
class TushareEngine:
    @staticmethod
    def get_data(api_name, token, params, fields=""):
        if not token:
            st.error("❌ Tushare Token 未配置，无法获取数据")
            return None
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
                    st.error(f"Tushare 接口报错: {res.get('msg')}")
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
# 2. 核心 AI 诊断引擎 (Gemini 2.5 Flash Preview)
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
            return "❌ API Key 缺失。请在侧边栏配置 Secrets 或手动输入。", []

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
# 3. UI 界面逻辑
# ==========================================
def main_app():
    st.set_page_config(page_title="Gemini 2.5 视觉量化系统", layout="wide", page_icon="📈")
    
    # 初始化状态
    if 'stock_data' not in st.session_state:
        st.session_state.stock_data = {"price": 0.0, "change": 0.0, "pe": 0.0, "pb": 0.0}
    if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0
    if 'last_report' not in st.session_state: st.session_state.last_report = ""
    if 'chat_history' not in st.session_state: st.session_state.chat_history = []

    st.title("🚀 视觉量化诊断系统")
    st.caption("核心能力：Secrets 安全加密 | 2.5 Preview 引擎 | 联网搜索 | 技术教学手册")
    st.markdown("---")
    
    # --- 侧边栏：安全 API 管理 ---
    with st.sidebar:
        st.header("🔑 API 密钥管理")
        
        # 1. Gemini Key 管理逻辑
        try:
            gemini_key_from_secrets = st.secrets.get("GEMINI_API_KEY", "")
        except:
            gemini_key_from_secrets = ""
            
        if gemini_key_from_secrets:
            st.success("✅ AI Key: 已从 Secrets 安全加载")
            # 提供覆盖选项
            if st.toggle("临时手动修改 AI Key"):
                gemini_key = st.text_input("输入新 Key (仅本次有效)", type="password", key="manual_gemini")
            else:
                gemini_key = gemini_key_from_secrets
        else:
            st.warning("⚠️ 未检测到 Secrets 配置")
            gemini_key = st.text_input("请输入 Gemini API Key", type="password", help="刷新页面需重新输入，建议配置 Secrets", key="manual_gemini_input")

        st.markdown("---")

        # 2. Tushare Token 管理逻辑
        try:
            ts_token_from_secrets = st.secrets.get("TUSHARE_TOKEN", "")
        except:
            ts_token_from_secrets = ""
            
        if ts_token_from_secrets:
            st.success("✅ Tushare Token: 已从 Secrets 安全加载")
            if st.toggle("临时手动修改 Tushare Token"):
                ts_token = st.text_input("输入新 Token (仅本次有效)", type="password", key="manual_ts")
            else:
                ts_token = ts_token_from_secrets
        else:
            st.info("ℹ️ Tushare Token (用于拉取数据)")
            ts_token = st.text_input("请输入 Tushare Token", type="password", key="manual_ts_input")

        st.divider()
        persona = st.radio("专家诊断风格选择：", ["平衡派", "价值派", "技术派"], index=0)
        
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
                if not ts_token: st.error("请先配置 Tushare Token")
                elif not stock_code: st.warning("请输入代码")
                else:
                    with st.spinner("实时数据抓取中..."):
                        f_code = TushareEngine.format_code(stock_code)
                        d = TushareEngine.get_data("daily", ts_token, {"ts_code": f_code, "limit": 1})
                        b = TushareEngine.get_data("daily_basic", ts_token, {"ts_code": f_code, "limit": 1})
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
            if not gemini_key:
                st.error("❌ 诊断失败：请在左侧配置有效的 Gemini API Key。")
            elif not name_input:
                st.error("请输入名称。")
            else:
                with st.spinner("AI 专家正在扫描并执行联网搜索..."):
                    imgs_b64 = GeminiAnalyst.process_images(up_files) if up_files else None
                    prompt_text = f"目标:{name_input}, 价格:{price_input}, 涨跌:{chg_input}%, PE:{pe_input}, PB:{pb_input}, ROE:{roe_input}%, 行业:{industry_input}, 趋势:{ma_input}, 量能:{vol_input}"
                    # 使用动态获取的 gemini_key
                    res_text, src_links = GeminiAnalyst.analyze_stock(prompt_text, gemini_key, imgs_b64, persona=persona, use_search=enable_search, use_radar=enable_radar)
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
                if not gemini_key:
                    st.error("请先在左侧配置 API Key")
                else:
                    st.session_state.chat_history.append({"role": "user", "content": query_input})
                    with st.chat_message("user"): st.markdown(query_input)
                    with st.chat_message("assistant"):
                        with st.spinner("专家正在思考..."):
                            follow_up_prompt = f"基于报告：\n{st.session_state.last_report}\n\n回答：{query_input}"
                            ans_text, _ = GeminiAnalyst.analyze_stock(follow_up_prompt, gemini_key, persona=persona)
                            st.markdown(ans_text)
                            st.session_state.chat_history.append({"role": "assistant", "content": ans_text})

    # --- Tab 3: 判定手册 (增强教学) ---
    with tab_guide:
        st.header("📖 股票技术知识与判定手册")
        
        # 1. 视频教学
        st.subheader("📺 视频教学课堂")
        st.caption("提示：以下视频来源于 B站，如果无法加载，请检查您的网络环境。")
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.info("🎥 基础教学")
            st.video("https://www.bilibili.com/video/BV1MsPWeUEXH?t=2.5")
        with v_col2:
            st.info("🎥 K线")
            st.video("https://www.bilibili.com/video/BV1MsPWeUEXH?t=2.5&p=5")

        st.divider()
        
        # 2. 经典 K 线形态
        st.subheader("🕯️ 经典 K 线形态图解")
        k_col1, k_col2 = st.columns(2)
        with k_col1:
            with st.expander("🔨 锤子线 (Hammer) - 看涨反转", expanded=True):
                st.markdown("""
                **形态特征**：
                - 实体较小，位于 K 线的上端。
                - 下影线长度至少是实体的 2 倍以上。
                - 几乎没有上影线。
                
                **操盘建议**：
                出现在连续下跌的底部，预示着空头抛压耗尽，主力资金在低位试探性买入。
                """)
            with st.expander("☀️ 启明之星 (Morning Star) - 底部确认"):
                st.markdown("""
                **形态特征**：
                - 由三根 K 线组成：长阴线 + 小十字星 + 长阳线。
                - 意味着股价由跌转平，再由平转涨。
                
                **操盘建议**：
                典型的反转信号。如果第三根阳线伴随成交量放大，可靠性极高。
                """)
        with k_col2:
            with st.expander("☁️ 乌云盖顶 (Dark Cloud Cover) - 看跌风险"):
                st.markdown("""
                **形态特征**：
                - 阳线后跟一根高开的阴线，且阴线收盘价深入阳线实体一半以下。
                
                **操盘建议**：
                出现在高位，意味着多头力量衰竭，主力正在撤离。
                """)
            with st.expander("🐦 三只乌鸦 (Three Black Crows) - 趋势下行"):
                st.markdown("""
                **形态特征**：
                - 连续出现三根收盘在最低点附近的长阴线。
                
                **操盘建议**：
                极强的看跌信号，暗示趋势已彻底转空，应坚决回避。
                """)

        st.divider()

        # 3. 判定口诀
        st.subheader("💠 系统评价逻辑")
        with st.expander("什么是 AI 五维能力图？"):
            st.markdown("""
            AI 在报告末尾生成的五维评价包含了：
            1. **成长性**：基于利润增速、ROE 和行业空间。
            2. **安全性**：基于估值位置和财务稳健度。
            3. **趋势性**：基于均线排列和 K 线形态。
            4. **资金面**：基于成交量异动和换手率。
            5. **热度**：基于新闻频次和板块共振。
            """)
            
        with st.expander("📌 技术指标判定口诀"):
            st.markdown("""
            - **多头排列**：MA5 > MA10 > MA20，买点在回踩。
            - **空头排列**：MA5 < MA10 < MA20，反弹即卖点。
            - **倍量拉升**：主力进攻，资金强介入。
            - **地量见底**：抛压耗尽，机会来临。
            """)

if __name__ == "__main__":
    main_app()

