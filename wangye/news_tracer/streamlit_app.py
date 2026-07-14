# streamlit_app.py
import streamlit as st
import requests
import json
import time
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from typing import List, Dict, Any
import os
from stock_analyzer import FinancialAnalyzer

# 页面配置
st.set_page_config(
    page_title="新闻溯源验证器",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        padding: 20px 0;
    }
    .card {
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        background-color: white;
    }
    .source-card {
        border-left: 4px solid #1f77b4;
        background-color: #f8f9fa;
        margin: 10px 0;
        padding: 15px;
    }
    .conclusion-card {
        background-color: #e8f4fd;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }
    .step-header {
        background-color: #1f77b4;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .tab-container {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 20px 0;
    }
    .metric-box {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        flex: 1;
        margin: 0 5px;
    }
    .metric-value {
        font-size: 1.5em;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9em;
        color: #666;
    }
    .news-event-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
        background-color: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

# 初始化session state
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "新闻溯源"  # 使用中文标签
    
if 'fin_submitted' not in st.session_state:
    st.session_state.fin_submitted = False

if 'history' not in st.session_state:
    # 从localStorage加载历史记录
    try:
        history_data = st.session_state.get('local_history', '[]')
        if isinstance(history_data, str):
            history_data = json.loads(history_data)
        st.session_state.history = history_data
    except:
        st.session_state.history = []

if 'current_result' not in st.session_state:
    st.session_state.current_result = None

if 'fin_symbol' not in st.session_state:
    st.session_state.fin_symbol = None

# API配置
API_BASE_URL = "http://localhost:8000/api"

# 页面标题
st.markdown("<h1 class='main-header'>📰 新闻溯源验证器</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>验证新闻声明的真实性和可靠性</p>", unsafe_allow_html=True)

# 创建标签页
tabs = ["新闻溯源", "金融数据分析"]
selected_tab = st.radio("选择功能", tabs, index=0)  # 默认选中第一个

# 更新当前标签页
st.session_state.current_tab = selected_tab

# 根据选中的标签页显示相应内容
if selected_tab == "新闻溯源":
    st.markdown("### 🔍 验证新闻声明")
    
    # 输入表单
    with st.form("verification_form"):
        claim = st.text_input("请输入要溯源的新闻声明或股票名称：", 
                             placeholder="例如：特斯拉最新消息 或 AAPL股票分析")
        max_depth = st.slider("溯源深度：", 1, 5, 2)
        submitted = st.form_submit_button("开始验证")
    
    if submitted and claim:
        with st.spinner("正在验证声明，请稍候..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/trace",
                    json={"claim": claim, "max_depth": max_depth},
                    timeout=300
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.current_result = result
                    
                    # 添加到历史记录
                    history_item = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "claim": claim,
                        "max_depth": max_depth,
                        "result": result,
                        "type": "news_trace"
                    }
                    st.session_state.history.append(history_item)
                    
                    # 保存到localStorage
                    st.session_state.local_history = json.dumps(st.session_state.history)
                    
                    st.success("验证完成！")
                else:
                    st.error(f"验证失败：{response.text}")
            except requests.exceptions.ConnectionError:
                st.error("连接被拒绝，请检查后端服务是否已启动。")
            except requests.exceptions.Timeout:
                st.error("请求超时，请稍后再试或联系管理员。")
            except requests.exceptions.RequestException as e:
                st.error(f"请求异常：{str(e)}")
            except Exception as e:
                st.error(f"请求出错：{str(e)}")

    # 显示结果
    if st.session_state.current_result:
        result = st.session_state.current_result

        # 1️⃣ 结论卡片
        st.markdown("### 📋 验证结论")
        with st.container():
            st.markdown(f"<div class='conclusion-card'>", unsafe_allow_html=True)
            claim_text = result.get('claim', '未知声明')
            st.markdown(f"**声明：** {claim_text}")
            st.markdown("**详细结论：**")
            conclusion_text = result.get('conclusion', '暂无结论')
            st.write(conclusion_text)
            st.markdown("</div>", unsafe_allow_html=True)

        # 2️⃣ 搜索步骤（原有来源卡片）
        st.markdown("### 🔍 搜索步骤")
        steps = result.get('steps', [])
        for step in steps:
            step_num = step.get('step', '未知')
            query = step.get('query', '未知查询')
            sources = step.get('sources', [])
            st.markdown(f"<div class='step-header'>步骤 {step_num}: {query}</div>", unsafe_allow_html=True)
            if sources:
                cols = st.columns(len(sources))
                for i, src in enumerate(sources):
                    with cols[i % len(cols)]:
                        st.markdown(f"<div class='source-card'>", unsafe_allow_html=True)
                        st.markdown(f"**{src.get('title','')}**")
                        st.markdown(f"[🔗 链接]({src.get('link','#')})")
                        st.markdown(src.get('snippet',''))
                        st.markdown(f"*发布日期: {src.get('date','未知')}*")
                        st.markdown("</div>", unsafe_allow_html=True)
        # 🔍 搜索步骤 下方新增
        st.markdown("### 🧭 递进溯源时间线")
        for r in result.get("rounds", []):
            with st.expander(f"第 {r['round']} 轮  —  关键词：{', '.join(r['keywords'])}"
                            f"{' ✋ 已追到 primary source' if r['stop_early'] else ''}"):
                st.write(f"**聚焦实体：** {r['next_focus']}")
                st.write(f"**搜索描述：** {r['query']}")
                if r["sources"]:
                    src_cols = st.columns(len(r["sources"]))
                    for i, src in enumerate(r["sources"]):
                        with src_cols[i % len(src_cols)]:
                            st.markdown(f"<div class='source-card'>", unsafe_allow_html=True)
                            st.markdown(f"**{src['title']}**")
                            st.markdown(f"[🔗 链接]({src['link']})")
                            st.caption(src["snippet"][:120] + "...")
                            st.caption(f"📅 {src['date']}")
                            st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.text("本轮未返回有效来源")

        # 3️⃣ 层级因果关系图（已升级）
        st.markdown("### 🌐 层级因果关系图")
        if len(steps) > 1:
            nodes, edges = [], []; node_id = 0; root_id = 0
            nodes.append(dict(id=node_id, label=claim_text, level=0, x=0, y=0, color="#ff4d4d", size=30)); node_id += 1
            for i, step in enumerate(steps):
                x_step = (i - (len(steps)-1)/2) * 1; y_step = -1
                nodes.append(dict(id=node_id, label=f"步骤{i+1}: {step.get('query','')}", level=1, x=x_step, y=y_step, color="#3399ff", size=20)); step_node_id = node_id; node_id += 1
                edges.append(dict(source=root_id, target=step_node_id))
                for j, src in enumerate(step.get('sources', [])[:2]):
                    x_src = x_step + (j - 0.5) * 0.3; y_src = y_step - 0.5
                    nodes.append(dict(id=node_id, label=src.get('title','')[:25]+"...", level=2, x=x_src, y=y_src, color="#52c41a", size=12)); edges.append(dict(source=step_node_id, target=node_id)); node_id += 1
            fig = go.Figure()
            for e in edges:
                fig.add_trace(go.Scatter(x=[nodes[e["source"]]["x"], nodes[e["target"]]["x"]], y=[nodes[e["source"]]["y"], nodes[e["target"]]["y"]], mode="lines", line=dict(color="#999", width=2), hoverinfo="none"))
            fig.add_trace(go.Scatter(x=[n["x"] for n in nodes], y=[n["y"] for n in nodes], mode="markers+text", marker=dict(size=[n["size"] for n in nodes], color=[n["color"] for n in nodes]), text=[n["label"] for n in nodes], textposition="top center", hovertemplate="%{text}<extra></extra>", showlegend=False))
            fig.update_layout(title="层级因果关系图谱", height=500, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), margin=dict(l=0, r=0, t=40, b=0), plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("至少需要 2 个搜索步骤才能生成层级图谱")

        # 4️⃣ 事件时间线（新）
        st.markdown("### 📅 事件时间线")
        timeline_data = []
        for step in steps:
            for src in step.get('sources', []):
                timeline_data.append({"title": src.get('title', ''), "date": src.get('date', '未知'), "snippet": src.get('snippet', '')[:120] + "...", "link": src.get('link', '#')})
        timeline_data.sort(key=lambda x: x["date"], reverse=True)
        for item in timeline_data:
            with st.expander(f"{item['date']}  {item['title'][:45]}"):
                st.write(item["snippet"])
                st.markdown(f"[🔗 阅读原文]({item['link']})")

        # 5️⃣ 词频统计（新）
        st.markdown("### 📊 关键词频")
        import re
        from collections import Counter
        STOP_WORDS = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "那", "他", "她", "它", "我们", "你们", "他们", "她们", "它们", "与", "或", "但", "如果", "因为", "所以", "虽然", "然而", "只是", "不过", "其实", "当然", "特别", "尤其", "比如", "例如", "像", "一样", "一般", "通常", "往往", "经常", "常常", "有时", "有时候", "可能", "也许", "大概", "或许", "应该", "必须", "一定", "当然", "其实", "实际上", "事实上", "确实", "的确", "真的", "肯定", "毫无疑问", "没问题", "行的", "可以", "好吧", "好吧", "好的", "好了", "好啦", "啦", "嘛", "呀", "啊", "呢", "吧", "哦", "哎", "哎呀", "唉", "哼", "嗯", "嗯嗯", "嘿嘿", "哈哈", "哈哈哈", "笑", "开心", "高兴", "快乐", "幸福", "美好", "漂亮", "美丽", "帅", "酷", "棒", "厉害", "牛逼", "强", "优秀", "出色", "精彩", "完美", "不错", "还行", "一般般", "凑合", "将就", "勉强", "马马虎虎", "差不多", "差一点", "不行", "不好", "差", "糟糕", "烂", "垃圾", "废物", "没用", "失败", "错误", "问题", "麻烦", "困难", "艰难", "艰苦", "辛苦", "累", "疲惫", "疲倦", "困", "想睡觉"}
        all_text = " ".join([src.get('title', '') + " " + src.get('snippet', '') for step in steps for src in step.get('sources', [])])
        words = re.findall(r"[a-zA-Z\u4e00-\u9fa5]{2,}", all_text.lower())
        filtered = [w for w in words if w not in STOP_WORDS and len(w) > 1]
        top20 = Counter(filtered).most_common(20)
        if top20:
            df_word = pd.DataFrame(top20, columns=["word", "count"])
            fig_bar = px.bar(df_word, x="count", y="word", orientation='h', title="Top 20 高频关键词", height=400, color="count", color_continuous_scale="Blues")
            fig_bar.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_bar, width='stretch')
        else:
            st.info("暂无有效词汇")
elif selected_tab == "金融数据分析":
    st.markdown("### 📈 金融数据分析")

    # ① 公司名 → 代码（只负责选代码）
    with st.form("name2code"):
        company = st.text_input("公司名称", placeholder="例如：小米、特斯拉")
        submitted_name = st.form_submit_button("查询代码")

    symbol = st.session_state.get("fin_symbol", None)
    if submitted_name and company:
        with st.spinner("正在搜索代码..."):
            resp = requests.post(f"{API_BASE_URL}/stock/code", json={"name": company}, timeout=15)
            if resp.status_code == 200:
                cand_raw = resp.json()["candidates"]

                # 过滤和格式化股票代码
                def normalize_hk_code(code: str) -> str:
                    if code.endswith(".HK"):
                        return code.replace(".HK", "").zfill(5) + ".HK"
                    return code

                import re
                pattern = re.compile(r'^\d+\.*\.(HK|SZ|SS|US)$', re.I)
                cand = [normalize_hk_code(c) for c in cand_raw if pattern.match(c.strip())]

                if not cand:
                    st.error("未识别到合法股票代码")
                    st.stop()
                symbol = cand[0] if len(cand) == 1 else st.selectbox("候选代码，请选择：", cand)
                st.session_state.fin_symbol = symbol
            else:
                st.error(resp.json().get("detail", "查询失败"))

    # ② 基础分析（只出现一次 finance_form）
    if symbol:
        # 创建选项卡来分隔不同功能
        finance_tabs = st.tabs(["基础分析", "策略回测", "AI投研仪表盘"])
        
        # 选项卡1: 基础分析
        with finance_tabs[0]:
            st.markdown("#### 📊 基础技术分析")
            
            with st.form("finance_form"):
                st.text_input("已选股票代码", value=symbol, disabled=True)
                time_range = st.selectbox("时间范围", ["w", "m", "y"], 
                                         format_func=lambda x: {"w": "本周", "m": "本月", "y": "今年"}[x])
                submitted = st.form_submit_button("开始分析")
                if submitted:
                    st.session_state.fin_submitted = True 

            if st.session_state.fin_submitted:
                symbol = st.session_state.fin_symbol
                from stock_analyzer import FinancialAnalyzer
                
                # ✅ 如果 session 里已带结果，直接复用
                if st.session_state.get('current_result') and st.session_state.current_result.get('symbol') == symbol:
                    info = st.session_state.current_result
                    st.success("已加载历史分析结果")
                    # 需要重新创建analyzer对象来生成图表
                    analyzer = FinancialAnalyzer()
                    analyzer.symbol = symbol
                    # 尝试从session获取数据
                    if hasattr(st.session_state, 'stock_data') and symbol in st.session_state.stock_data:
                        analyzer.data = st.session_state.stock_data[symbol]
                else:
                    # ✅ 否则实时拉取
                    with st.spinner("正在获取数据..."):
                        analyzer = FinancialAnalyzer()
                        days_map = {"w": 7, "m": 30, "y": 365}
                        end_date   = datetime.now().strftime("%Y%m%d")
                        start_date = (datetime.now() - timedelta(days=days_map[time_range])).strftime("%Y%m%d")
                        
                        # 获取数据
                        data = analyzer.fetch_data(symbol, period="daily", start_date=start_date, end_date=end_date)
                        if data.empty:
                            st.error("分析失败：未获取到数据")
                            st.stop()
                        
                        # 保存数据到session
                        if not hasattr(st.session_state, 'stock_data'):
                            st.session_state.stock_data = {}
                        st.session_state.stock_data[symbol] = data
                        
                        # 获取基本信息
                        info = analyzer.get_basic_info()
                        # 把结果缓存到 session
                        st.session_state.current_result = info

                # ✅ 统一渲染
                st.markdown("##### 📈 关键指标")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("代码", info.get("symbol", "N/A"))
                c2.metric("52W高/低", f"{info.get('high_52w', 0):.0f} / {info.get('low_52w', 0):.0f}")
                c3.metric("当前价", f"{info.get('current_price', 0):.2f}")
                c4.metric("RSI", f"{info.get('rsi', 0):.1f}")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("日均成交量", f"{info.get('avg_volume', 0):,.0f}")
                c2.metric("波动率", f"{info.get('volatility', 0):.2f}%")
                c3.metric("涨跌幅", "—")
                
                st.markdown("##### 📊 技术分析图表")
                try:
                    fig = analyzer.create_analysis_chart()
                    st.plotly_chart(fig, width='stretch', key='tech_analysis_chart')
                except Exception as e:
                    st.error(f"生成图表失败: {str(e)}")
                
                # 添加更多技术指标 - 使用现有的方法
                if st.button("显示更多技术指标", key="more_tech_btn"):
                    with st.spinner("计算技术指标..."):
                        try:
                            # 使用FinancialAnalyzer中的现有方法
                            st.markdown("##### 📈 详细技术指标")
                            
                            # 获取数据摘要
                            if hasattr(analyzer, 'get_data_summary'):
                                summary = analyzer.get_data_summary()
                                if summary:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write("**数据摘要**")
                                        st.write(f"数据期间: {summary.get('data_period', 'N/A')}")
                                        st.write(f"总交易日数: {summary.get('total_days', 0)}")
                                        st.write(f"价格范围: {summary.get('price_range', 'N/A')}")
                                    with col2:
                                        st.write("**风险指标**")
                                        st.write(f"年化波动率: {summary.get('volatility_annualized', 0):.2f}%")
                                        st.write(f"最大回撤: {summary.get('max_drawdown', 0):.2f}%")
                                        st.write(f"夏普比率: {summary.get('sharpe_ratio', 0):.3f}")
                            
                            # 获取技术分析
                            if hasattr(analyzer, 'get_technical_analysis'):
                                tech_analysis = analyzer.get_technical_analysis()
                                if tech_analysis:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write("**技术指标**")
                                        st.write(f"趋势判断: {tech_analysis.get('trend', 'N/A')}")
                                        st.write(f"RSI: {tech_analysis.get('rsi', 0):.2f}")
                                        st.write(f"MA20: {tech_analysis.get('ma_20', 0):.2f}")
                                        st.write(f"MA50: {tech_analysis.get('ma_50', 0):.2f}")
                                    
                                    with col2:
                                        st.write("**交易信号**")
                                        signals = tech_analysis.get('signals', {})
                                        for signal_name, signal_value in signals.items():
                                            st.write(f"{signal_name}: {signal_value}")
                                        
                                        # 显示其他信号
                                        st.write(f"布林带位置: {tech_analysis.get('bb_position', 'N/A')}")
                                        st.write(f"MACD信号: {tech_analysis.get('macd_signal', 'N/A')}")
                                        
                        except Exception as e:
                            st.error(f"计算技术指标时出错: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())

        # 选项卡2: 策略回测
        with finance_tabs[1]:
            st.markdown("#### 📈 策略回测")
            
            strategy_list = ["ma_crossover", "rsi_mean_revert", "bollinger_breakout"]
            strategy = st.selectbox("选择策略", ["ma_crossover", "rsi_mean_revert", "bollinger_breakout"],
                        key="strategy_key")
            initial_capital = st.number_input("初始资金（元）", min_value=10000, value=100000, 
                                           step=10000, key="capital_input")
            
            # 使用不同的容器和按钮
            if st.button("开始回测", key="backtest_btn"):
                with st.spinner("正在回测，预计需要 30-60 秒..."):
                    resp = requests.post(f"{API_BASE_URL}/stock/backtest",
                                 json={"symbol": symbol, "strategy": strategy, 
                                       "initial_capital": initial_capital},
                                 timeout=180)
                    if resp.status_code == 200:
                        bt = resp.json()
                        
                        st.markdown("##### 📊 回测结果")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("最终资产", f"¥{bt.get('final_value', 0):,.2f}")
                        col2.metric("总收益", f"{bt.get('total_return', 0)*100:.2f}%")
                        col3.metric("夏普比率", f"{bt.get('sharpe_ratio', 0):.2f}")
                        col4.metric("最大回撤", f"{bt.get('max_drawdown', 0)*100:.2f}%")
                        
                        # 回测图表
                        st.markdown("##### 📈 资金曲线")
                        if "equity_curve" in bt:
                            equity_data = bt["equity_curve"]
                            if isinstance(equity_data, dict) and "dates" in equity_data and "values" in equity_data:
                                import pandas as pd
                                equity_df = pd.DataFrame({
                                    'Date': equity_data["dates"],
                                    'Equity': equity_data["values"]
                                })
                                equity_df['Date'] = pd.to_datetime(equity_df['Date'])
                                st.line_chart(equity_df.set_index('Date'))
                        
                        if "chart_data" in bt:
                            st.markdown("##### 📊 回测详细图表")
                            fig = go.Figure(bt["chart_data"])
                            st.plotly_chart(fig, width='stretch', key='backtest_chart')
                        
                        # 添加交易详情
                        if st.checkbox("显示交易详情", key="show_trades"):
                            if "trades" in bt:
                                trades_df = pd.DataFrame(bt["trades"])
                                st.dataframe(trades_df)
                            else:
                                st.info("无交易记录")
                    else:
                        st.error(f"回测失败：{resp.text}")

            # ④ AI 策略推荐（独立按钮）
            st.markdown("### 🎯 AI 策略推荐")
            pref = st.selectbox("投资偏好", ["aggressive", "balanced", "conservative"])
            if st.button("生成策略建议"):
                with st.spinner("Agent 正在分析行情与新闻..."):
                    resp = requests.post(f"{API_BASE_URL}/stock/strategy_recommend",
                                        json={"symbol": symbol, "preference": pref},
                                        timeout=300)
                if resp.status_code == 200:
                    rec = resp.json()
                    st.success(f"推荐策略：**{rec['strategy']}**")
                    st.info(rec['reason'])
                    # 一键回填到策略回测下拉框（可选）
                    st.session_state.strategy_key = rec['strategy']   # 下拉框 key 用 session_state 绑定
                    st.experimental_rerun()   # 立即刷新，让下拉框选中推荐策略
                else:
                    st.error(f"策略推荐失败：{resp.text}")
        
        # 选项卡3: AI投研仪表盘
        with finance_tabs[2]:
            st.markdown("#### 📰 AI 投研仪表盘")
            
            # 创建子选项卡来组织AI投研内容
            ai_tabs = st.tabs(["核心指标", "新闻分析", "情绪分析"])
            
            with ai_tabs[0]:
                st.markdown("##### 🎯 核心指标")
                
                if st.button("生成投研仪表盘", key="ai_dashboard_btn"):
                    with st.spinner("正在生成投研仪表盘..."):
                        resp = requests.post(f"{API_BASE_URL}/stock/strategy_report",
                                             json={"symbol": symbol},
                                             timeout=300)
                    if resp.status_code == 200:
                        sr = resp.json()
                        st.session_state.ai_report = sr  # 保存到session state
                        
                        # ① 核心指标六宫格
                        col1, col2, col3, col4, col5, col6 = st.columns(6)
                        col1.metric("影响评分", f"{sr.get('impact_score', 0):.2f}")
                        col2.metric("综合方向", sr.get("direction", "neutral"))
                        col3.metric("策略信号", sr.get("strategy_signal", "暂无"))
                        col4.metric("置信度", f"{sr.get('reliability', 0):.1f}")
                        col5.metric("新闻条数", len(sr.get("news_events", [])))
                        col6.metric("平均可靠性", f"{sr.get('reliability', 0):.1f}")
                        
                        st.markdown("##### 🤖 Agent 建议")
                        st.info(sr.get("summary", "暂无总结"))
                    else:
                        st.error(f"投研仪表盘生成失败：{resp.text}")
            
            with ai_tabs[1]:
                st.markdown("##### 📰 新闻事件分析")
                
                if "ai_report" in st.session_state:
                    sr = st.session_state.ai_report
                    
                    # 事件时间轴
                    events = sr.get("news_events", [])
                    if events:
                        events.sort(key=lambda x: x.get("date", ""), reverse=True)
                        for ev in events:
                            with st.expander(f"{ev.get('date', '未知')}  {ev.get('title', '')[:50]}"):
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(ev.get("analysis", "")[:300] + "...")
                                with col2:
                                    st.markdown(f"**影响方向**: {ev.get('impact_direction', '中性')}")
                                    st.markdown(f"**影响强度**: {ev.get('impact_level', '中')}")
                                st.markdown(f"[🔗 原文链接]({ev.get('link', '#')})")
                    else:
                        st.info("本月暂无重大新闻事件")
                        
                    # 关键词分析
                    if "key_terms" in sr:
                        st.markdown("##### 🔑 关键词分析")
                        key_terms = sr["key_terms"]
                        if key_terms:
                            terms_df = pd.DataFrame(list(key_terms.items()), columns=["关键词", "出现次数"])
                            st.dataframe(terms_df)
            
            with ai_tabs[2]:
                st.markdown("##### 📊 情绪 & 可靠性分析")
                
                if "ai_report" in st.session_state:
                    sr = st.session_state.ai_report
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        score = sr.get("impact_score", 0.0)
                        fig_gauge1 = go.Figure(go.Indicator(
                            mode="gauge+number+delta", value=score,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "新闻影响强度"},
                            delta={'reference': 0},
                            gauge={'axis': {'range': [-1, 1]},
                                   'bar': {'color': "darkblue"},
                                   'steps': [{'range': [-1, -0.3], 'color': "lightgray"},
                                             {'range': [-0.3, 0.3], 'color': "gray"},
                                             {'range': [0.3, 1], 'color': "lightgreen"}]}))
                        fig_gauge1.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
                        st.plotly_chart(fig_gauge1, width='stretch', key='impact_gauge')

                    with col2:
                        reli = sr.get("reliability", 0.5)
                        fig_gauge2 = go.Figure(go.Indicator(
                            mode="gauge+number", value=reli,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "信源可靠性"},
                            gauge={'axis': {'range': [0, 1]},
                                   'bar': {'color': "#52c41a"},
                                   'steps': [{'range': [0, 0.5], 'color': "#ffccc7"},
                                             {'range': [0.5, 0.8], 'color': "#fff3cd"},
                                             {'range': [0.8, 1], 'color': "#d9f7d9"}]}))
                        fig_gauge2.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
                        st.plotly_chart(fig_gauge2, width='stretch', key='reliability_gauge')
                    
                    # 情绪趋势图表
                    if "sentiment_trend" in sr and sr["sentiment_trend"]:
                        st.markdown("##### 📈 情绪趋势")
                        trend_data = sr["sentiment_trend"]
                        if isinstance(trend_data, list) and len(trend_data) > 0:
                            trend_df = pd.DataFrame(trend_data)
                            fig_trend = px.line(trend_df, x='date', y='sentiment', 
                                               title='新闻情绪趋势变化',
                                               markers=True)
                            st.plotly_chart(fig_trend, width='stretch', key='sentiment_trend_chart')
        
# 添加历史记录面板
with st.sidebar:
    st.markdown("### 📚 历史记录")
    
    if st.session_state.history:
        # 按时间倒序排列
        sorted_history = sorted(st.session_state.history, 
                              key=lambda x: x['timestamp'], reverse=True)
        
        for i, item in enumerate(sorted_history[:10]):
            if item['type'] == 'news_trace':
                claim = item.get('claim', '未知声明')
                max_depth = item.get('max_depth', 0)
                history_text = f"🔍 {claim[:30]}..."
                history_subtitle = f"深度: {max_depth}"

            elif item['type'] == 'stock_analysis':          # ←←← 修复
                symbol = item.get('symbol', '未知股票')
                tr = item.get('time_range', '')
                history_text = f"📈 {symbol}  {tr}"
                history_subtitle = f"时间: {item['timestamp']}"

            else:  # 预留其他类型
                symbol = item.get('symbol', '未知股票')
                history_text = f"📰 {symbol} 新闻分析"
                history_subtitle = f"时间: {item['timestamp']}"
            
            with st.expander(f"{history_text}"):
                st.write(f"**时间**: {item['timestamp']}")
                analysis_type = item.get('analysis_type', '新闻溯源')
                st.write(f"**类型**: {analysis_type}")
                if st.button(f"加载结果 #{i+1}", key=f"load_{i}"):
                    if item['type'] == 'stock_analysis':
                        st.session_state.fin_symbol   = item['symbol']
                        st.session_state.current_tab  = "金融数据分析"
                        # ✅ 把历史结果写进缓存，并补 symbol 字段
                        hist_result = item['result']
                        hist_result['symbol'] = item['symbol']
                        st.session_state.current_result = hist_result
                        st.session_state.fin_submitted  = True
                        st.rerun()
                    else:  # 新闻溯源
                        st.session_state.current_tab = "新闻溯源"
                        st.session_state.current_result = item.get('result', {})
                        # ✅ 把这条记录重新压栈，保证下次还在历史里
                        st.session_state.history.append(item)
                        st.session_state.local_history = json.dumps(st.session_state.history)
                        st.success("结果已加载到当前页面")
                        st.rerun()
    else:
        st.info("暂无历史记录")
    
    # 清除历史记录按钮
    if st.button("🗑️ 清除历史记录"):
        st.session_state.history = []
        st.session_state.local_history = json.dumps([])
        st.success("历史记录已清除")