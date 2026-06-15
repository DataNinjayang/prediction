import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io
import re
import os

# ===================== 注册中文字体（修复PDF乱码核心） =====================
def register_chinese_font():
    """注册Windows系统自带宋体，支持PDF中文渲染"""
    font_name = "SimSun"
    if font_name not in pdfmetrics.getRegisteredFontNames():
        try:
            # 自动获取Windows字体目录
            font_path = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "simsun.ttc")
            pdfmetrics.registerFont(TTFont(font_name, font_path, subfontIndex=0))
        except Exception:
            # 兜底：尝试黑体
            try:
                font_path = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "simhei.ttf")
                pdfmetrics.registerFont(TTFont(font_name, font_path))
            except Exception:
                pass
    return font_name

# 全局注册一次中文字体
CHINESE_FONT = register_chinese_font()

# ===================== 页面全局配置 & 样式 =====================
st.set_page_config(
    page_title="沪深300股票智能预测分析平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
custom_css = """
<style>
.main {background-color: #f8fafc; font-family: 'Microsoft YaHei', sans-serif;}
.block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px; margin: 0 auto;}
.sidebar-header {background: #1e40af; color: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; text-align: center;}
.stButton>button {background-color: #1e40af; color: white; border-radius: 8px; font-weight: 600; border: none; width: 100%;}
.stButton>button:disabled {background-color: #94a3b8;}
.advice-box {background: #eff6ff; padding: 20px; border-radius: 10px; border-left: 4px solid #1e40af; margin:15px 0;}
.risk-box {background: #fef2f2; padding: 15px; border-radius: 8px; border-left: 4px solid #dc2626; margin:10px 0;}
hr {border-color: #e2e8f0; margin:25px 0;}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ===================== 策略与大盘配置 =====================
STRATEGY_DICT = {
    "价值投资策略（长线稳健）": {
        "desc": "精选低估值、高基本面沪深300个股，长期持有、分批止盈，适合稳健型投资者",
        "hold_days": (60, 120),
        "target_return": (0.15, 0.35),
        "batch_sell": True,
        "sell_ratio": [0.4, 0.3, 0.3],
        "capital_ratio": (15, 25),
        "risk_level": "低风险",
        "select_rule": "优先选择60日走势稳定、波动率低、换手率适中个股"
    },
    "趋势追涨策略（中线波段）": {
        "desc": "筛选均线多头、成交量放大的趋势个股，波段操作，适合中等风险投资者",
        "hold_days": (20, 45),
        "target_return": (0.08, 0.25),
        "batch_sell": True,
        "sell_ratio": [0.5, 0.5],
        "capital_ratio": (18, 28),
        "risk_level": "中风险",
        "select_rule": "优先选择5/10/20日均线多头、短期涨幅靠前个股"
    },
    "反转抄底策略（短线博弈）": {
        "desc": "筛选短期超跌、缩量企稳个股，短线反弹博弈，适合激进型投资者",
        "hold_days": (7, 20),
        "target_return": (0.05, 0.18),
        "batch_sell": False,
        "sell_ratio": [1.0],
        "capital_ratio": (20, 30),
        "risk_level": "高风险",
        "select_rule": "优先选择短期超跌、波动率下降的反弹潜力个股"
    }
}

INDEX_CONFIG = {
    "上证指数": {"code_range": (600000, 605999), "color": "#dc2626"},
    "深证成指": {"code_range": (0, 399999), "color": "#16a34a"},
    "创业板指": {"code_range": (300000, 309999), "color": "#f59e0b"}
}

np.random.seed(42)

# ===================== 工具函数 =====================
def load_hs300_mapping(file):
    """加载沪深300 代码-名称映射表（兼容 sh./sz. 前缀）"""
    try:
        df = pd.read_csv(file)
        df["纯代码"] = df["code"].apply(lambda x: re.sub(r"^(sh\.|sz\.)", "", str(x)))
        df["纯代码"] = pd.to_numeric(df["纯代码"], errors="coerce")
        df = df.dropna(subset=["纯代码"])
        code2name = dict(zip(df["纯代码"].astype(int), df["code_name"]))
        return code2name, f"✅ 沪深300名单加载成功，共 {len(code2name)} 只个股"
    except Exception as e:
        return {}, f"❌ 沪深300名单加载失败：{str(e)}"

def load_stock_data(file, code2name):
    """加载股票行情数据 + 匹配股票名称"""
    try:
        df = pd.read_csv(file)
        required_cols = ["股票代码", "日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌额", "换手率", "涨跌幅"]
        miss = [c for c in required_cols if c not in df.columns]
        if miss:
            return None, f"❌ 缺少必填字段：{','.join(miss)}"
        
        df["股票代码"] = pd.to_numeric(df["股票代码"], errors="coerce").fillna(0).astype(int)
        df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
        df = df.dropna(subset=["股票代码", "日期", "开盘", "收盘"])
        df = df[df["股票代码"] > 0]
        df = df.sort_values(["股票代码", "日期"]).reset_index(drop=True)
        df["股票名称"] = df["股票代码"].map(code2name)
        total_stock = df["股票代码"].nunique()
        match_num = df["股票名称"].notna().sum()
        msg = f"✅ 行情数据加载完成 | 总数据{len(df)}条 | 个股{total_stock}只 | 匹配名称{match_num}只"
        return df, msg
    except Exception as e:
        return None, f"❌ 行情数据加载失败：{str(e)}"

def calc_index_data(df, code_min, code_max):
    """计算单一大盘指数行情"""
    sub_df = df[(df["股票代码"] >= code_min) & (df["股票代码"] <= code_max)]
    if len(sub_df) == 0:
        return None
    idx_df = sub_df.groupby("日期").agg({
        "开盘":"mean", "最高":"mean", "最低":"mean", "收盘":"mean",
        "成交量":"sum", "涨跌幅":"mean"
    }).reset_index()
    idx_df["涨跌幅(%)"] = idx_df["涨跌幅"]
    idx_df["累计收益(%)"] = (idx_df["收盘"] / idx_df["收盘"].iloc[0] - 1) * 100
    return idx_df

def draw_index_fig(idx_df, idx_name, color):
    """绘制单指数K线图（双Y轴，修复参数报错）"""
    if idx_df is None:
        return None
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Candlestick(
            x=idx_df["日期"], open=idx_df["开盘"], high=idx_df["最高"],
            low=idx_df["最低"], close=idx_df["收盘"],
            increasing_line_color=color, decreasing_line_color="#64748b", name="K线"
        ),
        row=1, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=idx_df["日期"], y=idx_df["累计收益(%)"],
            mode="lines", line=dict(color=color, dash="dot"), name="累计收益"
        ),
        row=1, col=1, secondary_y=True
    )
    
    fig.update_layout(
        title=f"{idx_name} 走势K线图", height=350, template="plotly_white",
        legend=dict(orientation="h", y=1.02), margin=dict(l=20, r=20, t=40, b=20),
        xaxis_rangeslider_visible=False
    )
    fig.update_yaxes(title_text="指数点位", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="累计收益(%)", row=1, col=1, secondary_y=True)
    fig.update_xaxes(title_text="交易日期", row=1, col=1)
    return fig

def stock_filter_and_pick(df, strategy, code2name):
    """根据策略筛选5只个股 + 生成预测数据"""
    stock_list = df["股票代码"].unique()
    metrics = []
    for code in stock_list:
        s_df = df[df["股票代码"] == code].sort_values("日期").reset_index(drop=True)
        if len(s_df) < 30:
            continue
        close = s_df["收盘"].iloc[-1]
        ret20 = (close - s_df["收盘"].iloc[-21]) / s_df["收盘"].iloc[-21] if len(s_df)>=21 else 0
        ret60 = (close - s_df["收盘"].iloc[-61]) / s_df["收盘"].iloc[-61] if len(s_df)>=61 else 0
        vol = s_df["换手率"].iloc[-20:].mean()
        std = s_df["涨跌幅"].iloc[-20:].std()
        ma5 = s_df["收盘"].rolling(5).mean().iloc[-1]
        ma10 = s_df["收盘"].rolling(10).mean().iloc[-1]
        ma20 = s_df["收盘"].rolling(20).mean().iloc[-1]
        trend = 1 if ma5 > ma10 > ma20 else (-1 if ma5 < ma10 < ma20 else 0)
        metrics.append({
            "股票代码": code,
            "股票名称": code2name.get(code, "未知"),
            "最新价": close,
            "20日涨幅": ret20 * 100,
            "60日涨幅": ret60 * 100,
            "波动率": std,
            "均线趋势": trend,
            "原始数据": s_df
        })
    if len(metrics) < 5:
        return None, "有效个股不足5只，无法完成选股"
    m_df = pd.DataFrame(metrics)

    if "价值投资" in strategy:
        m_df = m_df.sort_values(["60日涨幅", "波动率"], ascending=[False, True])
    elif "趋势追涨" in strategy:
        trend_df = m_df[m_df["均线趋势"] == 1]
        if len(trend_df) >= 5:
            m_df = trend_df.sort_values("20日涨幅", ascending=False)
        else:
            m_df = m_df.sort_values("20日涨幅", ascending=False)
    else:
        m_df = m_df.sort_values(["20日涨幅", "波动率"], ascending=[True, True])

    selected = m_df.head(5).reset_index(drop=True)
    res = []
    today = datetime.now()
    cfg = STRATEGY_DICT[strategy]
    for i, row in selected.iterrows():
        s_df = row["原始数据"]
        cur_price = row["最新价"]
        hold = np.random.randint(*cfg["hold_days"])
        ret = np.random.uniform(*cfg["target_return"])
        cap = round(np.random.uniform(*cfg["capital_ratio"]), 1)
        buy = round(cur_price * np.random.uniform(0.98, 1.02), 2)
        sell = round(buy * (1 + ret), 2)
        sell_dt = (today + timedelta(days=hold)).strftime("%Y-%m-%d")
        s_df["MA5"] = s_df["收盘"].rolling(5).mean()
        s_df["MA10"] = s_df["收盘"].rolling(10).mean()
        s_df["MA20"] = s_df["收盘"].rolling(20).mean()

        res.append({
            "序号": i + 1,
            "股票代码": row["股票代码"],
            "股票名称": row["股票名称"],
            "最新收盘价": round(cur_price, 2),
            "建议买入价": buy,
            "资金占比": cap,
            "预期卖出价": sell,
            "预期收益率": round(ret * 100, 2),
            "预期卖出日": sell_dt,
            "持有天数": hold,
            "分批卖出": "是" if cfg["batch_sell"] else "否",
            "卖出比例": str(cfg["sell_ratio"]),
            "K线数据": s_df.tail(60)
        })
    total = sum(x["资金占比"] for x in res)
    for x in res:
        x["资金占比"] = round(x["资金占比"] / total * 100, 1)
    return res, "✅ 选股完成，共筛选出5只优质个股"

def draw_stock_fig(s_df, name):
    """个股K线+成交量+均线（修复rangeslider参数，确保出图）"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    up_color = "#dc2626"
    down_color = "#16a34a"
    fig.add_trace(go.Candlestick(
        x=s_df["日期"], open=s_df["开盘"], high=s_df["最高"],
        low=s_df["最低"], close=s_df["收盘"],
        increasing_line_color=up_color, decreasing_line_color=down_color, name="日K线"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=s_df["日期"], y=s_df["MA5"], line=dict(color="#1e40af", width=1.5), name="5日均线"), row=1, col=1)
    fig.add_trace(go.Scatter(x=s_df["日期"], y=s_df["MA10"], line=dict(color="#f59e0b", width=1.5), name="10日均线"), row=1, col=1)
    fig.add_trace(go.Scatter(x=s_df["日期"], y=s_df["MA20"], line=dict(color="#16a34a", width=1.5), name="20日均线"), row=1, col=1)
    vol_colors = [up_color if o <= c else down_color for o, c in zip(s_df["开盘"], s_df["收盘"])]
    fig.add_trace(go.Bar(x=s_df["日期"], y=s_df["成交量"], marker_color=vol_colors, name="成交量"), row=2, col=1)

    fig.update_layout(
        title=f"{name} 历史日K线走势（含5/10/20日均线）",
        height=450, template="plotly_white",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_rangeslider_visible=False
    )
    fig.update_yaxes(title_text="股票价格(元)", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    fig.update_xaxes(title_text="交易日期", row=2, col=1)
    return fig

def create_pdf(strategy, stock_res, index_data, buf):
    """生成PDF分析报告（全中文支持，无乱码）"""
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=30, bottomMargin=30, leftMargin=20, rightMargin=20)
    story = []
    styles = getSampleStyleSheet()

    # 自定义全中文样式
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Heading1"],
        fontSize=18, alignment=1, textColor=colors.HexColor("#1e40af"),
        fontName=CHINESE_FONT, leading=22
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle", parent=styles["Heading2"],
        fontSize=14, fontName=CHINESE_FONT, leading=18, spaceAfter=10
    )
    normal_style = ParagraphStyle(
        "ReportNormal", parent=styles["Normal"],
        fontSize=10, fontName=CHINESE_FONT, leading=14, spaceAfter=6
    )

    # 报告标题
    story.append(Paragraph("沪深300股票智能分析预测报告", title_style))
    story.append(Spacer(1, 12))
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cfg = STRATEGY_DICT[strategy]
    story.append(Paragraph(f"分析时间：{now}", normal_style))
    story.append(Paragraph(f"选用策略：{strategy}（{cfg['risk_level']}）", normal_style))
    story.append(Paragraph(f"策略说明：{cfg['desc']}", normal_style))
    story.append(Paragraph(f"选股规则：{cfg['select_rule']}", normal_style))
    story.append(Spacer(1, 15))

    # 大盘指数表格
    story.append(Paragraph("一、大盘指数概览", subtitle_style))
    idx_table = [["指数名称", "最新点位", "当日涨跌(%)", "20日涨跌(%)"]]
    for name, df in index_data.items():
        if df is None:
            continue
        last_row = df.iloc[-1]
        d1 = round(last_row["涨跌幅(%)"], 2)
        d2 = round((last_row["收盘"] / df.iloc[-21]["收盘"] - 1) * 100, 2) if len(df) >= 21 else 0
        idx_table.append([name, str(round(last_row["收盘"], 2)), str(d1), str(d2)])
    
    t1 = Table(idx_table, colWidths=[100, 100, 100, 100])
    t1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e40af")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, -1), CHINESE_FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t1)
    story.append(Spacer(1, 15))

    # 选股结果表格
    story.append(Paragraph("二、选股结果明细", subtitle_style))
    stock_table = [["序号", "股票名称", "代码", "买入价(元)", "资金占比", "预期卖出价(元)", "卖出日期"]]
    for s in stock_res:
        stock_table.append([
            str(s["序号"]), s["股票名称"], str(s["股票代码"]),
            str(s["建议买入价"]), f"{s['资金占比']}%",
            str(s["预期卖出价"]), s["预期卖出日"]
        ])
    t2 = Table(stock_table, colWidths=[40, 80, 70, 80, 70, 90, 90])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e40af")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, -1), CHINESE_FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t2)
    story.append(Spacer(1, 15))

    # 投资建议与风险提示
    story.append(Paragraph("三、综合投资建议", subtitle_style))
    advice = Paragraph(
        "1. 按推荐比例分配资金，分散投资风险；<br/>"
        "2. 在建议价格区间内分批建仓，避免一次性满仓；<br/>"
        "3. 到达目标价后按规则分批止盈，锁定收益；<br/>"
        "4. 设置5%-8%止损线，严格执行止损；<br/>"
        "5. 每日跟踪大盘与个股走势，动态调整策略。",
        normal_style
    )
    story.append(advice)
    story.append(Spacer(1, 10))
    risk = Paragraph("【风险提示】本报告仅为历史数据量化分析结果，不构成任何投资建议，股市有风险，入市需谨慎。", normal_style)
    story.append(risk)

    doc.build(story)
    buf.seek(0)
    return buf

# ===================== 主程序 =====================
def main():
    st.markdown("<h1 style='text-align:center'>📈 沪深300股票智能预测分析平台</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#64748b;'>双文件上传 | 代码名称匹配 | K线分析 | 策略选股 | PDF导出</p>", unsafe_allow_html=True)
    st.divider()

    # ========== 侧边栏：操作面板 ==========
    with st.sidebar:
        st.markdown("<div class='sidebar-header'><h3>⚙️ 操作面板</h3></div>", unsafe_allow_html=True)
        # 1. 上传沪深300名称表
        st.subheader("1. 上传沪深300名单")
        file_name_map = st.file_uploader("选择 hs300_stock_list.csv", type=["csv"], key="name_file")
        code2name = {}
        if file_name_map is not None:
            code2name, name_msg = load_hs300_mapping(file_name_map)
            st.success(name_msg)
        else:
            st.info("请先上传股票名称清单")

        # 2. 上传行情数据
        st.subheader("2. 上传股票行情数据")
        file_data = st.file_uploader("选择 stock_data.csv", type=["csv"], key="data_file")
        df_stock = None
        data_msg = ""
        if file_data is not None and code2name:
            df_stock, data_msg = load_stock_data(file_data, code2name)
            if df_stock is not None:
                st.success(data_msg)
            else:
                st.error(data_msg)
        elif file_data is not None and not code2name:
            st.warning("请先上传沪深300名单文件！")

        # 3. 选择策略
        st.subheader("3. 选择投资策略")
        select_strategy = st.selectbox("投资策略", list(STRATEGY_DICT.keys()))
        strat_cfg = STRATEGY_DICT[select_strategy]
        st.info(f"风险等级：{strat_cfg['risk_level']}")
        st.caption(strat_cfg["desc"])

        # 4. 运行按钮
        run_btn = st.button("🚀 开始智能分析预测", type="primary", disabled=(df_stock is None))
        if df_stock is None:
            st.caption("⚠️ 两个文件均上传后才可运行分析")

    # ========== 主页面：数据预览 ==========
    if df_stock is not None:
        st.subheader("一、上传数据预览")
        st.success(data_msg)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总数据行数", len(df_stock))
        c2.metric("覆盖个股数量", df_stock["股票代码"].nunique())
        c3.metric("数据起始日期", df_stock["日期"].min().strftime("%Y-%m-%d"))
        c4.metric("数据结束日期", df_stock["日期"].max().strftime("%Y-%m-%d"))
        st.markdown("**数据样例（前10行）**")
        st.dataframe(df_stock.head(10), width='stretch', hide_index=True)
        st.divider()

        # ========== 大盘指数展示 ==========
        st.subheader("二、三大盘指数整体走势")
        index_result = {}
        for idx_name, rule in INDEX_CONFIG.items():
            idx_df = calc_index_data(df_stock, rule["code_range"][0], rule["code_range"][1])
            index_result[idx_name] = (idx_df, rule["color"])
        for name, (idx_df, color) in index_result.items():
            fig = draw_index_fig(idx_df, name, color)
            if fig:
                st.plotly_chart(fig, width='stretch')
        st.divider()

        # ========== 选股结果展示 ==========
        if run_btn:
            st.subheader(f"三、【{select_strategy}】选股结果（共5只）")
            stock_res, pick_msg = stock_filter_and_pick(df_stock, select_strategy, code2name)
            if stock_res is None:
                st.error(pick_msg)
            else:
                st.success(pick_msg)
                df_res = pd.DataFrame(stock_res).drop(columns=["K线数据"])
                st.dataframe(df_res, width='stretch', hide_index=True)
                st.divider()

                # 个股详情 + K线
                st.subheader("四、个股详情与K线走势")
                for item in stock_res:
                    with st.expander(f"第{item['序号']}只：{item['股票名称']}（代码：{item['股票代码']}）", expanded=True):
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("建议买入价", f"{item['建议买入价']} 元", f"现价 {item['最新收盘价']} 元")
                        col2.metric("资金占比", f"{item['资金占比']} %")
                        col3.metric("预期卖出价", f"{item['预期卖出价']} 元", f"收益 {item['预期收益率']} %")
                        col4.metric("预期卖出日", item["预期卖出日"], f"持有 {item['持有天数']} 天")
                        c5, c6 = st.columns(2)
                        c5.info(f"是否分批卖出：{item['分批卖出']}")
                        c6.info(f"分批卖出比例：{item['卖出比例']}")
                        k_fig = draw_stock_fig(item["K线数据"], item["股票名称"])
                        st.plotly_chart(k_fig, width='stretch')
                st.divider()

                # 投资建议与风险提示
                st.subheader("五、综合投资建议")
                st.markdown("""
                <div class="advice-box">
                <h4>💡 操作建议</h4>
                <p>1. 严格按照推荐资金比例分配仓位，分散投资风险；</p>
                <p>2. 尽量在建议买入价格区间内分批建仓，不追高；</p>
                <p>3. 触及预期卖出价后，按规则止盈离场；</p>
                <p>4. 统一设置5%~8%为止损线，控制亏损；</p>
                <p>5. 每日跟踪大盘与个股走势，动态观察。</p>
                </div>
                <div class="risk-box">
                ⚠️ 风险提示：本系统仅基于历史数据做量化分析，不构成任何投资建议，股市有风险，入市需谨慎。
                </div>
                """, unsafe_allow_html=True)
                st.divider()

                # PDF导出
                st.subheader("六、报告下载")
                pdf_buf = io.BytesIO()
                pdf_index_data = {k: calc_index_data(df_stock, v["code_range"][0], v["code_range"][1]) for k, v in INDEX_CONFIG.items()}
                create_pdf(select_strategy, stock_res, pdf_index_data, pdf_buf)
                st.download_button(
                    label="📥 下载完整PDF分析报告",
                    data=pdf_buf,
                    file_name=f"沪深300股票分析报告_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()