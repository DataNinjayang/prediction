#!/usr/bin/env python3
"""
沪深300股票智能预测分析平台 V6.0
- 绝对估值: DCF现金流折现模型 + DDM股利贴现模型
- 相对估值: PE/PB/PS/PCF 行业对比分析
- 多因子综合评分选股
- 支持上传CSV数据文件
- 精美K线图 + 投资建议
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
import re
import os

st.set_page_config(
    page_title="沪深300股票智能预测分析平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== 自定义CSS =====================
st.markdown("""
<style>
.main {background-color: #f8fafc; font-family: 'Microsoft YaHei', sans-serif;}
.block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px; margin: 0 auto;}
.sidebar-header {background: linear-gradient(135deg, #1e40af, #3b82f6); color: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; text-align: center;}
.stButton>button {background-color: #1e40af; color: white; border-radius: 8px; font-weight: 600; border: none; width: 100%;}
.advice-box {background: #eff6ff; padding: 20px; border-radius: 10px; border-left: 4px solid #1e40af; margin:15px 0;}
.risk-box {background: #fef2f2; padding: 15px; border-radius: 8px; border-left: 4px solid #dc2626; margin:10px 0;}
.valuation-card {background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin: 10px 0;}
.metric-good {color: #16a34a; font-weight: bold;}
.metric-bad {color: #dc2626; font-weight: bold;}
.metric-neutral {color: #64748b;}
hr {border-color: #e2e8f0; margin:25px 0;}
</style>
""", unsafe_allow_html=True)

np.random.seed(42)

# ===================== 工具函数 =====================
def load_hs300_mapping(file):
    """加载沪深300代码-名称映射"""
    try:
        df = pd.read_csv(file)
        df["纯代码"] = df["code"].apply(lambda x: re.sub(r"^(sh\.|sz\.)", "", str(x)))
        df["纯代码"] = pd.to_numeric(df["纯代码"], errors="coerce")
        df = df.dropna(subset=["纯代码"])
        code2name = dict(zip(df["纯代码"].astype(int), df["code_name"]))
        return code2name, f"沪深300名单加载成功，共 {len(code2name)} 只个股"
    except Exception as e:
        return {}, f"名单加载失败: {str(e)}"


def load_stock_data(file, code2name):
    """加载股票行情数据"""
    try:
        df = pd.read_csv(file)
        required_cols = ["股票代码", "日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌额", "换手率", "涨跌幅"]
        miss = [c for c in required_cols if c not in df.columns]
        if miss:
            return None, f"缺少必填字段: {','.join(miss)}"
        
        df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
        df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
        df = df.dropna(subset=["股票代码", "日期", "开盘", "收盘"])
        df = df.sort_values(["股票代码", "日期"]).reset_index(drop=True)
        df["股票名称"] = df["股票代码"].map(code2name)
        
        # 转换估值字段
        for col in ["PE_TTM", "PB_MRQ", "PS_TTM", "PCF_TTM"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        total_stock = df["股票代码"].nunique()
        match_num = df["股票名称"].notna().sum()
        return df, f"行情数据加载完成 | 总数据{len(df)}条 | 个股{total_stock}只 | 匹配名称{match_num}只"
    except Exception as e:
        return None, f"行情数据加载失败: {str(e)}"


# ===================== 估值模型 =====================
def calc_dcf_valuation(s_df, code, name):
    """DCF现金流折现模型"""
    try:
        close = s_df["收盘"].iloc[-1]
        avg_profit_growth = 0.05  # 假设5%平均盈利增长
        wacc = 0.10  # 假设10%加权平均资本成本
        terminal_growth = 0.03  # 永续增长率3%
        
        # 简化DCF: 使用当前价格反推隐含增长率
        # 假设未来5年FCF = 当前价格 * (1+g)^n
        # 这里用简化版本
        fair_value = close * (1 + avg_profit_growth) / (wacc - terminal_growth) * (wacc - terminal_growth)
        
        # 更实际的简化估值
        pe = s_df["PE_TTM"].iloc[-1] if "PE_TTM" in s_df.columns and pd.notna(s_df["PE_TTM"].iloc[-1]) else 15
        pb = s_df["PB_MRQ"].iloc[-1] if "PB_MRQ" in s_df.columns and pd.notna(s_df["PB_MRQ"].iloc[-1]) else 1.5
        
        # DCF简化: 基于PE和盈利增长
        eps_estimate = close / pe if pe > 0 else close / 15
        future_eps = eps_estimate * ((1 + avg_profit_growth) ** 5)
        target_pe = 15  # 目标PE
        dcf_value = future_eps * target_pe / ((1 + wacc) ** 5)
        
        return {
            "当前价格": round(close, 2),
            "DCF估值": round(dcf_value, 2),
            "估值差异": round((dcf_value - close) / close * 100, 2),
            "WACC": f"{wacc*100:.0f}%",
            "永续增长率": f"{terminal_growth*100:.0f}%",
            "盈利增长假设": f"{avg_profit_growth*100:.0f}%"
        }
    except Exception:
        return None


def calc_ddm_valuation(s_df, code, name):
    """DDM股利贴现模型 (Gordon Growth Model)"""
    try:
        close = s_df["收盘"].iloc[-1]
        # 假设股息率2-4%
        dividend_yield = 0.03
        d1 = close * dividend_yield  # 预期下一年股息
        required_return = 0.10  # 要求回报率
        growth_rate = 0.03  # 股息增长率
        
        if required_return > growth_rate:
            ddm_value = d1 / (required_return - growth_rate)
        else:
            ddm_value = close
        
        return {
            "当前价格": round(close, 2),
            "DDM估值": round(ddm_value, 2),
            "估值差异": round((ddm_value - close) / close * 100, 2),
            "预期股息": round(d1, 2),
            "要求回报率": f"{required_return*100:.0f}%",
            "股息增长率": f"{growth_rate*100:.0f}%"
        }
    except Exception:
        return None


def calc_relative_valuation(s_df, all_df, code, name):
    """相对估值: PE/PB/PS/PCF 行业对比"""
    try:
        close = s_df["收盘"].iloc[-1]
        
        # 获取当前股票的估值指标
        pe = s_df["PE_TTM"].iloc[-1] if "PE_TTM" in s_df.columns and pd.notna(s_df["PE_TTM"].iloc[-1]) else None
        pb = s_df["PB_MRQ"].iloc[-1] if "PB_MRQ" in s_df.columns and pd.notna(s_df["PB_MRQ"].iloc[-1]) else None
        ps = s_df["PS_TTM"].iloc[-1] if "PS_TTM" in s_df.columns and pd.notna(s_df["PS_TTM"].iloc[-1]) else None
        pcf = s_df["PCF_TTM"].iloc[-1] if "PCF_TTM" in s_df.columns and pd.notna(s_df["PCF_TTM"].iloc[-1]) else None
        
        # 计算行业均值（使用所有股票的最新数据）
        latest = all_df.groupby("股票代码").last()
        pe_mean = latest["PE_TTM"].replace(0, np.nan).median() if "PE_TTM" in latest.columns else None
        pb_mean = latest["PB_MRQ"].replace(0, np.nan).median() if "PB_MRQ" in latest.columns else None
        ps_mean = latest["PS_TTM"].replace(0, np.nan).median() if "PS_TTM" in latest.columns else None
        pcf_mean = latest["PCF_TTM"].replace(0, np.nan).median() if "PCF_TTM" in latest.columns else None
        
        results = {}
        
        if pe is not None and pe_mean is not None and pe_mean > 0:
            pe_fair = close * (pe_mean / pe) if pe > 0 else close
            results["PE_TTM"] = {"当前": round(pe, 2), "行业中位数": round(pe_mean, 2), 
                                  "估值": round(pe_fair, 2), "差异%": round((pe_mean - pe) / pe * 100, 2)}
        
        if pb is not None and pb_mean is not None and pb_mean > 0:
            pb_fair = close * (pb_mean / pb) if pb > 0 else close
            results["PB_MRQ"] = {"当前": round(pb, 2), "行业中位数": round(pb_mean, 2), 
                                  "估值": round(pb_fair, 2), "差异%": round((pb_mean - pb) / pb * 100, 2)}
        
        if ps is not None and ps_mean is not None and ps_mean > 0:
            ps_fair = close * (ps_mean / ps) if ps > 0 else close
            results["PS_TTM"] = {"当前": round(ps, 2), "行业中位数": round(ps_mean, 2), 
                                  "估值": round(ps_fair, 2), "差异%": round((ps_mean - ps) / ps * 100, 2)}
        
        if pcf is not None and pcf_mean is not None and pcf_mean > 0:
            pcf_fair = close * (pcf_mean / pcf) if pcf > 0 else close
            results["PCF_TTM"] = {"当前": round(pcf, 2), "行业中位数": round(pcf_mean, 2), 
                                   "估值": round(pcf_fair, 2), "差异%": round((pcf_mean - pcf) / pcf * 100, 2)}
        
        return results
    except Exception:
        return {}


def comprehensive_valuation(dcf, ddm, relative):
    """综合估值分析"""
    valuations = []
    
    if dcf:
        valuations.append(dcf["DCF估值"])
    if ddm:
        valuations.append(ddm["DDM估值"])
    
    for key, val in relative.items():
        if "估值" in val:
            valuations.append(val["估值"])
    
    if not valuations:
        return None
    
    avg_valuation = np.mean(valuations)
    min_valuation = np.min(valuations)
    max_valuation = np.max(valuations)
    
    current_price = dcf["当前价格"] if dcf else (ddm["当前价格"] if ddm else None)
    
    if current_price:
        upside = (avg_valuation - current_price) / current_price * 100
        if upside > 15:
            rating = "强烈低估"
            color = "#16a34a"
        elif upside > 5:
            rating = "轻度低估"
            color = "#22c55e"
        elif upside > -5:
            rating = "合理估值"
            color = "#64748b"
        elif upside > -15:
            rating = "轻度高估"
            color = "#f59e0b"
        else:
            rating = "严重高估"
            color = "#dc2626"
        
        return {
            "综合估值": round(avg_valuation, 2),
            "估值区间": f"{round(min_valuation, 2)} - {round(max_valuation, 2)}",
            "当前价格": current_price,
            "上涨空间": f"{upside:.1f}%",
            "评级": rating,
            "评级颜色": color
        }
    return None


# ===================== 选股与图表 =====================
def multi_factor_stock_pick(df, code2name):
    """多因子综合评分选股"""
    stock_list = df["股票代码"].unique()
    metrics = []
    
    for code in stock_list:
        s_df = df[df["股票代码"] == code].sort_values("日期").reset_index(drop=True)
        if len(s_df) < 30:
            continue
        
        close = s_df["收盘"].iloc[-1]
        ret20 = (close - s_df["收盘"].iloc[-21]) / s_df["收盘"].iloc[-21] if len(s_df) >= 21 else 0
        ret60 = (close - s_df["收盘"].iloc[-61]) / s_df["收盘"].iloc[-61] if len(s_df) >= 61 else 0
        vol = s_df["换手率"].iloc[-20:].mean()
        std = s_df["涨跌幅"].iloc[-20:].std()
        
        ma5 = s_df["收盘"].rolling(5).mean().iloc[-1]
        ma10 = s_df["收盘"].rolling(10).mean().iloc[-1]
        ma20 = s_df["收盘"].rolling(20).mean().iloc[-1]
        ma60 = s_df["收盘"].rolling(60).mean().iloc[-1]
        
        trend = 1 if ma5 > ma10 > ma20 else (-1 if ma5 < ma10 < ma20 else 0)
        
        # 估值因子
        pe = s_df["PE_TTM"].iloc[-1] if "PE_TTM" in s_df.columns and pd.notna(s_df["PE_TTM"].iloc[-1]) else 20
        pb = s_df["PB_MRQ"].iloc[-1] if "PB_MRQ" in s_df.columns and pd.notna(s_df["PB_MRQ"].iloc[-1]) else 2
        
        # 多因子评分
        value_score = max(0, min(100, (30 - pe) / 30 * 100)) if pe > 0 else 50  # PE越低越好
        value_score += max(0, min(100, (3 - pb) / 3 * 100)) if pb > 0 else 50  # PB越低越好
        value_score /= 2
        
        momentum_score = max(0, min(100, ret20 * 100 + 50))  # 20日涨幅
        quality_score = max(0, min(100, 100 - std * 10))  # 波动率越低越好
        trend_score = 80 if trend == 1 else (40 if trend == 0 else 20)
        
        total_score = value_score * 0.30 + momentum_score * 0.25 + quality_score * 0.25 + trend_score * 0.20
        
        metrics.append({
            "股票代码": code,
            "股票名称": code2name.get(code, "未知"),
            "最新价": round(close, 2),
            "PE_TTM": round(pe, 2) if pd.notna(pe) else "N/A",
            "PB_MRQ": round(pb, 2) if pd.notna(pb) else "N/A",
            "20日涨幅": round(ret20 * 100, 2),
            "60日涨幅": round(ret60 * 100, 2),
            "波动率": round(std, 2),
            "均线趋势": "多头" if trend == 1 else ("空头" if trend == -1 else "震荡"),
            "价值评分": round(value_score, 1),
            "动量评分": round(momentum_score, 1),
            "质量评分": round(quality_score, 1),
            "趋势评分": round(trend_score, 1),
            "综合评分": round(total_score, 1),
            "原始数据": s_df
        })
    
    if len(metrics) < 5:
        return None, "有效个股不足5只"
    
    m_df = pd.DataFrame(metrics)
    selected = m_df.nlargest(5, "综合评分").reset_index(drop=True)
    return selected, "选股完成"


def draw_stock_fig(s_df, name, code):
    """个股K线+均线+成交量+MACD+RSI"""
    s_df = s_df.copy()
    s_df["MA5"] = s_df["收盘"].rolling(5).mean()
    s_df["MA10"] = s_df["收盘"].rolling(10).mean()
    s_df["MA20"] = s_df["收盘"].rolling(20).mean()
    s_df["MA60"] = s_df["收盘"].rolling(60).mean()
    
    # MACD
    exp1 = s_df["收盘"].ewm(span=12).mean()
    exp2 = s_df["收盘"].ewm(span=26).mean()
    s_df["MACD_DIF"] = exp1 - exp2
    s_df["MACD_DEA"] = s_df["MACD_DIF"].ewm(span=9).mean()
    s_df["MACD_HIST"] = 2 * (s_df["MACD_DIF"] - s_df["MACD_DEA"])
    
    # RSI
    delta = s_df["收盘"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    s_df["RSI"] = 100 - (100 / (1 + rs))
    
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=(f"{name}({code}) K线图", "MACD", "RSI")
    )
    
    up_color, down_color = "#dc2626", "#16a34a"
    
    # K线
    fig.add_trace(go.Candlestick(
        x=s_df["日期"], open=s_df["开盘"], high=s_df["最高"],
        low=s_df["最低"], close=s_df["收盘"],
        increasing_line_color=up_color, decreasing_line_color=down_color, name="日K"
    ), row=1, col=1)
    
    # 均线
    fig.add_trace(go.Scatter(x=s_df["日期"], y=s_df["MA5"], line=dict(color="#1e40af", width=1), name="MA5"), row=1, col=1)
    fig.add_trace(go.Scatter(x=s_df["日期"], y=s_df["MA10"], line=dict(color="#f59e0b", width=1), name="MA10"), row=1, col=1)
    fig.add_trace(go.Scatter(x=s_df["日期"], y=s_df["MA20"], line=dict(color="#16a34a", width=1), name="MA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=s_df["日期"], y=s_df["MA60"], line=dict(color="#9333ea", width=1.5), name="MA60"), row=1, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=s_df["日期"], y=s_df["MACD_DIF"], line=dict(color="#1e40af", width=1), name="DIF"), row=2, col=1)
    fig.add_trace(go.Scatter(x=s_df["日期"], y=s_df["MACD_DEA"], line=dict(color="#f59e0b", width=1), name="DEA"), row=2, col=1)
    colors_macd = [up_color if h >= 0 else down_color for h in s_df["MACD_HIST"]]
    fig.add_trace(go.Bar(x=s_df["日期"], y=s_df["MACD_HIST"], marker_color=colors_macd, name="MACD柱"), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=s_df["日期"], y=s_df["RSI"], line=dict(color="#8b5cf6", width=1.5), name="RSI"), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#dc2626", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#16a34a", row=3, col=1)
    
    fig.update_layout(
        height=700, template="plotly_white",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_rangeslider_visible=False
    )
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    
    return fig


def monte_carlo_prediction(s_df, days=30, simulations=500):
    """蒙特卡洛价格预测"""
    returns = s_df["涨跌幅"].dropna() / 100
    if len(returns) < 20:
        return None
    
    mu = returns.mean()
    sigma = returns.std()
    last_price = s_df["收盘"].iloc[-1]
    
    paths = []
    for _ in range(simulations):
        prices = [last_price]
        for _ in range(days):
            prices.append(prices[-1] * (1 + np.random.normal(mu, sigma)))
        paths.append(prices)
    
    paths = np.array(paths)
    mean_path = paths.mean(axis=0)
    p5 = np.percentile(paths, 5, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    
    future_dates = [s_df["日期"].iloc[-1] + timedelta(days=i) for i in range(days + 1)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_dates, y=mean_path, mode="lines", name="预期价格", line=dict(color="#1e40af", width=2)))
    fig.add_trace(go.Scatter(x=future_dates, y=p95, mode="lines", name="95%置信区间", line=dict(color="#93c5fd", dash="dash")))
    fig.add_trace(go.Scatter(x=future_dates, y=p5, mode="lines", name="5%置信区间", line=dict(color="#93c5fd", dash="dash"), fill="tonexty"))
    fig.add_vline(x=s_df["日期"].iloc[-1], line_dash="dot", line_color="#64748b")
    
    fig.update_layout(
        title=f"未来{days}天价格预测 (蒙特卡洛模拟 {simulations}次)",
        height=400, template="plotly_white",
        xaxis_title="日期", yaxis_title="价格"
    )
    
    target_price = mean_path[-1]
    upside = (target_price - last_price) / last_price * 100
    return fig, round(target_price, 2), round(upside, 2)


# ===================== 主程序 =====================
def main():
    st.markdown("<h1 style='text-align:center'>📈 沪深300股票智能预测分析平台</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#64748b;'>绝对估值(DCF+DDM) + 相对估值(PE/PB/PS) | 多因子选股 | K线分析 | 价格预测</p>", unsafe_allow_html=True)
    st.divider()
    
    # ========== 侧边栏 ==========
    with st.sidebar:
        st.markdown("<div class='sidebar-header'><h3>⚙️ 操作面板</h3></div>", unsafe_allow_html=True)
        
        st.subheader("1. 上传沪深300名单")
        file_name_map = st.file_uploader("选择 hs300_stock_list.csv", type=["csv"], key="name_file")
        code2name = {}
        if file_name_map is not None:
            code2name, name_msg = load_hs300_mapping(file_name_map)
            if code2name:
                st.success(name_msg)
            else:
                st.error(name_msg)
        else:
            st.info("请先上传股票名称清单")
        
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
        
        run_btn = st.button("🚀 开始智能分析", type="primary", disabled=(df_stock is None))
        if df_stock is None:
            st.caption("⚠️ 两个文件均上传后才可运行分析")
    
    # ========== 主页面 ==========
    if df_stock is not None:
        st.subheader("一、数据概览")
        st.success(data_msg)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总数据行数", f"{len(df_stock):,}")
        c2.metric("覆盖个股数", df_stock["股票代码"].nunique())
        c3.metric("数据起始日", df_stock["日期"].min().strftime("%Y-%m-%d"))
        c4.metric("数据截止日", df_stock["日期"].max().strftime("%Y-%m-%d"))
        
        with st.expander("查看数据样例"):
            st.dataframe(df_stock.head(10), use_container_width=True, hide_index=True)
        st.divider()
        
        if run_btn:
            # ========== 智能选股 ==========
            st.subheader("二、多因子智能选股（Top 5）")
            with st.spinner("正在进行多因子评分..."):
                selected_df, pick_msg = multi_factor_stock_pick(df_stock, code2name)
            
            if selected_df is None:
                st.error(pick_msg)
                return
            
            st.success(pick_msg)
            display_df = selected_df.drop(columns=["原始数据"])
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            st.divider()
            
            # ========== 个股深度分析 ==========
            st.subheader("三、个股深度估值分析")
            
            for idx, row in selected_df.iterrows():
                s_df = row["原始数据"]
                code = row["股票代码"]
                name = row["股票名称"]
                
                with st.expander(f"#{idx+1} {name} ({code}) | 综合评分: {row['综合评分']}", expanded=(idx==0)):
                    # 基本信息
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("最新价", f"{row['最新价']}")
                    c2.metric("PE_TTM", row['PE_TTM'])
                    c3.metric("PB_MRQ", row['PB_MRQ'])
                    c4.metric("20日涨幅", f"{row['20日涨幅']}%")
                    c5.metric("均线趋势", row['均线趋势'])
                    
                    # 绝对估值
                    st.markdown("#### 📊 绝对估值")
                    col_dcf, col_ddm = st.columns(2)
                    
                    dcf = calc_dcf_valuation(s_df, code, name)
                    if dcf:
                        with col_dcf:
                            st.markdown("<div class='valuation-card'>", unsafe_allow_html=True)
                            st.markdown(f"**DCF现金流折现模型**")
                            st.markdown(f"当前价格: **{dcf['当前价格']}**")
                            st.markdown(f"DCF估值: **<span class='metric-good'>{dcf['DCF估值']}</span>**" if dcf['估值差异'] > 0 else f"DCF估值: **<span class='metric-bad'>{dcf['DCF估值']}</span>**", unsafe_allow_html=True)
                            st.markdown(f"估值差异: **{dcf['估值差异']}%**")
                            st.markdown(f"WACC: {dcf['WACC']} | 永续增长率: {dcf['永续增长率']}")
                            st.markdown("</div>", unsafe_allow_html=True)
                    
                    ddm = calc_ddm_valuation(s_df, code, name)
                    if ddm:
                        with col_ddm:
                            st.markdown("<div class='valuation-card'>", unsafe_allow_html=True)
                            st.markdown(f"**DDM股利贴现模型**")
                            st.markdown(f"当前价格: **{ddm['当前价格']}**")
                            st.markdown(f"DDM估值: **<span class='metric-good'>{ddm['DDM估值']}</span>**" if ddm['估值差异'] > 0 else f"DDM估值: **<span class='metric-bad'>{ddm['DDM估值']}</span>**", unsafe_allow_html=True)
                            st.markdown(f"估值差异: **{ddm['估值差异']}%**")
                            st.markdown(f"预期股息: {ddm['预期股息']} | 要求回报率: {ddm['要求回报率']}")
                            st.markdown("</div>", unsafe_allow_html=True)
                    
                    # 相对估值
                    st.markdown("#### 📈 相对估值（行业对比）")
                    relative = calc_relative_valuation(s_df, df_stock, code, name)
                    if relative:
                        rel_cols = st.columns(len(relative))
                        for i, (key, val) in enumerate(relative.items()):
                            with rel_cols[i]:
                                st.markdown("<div class='valuation-card'>", unsafe_allow_html=True)
                                st.markdown(f"**{key}**")
                                st.markdown(f"当前: {val['当前']}")
                                st.markdown(f"行业中位数: {val['行业中位数']}")
                                st.markdown(f"合理估值: **<span class='metric-good'>{val['估值']}</span>**" if val['差异%'] > 0 else f"合理估值: **<span class='metric-bad'>{val['估值']}</span>**", unsafe_allow_html=True)
                                st.markdown(f"差异: **{val['差异%']}%**")
                                st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.info("无可用的估值指标数据进行行业对比")
                    
                    # 综合估值结论
                    comp = comprehensive_valuation(dcf, ddm, relative)
                    if comp:
                        st.markdown(f"""
                        <div style="background: {comp['评级颜色']}15; padding: 15px; border-radius: 10px; border-left: 4px solid {comp['评级颜色']}; margin: 15px 0;">
                        <h4 style="color: {comp['评级颜色']}; margin: 0;">综合估值结论: {comp['评级']}</h4>
                        <p style="margin: 5px 0;">综合估值: <strong>{comp['综合估值']}</strong> | 估值区间: {comp['估值区间']} | 当前价格: {comp['当前价格']}</p>
                        <p style="margin: 5px 0;">预期上涨空间: <strong>{comp['上涨空间']}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # K线图
                    st.markdown("#### 📉 K线走势与技术指标")
                    k_fig = draw_stock_fig(s_df, name, code)
                    st.plotly_chart(k_fig, use_container_width=True)
                    
                    # 价格预测
                    st.markdown("#### 🔮 蒙特卡洛价格预测")
                    pred_result = monte_carlo_prediction(s_df)
                    if pred_result:
                        pred_fig, target_price, upside = pred_result
                        st.plotly_chart(pred_fig, use_container_width=True)
                        st.info(f"预期{days}天后价格: **{target_price}** | 预期涨跌: **{upside}%**")
                    
                    st.divider()
            
            # ========== 投资建议 ==========
            st.subheader("四、综合投资建议")
            st.markdown("""
            <div class="advice-box">
            <h4>💡 操作建议</h4>
            <p>1. <strong>绝对估值</strong>：DCF和DDM模型从企业内在价值角度评估，适合长期投资者参考；</p>
            <p>2. <strong>相对估值</strong>：PE/PB/PS行业对比从市场定价角度评估，适合判断当前估值高低；</p>
            <p>3. <strong>综合评级</strong>：结合绝对和相对估值，给出综合判断，建议关注"强烈低估"和"轻度低估"个股；</p>
            <p>4. <strong>分批建仓</strong>：在建议价格区间内分批买入，避免一次性满仓；</p>
            <p>5. <strong>设置止损</strong>：统一设置5%~8%为止损线，严格执行风险控制。</p>
            </div>
            <div class="risk-box">
            ⚠️ 风险提示：本系统仅基于历史数据做量化分析，估值模型使用简化假设，不构成任何投资建议，股市有风险，入市需谨慎。
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
