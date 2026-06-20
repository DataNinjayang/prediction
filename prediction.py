#!/usr/bin/env python3
"""
沪深300股票智能预测分析平台 V6.1
- 上传CSV数据文件进行分析
- 绝对估值: DCF现金流折现模型 + DDM股利贴现模型
- 相对估值: PE/PB/PS/PCF 行业对比分析
- 多因子综合评分选股
- 精美K线图 + 投资建议 + 主营业务信息
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
import re

st.set_page_config(
    page_title="沪深300股票智能预测分析平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# ===================== 内置主营业务数据 =====================
STOCK_BUSINESS = {
    "600519": "贵州茅台 - 白酒生产销售，高端白酒龙头",
    "600036": "招商银行 - 商业银行，零售银行业务领先",
    "601318": "中国平安 - 保险、银行、投资综合金融",
    "000858": "五粮液 - 白酒生产销售",
    "600900": "长江电力 - 水电发电运营",
    "601012": "隆基绿能 - 光伏硅片及组件",
    "002594": "比亚迪 - 新能源汽车及电池",
    "600276": "恒瑞医药 - 创新药研发及销售",
    "000333": "美的集团 - 家电制造",
    "601888": "中国中免 - 免税商品销售",
    "300750": "宁德时代 - 动力电池及储能",
    "601398": "工商银行 - 大型国有商业银行",
    "601288": "农业银行 - 大型国有商业银行",
    "601939": "建设银行 - 大型国有商业银行",
    "601988": "中国银行 - 大型国有商业银行",
    "600030": "中信证券 - 证券经纪及投行",
    "600887": "伊利股份 - 乳制品生产销售",
    "000568": "泸州老窖 - 白酒生产销售",
    "000001": "平安银行 - 商业银行",
    "600809": "山西汾酒 - 白酒生产销售",
    "002415": "海康威视 - 安防监控设备",
    "601166": "兴业银行 - 股份制商业银行",
    "600309": "万华化学 - 化工新材料",
    "601899": "紫金矿业 - 金铜锌等矿产开采",
    "300059": "东方财富 - 互联网金融服务",
    "601668": "中国建筑 - 建筑工程承包",
    "603259": "药明康德 - 医药研发外包",
    "002352": "顺丰控股 - 快递物流服务",
    "600436": "片仔癀 - 中药制造",
    "601088": "中国神华 - 煤炭开采及发电",
    "002714": "牧原股份 - 生猪养殖",
    "601225": "陕西煤业 - 煤炭开采",
    "600031": "三一重工 - 工程机械制造",
    "000002": "万科A - 房地产开发",
    "600048": "保利发展 - 房地产开发",
    "002230": "科大讯飞 - 人工智能语音",
    "601857": "中国石油 - 石油天然气开采",
    "600028": "中国石化 - 石油炼化及销售",
    "300760": "迈瑞医疗 - 医疗器械",
    "603288": "海天味业 - 调味品制造",
    "600438": "通威股份 - 光伏及农牧",
    "002142": "宁波银行 - 城市商业银行",
    "601601": "中国太保 - 保险业务",
    "601628": "中国人寿 - 人寿保险",
    "601766": "中国中车 - 轨道交通装备",
    "601390": "中国中铁 - 铁路工程建设",
    "601186": "中国铁建 - 铁路工程建设",
    "601800": "中国交建 - 交通基建",
    "601618": "中国中冶 - 冶金工程建设",
    "601669": "中国电建 - 电力工程建设",
    "601989": "中国重工 - 船舶制造",
    "600089": "特变电工 - 输变电设备",
    "600406": "国电南瑞 - 电力自动化",
    "002074": "国轩高科 - 动力电池",
    "300014": "亿纬锂能 - 锂电池",
    "002709": "天赐材料 - 锂电池材料",
    "002460": "赣锋锂业 - 锂化合物",
    "603799": "华友钴业 - 钴镍新材料",
    "300433": "蓝思科技 - 视窗防护玻璃",
    "002475": "立讯精密 - 消费电子连接器",
    "000725": "京东方A - 显示面板",
    "601138": "工业富联 - 电子制造服务",
    "002241": "歌尔股份 - 声学器件",
    "603501": "韦尔股份 - 半导体设计",
    "688981": "中芯国际 - 晶圆代工",
    "600584": "长电科技 - 半导体封测",
    "002371": "北方华创 - 半导体设备",
    "603893": "瑞芯微 - 芯片设计",
    "300782": "卓胜微 - 射频芯片",
    "688012": "中微公司 - 半导体设备",
    "688008": "澜起科技 - 内存接口芯片",
    "603986": "兆易创新 - 存储芯片",
    "300661": "圣邦股份 - 模拟芯片",
    "688396": "华润微 - 功率半导体",
    "600703": "三安光电 - LED芯片",
    "002049": "紫光国微 - 智能安全芯片",
    "300223": "北京君正 - 嵌入式CPU",
    "603160": "汇顶科技 - 指纹识别芯片",
    "300408": "三环集团 - 电子陶瓷",
    "000063": "中兴通讯 - 通信设备",
    "600498": "烽火通信 - 光通信设备",
    "600487": "亨通光电 - 光纤光缆",
    "002281": "光迅科技 - 光器件",
    "300308": "中际旭创 - 光模块",
    "600745": "闻泰科技 - 通讯终端",
    "002156": "通富微电 - 集成电路封测",
    "600460": "士兰微 - 半导体分立器件",
    "300373": "扬杰科技 - 功率半导体",
    "002185": "华天科技 - 集成电路封测",
    "603005": "晶方科技 - 晶圆级封装",
}

# ===================== 数据加载函数 =====================
def load_hs300_mapping(file):
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
    try:
        close = s_df["收盘"].iloc[-1]
        pe = s_df["PE_TTM"].iloc[-1] if "PE_TTM" in s_df.columns and pd.notna(s_df["PE_TTM"].iloc[-1]) else 15
        pb = s_df["PB_MRQ"].iloc[-1] if "PB_MRQ" in s_df.columns and pd.notna(s_df["PB_MRQ"].iloc[-1]) else 1.5
        
        eps_estimate = close / pe if pe > 0 else close / 15
        future_eps = eps_estimate * ((1.05) ** 5)
        target_pe = 15
        wacc = 0.10
        dcf_value = future_eps * target_pe / ((1 + wacc) ** 5)
        
        return {
            "当前价格": round(close, 2),
            "DCF估值": round(dcf_value, 2),
            "估值差异": round((dcf_value - close) / close * 100, 2),
            "WACC": "10%",
            "永续增长率": "3%",
            "盈利增长假设": "5%"
        }
    except Exception:
        return None


def calc_ddm_valuation(s_df, code, name):
    try:
        close = s_df["收盘"].iloc[-1]
        dividend_yield = 0.03
        d1 = close * dividend_yield
        required_return = 0.10
        growth_rate = 0.03
        
        if required_return > growth_rate:
            ddm_value = d1 / (required_return - growth_rate)
        else:
            ddm_value = close
        
        return {
            "当前价格": round(close, 2),
            "DDM估值": round(ddm_value, 2),
            "估值差异": round((ddm_value - close) / close * 100, 2),
            "预期股息": round(d1, 2),
            "要求回报率": "10%",
            "股息增长率": "3%"
        }
    except Exception:
        return None


def calc_relative_valuation(s_df, all_df, code, name):
    try:
        close = s_df["收盘"].iloc[-1]
        pe = s_df["PE_TTM"].iloc[-1] if "PE_TTM" in s_df.columns and pd.notna(s_df["PE_TTM"].iloc[-1]) else None
        pb = s_df["PB_MRQ"].iloc[-1] if "PB_MRQ" in s_df.columns and pd.notna(s_df["PB_MRQ"].iloc[-1]) else None
        ps = s_df["PS_TTM"].iloc[-1] if "PS_TTM" in s_df.columns and pd.notna(s_df["PS_TTM"].iloc[-1]) else None
        pcf = s_df["PCF_TTM"].iloc[-1] if "PCF_TTM" in s_df.columns and pd.notna(s_df["PCF_TTM"].iloc[-1]) else None
        
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
            rating = "强烈低估"; color = "#16a34a"
        elif upside > 5:
            rating = "轻度低估"; color = "#22c55e"
        elif upside > -5:
            rating = "合理估值"; color = "#64748b"
        elif upside > -15:
            rating = "轻度高估"; color = "#f59e0b"
        else:
            rating = "严重高估"; color = "#dc2626"
        
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
    stock_list = df["股票代码"].unique()
    metrics = []
    
    for code in stock_list:
        s_df = df[df["股票代码"] == code].sort_values("日期").reset_index(drop=True)
        if len(s_df) < 30:
            continue
        
        close = s_df["收盘"].iloc[-1]
        ret20 = (close - s_df["收盘"].iloc[-21]) / s_df["收盘"].iloc[-21] if len(s_df) >= 21 else 0
        ret60 = (close - s_df["收盘"].iloc[-61]) / s_df["收盘"].iloc[-61] if len(s_df) >= 61 else 0
        std = s_df["涨跌幅"].iloc[-20:].std()
        
        ma5 = s_df["收盘"].rolling(5).mean().iloc[-1]
        ma10 = s_df["收盘"].rolling(10).mean().iloc[-1]
        ma20 = s_df["收盘"].rolling(20).mean().iloc[-1]
        ma60 = s_df["收盘"].rolling(60).mean().iloc[-1]
        trend = 1 if ma5 > ma10 > ma20 else (-1 if ma5 < ma10 < ma20 else 0)
        
        pe = s_df["PE_TTM"].iloc[-1] if "PE_TTM" in s_df.columns and pd.notna(s_df["PE_TTM"].iloc[-1]) else 20
        pb = s_df["PB_MRQ"].iloc[-1] if "PB_MRQ" in s_df.columns and pd.notna(s_df["PB_MRQ"].iloc[-1]) else 2
        
        value_score = max(0, min(100, (30 - pe) / 30 * 100)) if pe > 0 else 50
        value_score += max(0, min(100, (3 - pb) / 3 * 100)) if pb > 0 else 50
        value_score /= 2
        
        momentum_score = max(0, min(100, ret20 * 100 + 50))
        quality_score = max(0, min(100, 100 - std * 10))
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
    s_df = s_df.copy()
    s_df["MA5"] = s_df["收盘"].rolling(5).mean()
    s_df["MA10"] = s_df["收盘"].rolling(10).mean()
    s_df["MA20"] = s_df["收盘"].rolling(20).mean()
    s_df["MA60"] = s_df["收盘"].rolling(60).mean()
    
    exp1 = s_df["收盘"].ewm(span=12).mean()
    exp2 = s_df["收盘"].ewm(span=26).mean()
    s_df["MACD_DIF"] = exp1 - exp2
    s_df["MACD_DEA"] = s_df["MACD_DIF"].ewm(span=9).mean()
    s_df["MACD_HIST"] = 2 * (s_df["MACD_DIF"] - s_df["MACD_DEA"])
    
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
    
    fig.add_trace(go.Candlestick(
        x=s_df["日期"], open=s_df["开盘"], high=s_df["最高"],
        low=s_df["最低"], close=s_df["收盘"],
        increasing_line_color=up_color, decreasing_line_color=down_color, name="日K"
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=s_df["日期"], y=s_df["MA5"], line=dict(color="#1e40af", width=1), name="MA5"), row=1, col=1)
    fig.add_trace(go.Scatter(x=s_df["日期"], y=s_df["MA10"], line=dict(color="#f59e0b", width=1), name="MA10"), row=1, col=1)
    fig.add_trace(go.Scatter(x=s_df["日期"], y=s_df["MA20"], line=dict(color="#16a34a", width=1), name="MA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=s_df["日期"], y=s_df["MA60"], line=dict(color="#9333ea", width=1.5), name="MA60"), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=s_df["日期"], y=s_df["MACD_DIF"], line=dict(color="#1e40af", width=1), name="DIF"), row=2, col=1)
    fig.add_trace(go.Scatter(x=s_df["日期"], y=s_df["MACD_DEA"], line=dict(color="#f59e0b", width=1), name="DEA"), row=2, col=1)
    colors_macd = [up_color if h >= 0 else down_color for h in s_df["MACD_HIST"]]
    fig.add_trace(go.Bar(x=s_df["日期"], y=s_df["MACD_HIST"], marker_color=colors_macd, name="MACD柱"), row=2, col=1)
    
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


# ===================== 投资建议生成 =====================
def generate_investment_advice(comp, dcf, ddm, relative, row):
    """生成个性化投资建议"""
    advice = []
    
    # 基于综合评级
    if comp:
        rating = comp["评级"]
        upside = float(comp["上涨空间"].replace("%", ""))
        
        if "低估" in rating:
            advice.append(f"✅ **估值优势**: 综合估值显示该股票{rating}，预期上涨空间约{comp['上涨空间']}，具备中长期投资价值。")
        elif "高估" in rating:
            advice.append(f"⚠️ **估值风险**: 综合估值显示该股票{rating}，当前价格可能透支未来增长，建议谨慎。")
        else:
            advice.append(f"➖ **估值合理**: 当前估值处于合理区间，可关注技术面突破机会。")
    
    # 基于DCF
    if dcf and dcf["估值差异"] > 10:
        advice.append(f"📊 **DCF视角**: 现金流折现估值{dcf['DCF估值']}元，较当前价格低估{dcf['估值差异']:.1f}%，反映企业内在价值被低估。")
    elif dcf and dcf["估值差异"] < -10:
        advice.append(f"📊 **DCF视角**: 现金流折现估值{dcf['DCF估值']}元，较当前价格高估{abs(dcf['估值差异']):.1f}%，需谨慎评估增长假设。")
    
    # 基于DDM
    if ddm and ddm["估值差异"] > 10:
        advice.append(f"💰 **DDM视角**: 股利贴现估值{ddm['DDM估值']}元，股息回报吸引力较强，适合价值型投资者。")
    
    # 基于相对估值
    pe_diff = relative.get("PE_TTM", {}).get("差异%", 0)
    pb_diff = relative.get("PB_MRQ", {}).get("差异%", 0)
    if pe_diff > 20:
        advice.append(f"📈 **PE对比**: 当前PE较行业中位数低{pe_diff:.1f}%，估值具有相对优势。")
    elif pe_diff < -20:
        advice.append(f"📉 **PE对比**: 当前PE较行业中位数高{abs(pe_diff):.1f}%，相对行业偏贵。")
    
    # 基于技术面
    trend = row.get("均线趋势", "")
    if trend == "多头":
        advice.append("📉 **技术形态**: 均线呈多头排列，短期趋势向好，可考虑顺势操作。")
    elif trend == "空头":
        advice.append("📉 **技术形态**: 均线呈空头排列，短期承压，建议等待企稳信号。")
    
    # 波动率建议
    std = row.get("波动率", 0)
    if std > 3:
        advice.append(f"📊 **波动提示**: 近20日波动率{std:.2f}%较高，注意控制仓位，建议分批建仓。")
    else:
        advice.append(f"📊 **波动提示**: 近20日波动率{std:.2f}%较低，股价相对稳定。")
    
    return "\n\n".join(advice)


# ===================== 主程序 =====================
def main():
    st.markdown("<h1 style='text-align:center'>📈 沪深300股票智能预测分析平台</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#64748b;'>绝对估值(DCF+DDM) + 相对估值(PE/PB/PS) | 多因子选股 | K线分析 | 价格预测 | 投资建议</p>", unsafe_allow_html=True)
    st.divider()
    
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
                    
                    # 主营业务
                    business = STOCK_BUSINESS.get(code, f"{name} - 具体业务信息请查阅公司年报")
                    st.markdown(f"""
                    <div style="background: #f0fdf4; padding: 12px; border-radius: 8px; border-left: 4px solid #16a34a; margin: 10px 0;">
                    <strong>🏢 主营业务:</strong> {business}
                    </div>
                    """, unsafe_allow_html=True)
                    
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
                    pred_result = monte_carlo_prediction(s_df, days=30, simulations=500)
                    if pred_result:
                        pred_fig, target_price, upside = pred_result
                        st.plotly_chart(pred_fig, use_container_width=True)
                        st.info(f"预期30天后价格: **{target_price}** | 预期涨跌: **{upside}%**")
                    
                    # 投资建议
                    st.markdown("#### 💡 个性化投资建议")
                    advice = generate_investment_advice(comp, dcf, ddm, relative, row)
                    st.markdown(f"""
                    <div class="advice-box">
                    {advice}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 未来是否适合投资
                    if comp:
                        rating = comp["评级"]
                        if "低估" in rating:
                            st.success(f"✅ **未来投资判断**: 该股票当前{rating}，综合估值有{comp['上涨空间']}的上涨空间，**适合中长期投资**。建议分批建仓，设置止损位在{row['最新价']*0.92:.2f}元（-8%）附近。")
                        elif "高估" in rating:
                            st.error(f"❌ **未来投资判断**: 该股票当前{rating}，估值偏高，**不建议新建仓位**。已持仓者建议逐步减仓，或等待回调至合理估值区间再考虑。")
                        else:
                            st.info(f"➖ **未来投资判断**: 该股票估值处于合理区间，**可观望或轻仓参与**。建议等待技术面突破或估值进一步下移后再加大仓位。")
                    
                    st.divider()
            
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
