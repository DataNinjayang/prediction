"""
沪深300股票智能预测分析平台 V2.0
- 使用 akshare 获取实时沪深300成分股及历史数据
- 整合 DDM 股利贴现模型进行基本面估值
- 获取个股行业分类、主营业务、财务指标
- 多维度数据分析与精美可视化
- 适配 Streamlit Cloud 部署
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import akshare as ak
from datetime import datetime, timedelta
import time
import warnings
import io
import re
from scipy import stats

warnings.filterwarnings("ignore")

# ===================== 页面全局配置 =====================
st.set_page_config(
    page_title="沪深300股票智能预测分析平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== 自定义CSS样式 =====================
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&display=swap');

/* 全局样式 */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1440px;
    margin: 0 auto;
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border-radius: 16px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.06);
}

/* 标题区域 */
.hero-title {
    text-align: center;
    padding: 30px 0 10px;
}
.hero-title h1 {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
}
.hero-title p {
    color: #64748b;
    font-size: 1rem;
    letter-spacing: 1px;
}

/* 侧边栏 */
.sidebar-header {
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
    color: white;
    padding: 18px 15px;
    border-radius: 12px;
    margin-bottom: 20px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(30,64,175,0.3);
}
.sidebar-header h3 {
    margin: 0;
    font-size: 1.1rem;
}

/* 按钮 */
.stButton > button {
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 12px 24px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(30,64,175,0.3);
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(30,64,175,0.4);
}
.stButton > button:disabled {
    background: #94a3b8;
    box-shadow: none;
    transform: none;
}

/* 信息框 */
.advice-box {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    padding: 24px;
    border-radius: 12px;
    border-left: 5px solid #1e40af;
    margin: 15px 0;
    box-shadow: 0 2px 8px rgba(30,64,175,0.1);
}
.risk-box {
    background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    padding: 20px;
    border-radius: 12px;
    border-left: 5px solid #dc2626;
    margin: 10px 0;
    box-shadow: 0 2px 8px rgba(220,38,38,0.1);
}
.info-box {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    padding: 20px;
    border-radius: 12px;
    border-left: 5px solid #16a34a;
    margin: 10px 0;
    box-shadow: 0 2px 8px rgba(22,163,74,0.1);
}
.valuation-box {
    background: linear-gradient(135deg, #fefce8 0%, #fef9c3 100%);
    padding: 20px;
    border-radius: 12px;
    border-left: 5px solid #f59e0b;
    margin: 10px 0;
    box-shadow: 0 2px 8px rgba(245,158,11,0.1);
}

/* 分割线 */
hr, .stDivider {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent 0%, #cbd5e1 50%, transparent 100%);
    margin: 30px 0;
}

/* Tab样式 */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #f1f5f9;
    padding: 8px;
    border-radius: 12px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: white;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

/* 隐藏默认元素 */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ===================== 策略配置 =====================
STRATEGY_DICT = {
    "价值投资策略（长线稳健）": {
        "desc": "精选低估值、高基本面沪深300个股，长期持有、分批止盈，适合稳健型投资者",
        "hold_days": (60, 120),
        "target_return": (0.15, 0.35),
        "batch_sell": True,
        "sell_ratio": [0.4, 0.3, 0.3],
        "capital_ratio": (15, 25),
        "risk_level": "低风险",
        "risk_color": "#16a34a",
        "select_rule": "优先选择DDM估值合理、PE/PB偏低、ROE稳定、60日走势稳定的个股"
    },
    "趋势追涨策略（中线波段）": {
        "desc": "筛选均线多头、成交量放大的趋势个股，波段操作，适合中等风险投资者",
        "hold_days": (20, 45),
        "target_return": (0.08, 0.25),
        "batch_sell": True,
        "sell_ratio": [0.5, 0.5],
        "capital_ratio": (18, 28),
        "risk_level": "中风险",
        "risk_color": "#f59e0b",
        "select_rule": "优先选择均线多头排列、MACD金叉、短期涨幅靠前的个股"
    },
    "反转抄底策略（短线博弈）": {
        "desc": "筛选短期超跌、缩量企稳个股，短线反弹博弈，适合激进型投资者",
        "hold_days": (7, 20),
        "target_return": (0.05, 0.18),
        "batch_sell": False,
        "sell_ratio": [1.0],
        "capital_ratio": (20, 30),
        "risk_level": "高风险",
        "risk_color": "#dc2626",
        "select_rule": "优先选择RSI超卖、短期超跌、波动率下降的反弹潜力个股"
    }
}

# ===================== 行业分类映射 =====================
INDUSTRY_CATEGORIES = {
    "金融": ["银行", "保险", "证券", "多元金融", "信托", "期货"],
    "消费": ["白酒", "食品", "饮料", "家电", "零售", "服装", "化妆品", "旅游", "酒店"],
    "医药": ["医药", "生物", "医疗器械", "医疗服务", "中药", "化学制药"],
    "科技": ["半导体", "电子", "软件", "计算机", "通信", "互联网", "人工智能"],
    "新能源": ["光伏", "风电", "锂电池", "新能源汽车", "储能", "氢能"],
    "制造": ["机械", "汽车", "军工", "航空航天", "船舶", "重工"],
    "材料": ["化工", "钢铁", "有色", "建材", "造纸", "塑料", "橡胶"],
    "基建": ["建筑", "房地产", "建材", "工程", "水泥"],
    "能源": ["煤炭", "石油", "天然气", "电力", "公用事业"],
    "物流": ["交通运输", "物流", "港口", "航运", "航空"]
}

# ===================== 数据获取函数 =====================
@st.cache_data(ttl=1800, show_spinner="正在获取沪深300成分股列表...")
def get_hs300_constituents():
    """获取沪深300成分股列表（使用akshare实时数据）"""
    try:
        df = ak.index_stock_cons(symbol="000300")
        code_col = "品种代码" if "品种代码" in df.columns else "code"
        name_col = "品种名称" if "品种名称" in df.columns else "code_name"
        df["纯代码"] = df[code_col].apply(lambda x: re.sub(r"^(sh\.|sz\.)", "", str(x)))
        df["纯代码"] = pd.to_numeric(df["纯代码"], errors="coerce")
        df = df.dropna(subset=["纯代码"])
        code2name = dict(zip(df["纯代码"].astype(int), df[name_col]))
        return code2name, f"成功获取沪深300成分股，共 {len(code2name)} 只"
    except Exception as e:
        st.error(f"获取沪深300成分股失败: {e}")
        return {}, f"获取失败: {e}"


@st.cache_data(ttl=1800, show_spinner=False)
def get_stock_individual_info(symbol):
    """获取个股基本信息：行业、主营业务等"""
    try:
        df = ak.stock_individual_info_em(symbol=symbol)
        info_dict = dict(zip(df["item"], df["value"]))
        return info_dict
    except Exception:
        return {}


@st.cache_data(ttl=1800, show_spinner=False)
def get_stock_financial_indicators(symbol):
    """获取个股财务分析指标"""
    try:
        df = ak.stock_financial_analysis_indicator_em(symbol=symbol)
        return df
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def get_stock_main_business(symbol):
    """获取个股主营业务信息"""
    try:
        df = ak.stock_main_business_em(symbol=symbol)
        return df
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def get_stock_dividend_history(symbol):
    """获取个股分红历史"""
    try:
        df = ak.stock_dividend_cninfo(symbol=symbol)
        return df
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def get_stock_history_data(_stock_code, _stock_name, start_date, end_date):
    """获取单只股票历史日线数据"""
    try:
        code_str = str(_stock_code).zfill(6)
        df = ak.stock_zh_a_hist(
            symbol=code_str,
            period="daily",
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            adjust="qfq"
        )
        if df is None or df.empty:
            return None

        df = df.rename(columns={
            "日期": "日期", "开盘": "开盘", "收盘": "收盘",
            "最高": "最高", "最低": "最低", "成交量": "成交量",
            "成交额": "成交额", "振幅": "振幅", "涨跌额": "涨跌额",
            "换手率": "换手率", "涨跌幅": "涨跌幅"
        })

        df["股票代码"] = _stock_code
        df["股票名称"] = _stock_name
        df["日期"] = pd.to_datetime(df["日期"])

        for col in ["开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌额", "换手率", "涨跌幅"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["日期", "开盘", "收盘", "最高", "最低"])
        df = df.sort_values("日期").reset_index(drop=True)
        return df
    except Exception:
        return None


def fetch_all_hs300_data(code2name, start_date, end_date, progress_bar, status_text):
    """批量获取沪深300成分股数据"""
    all_data = []
    total = len(code2name)
    success = 0
    failed = 0

    for i, (code, name) in enumerate(code2name.items()):
        progress_bar.progress(min(i / total, 0.99),
                              text=f"正在获取: {name}({code}) [{i+1}/{total}]")
        status_text.text(f"进度: {i+1}/{total} | 成功: {success} | 失败: {failed}")

        df = get_stock_history_data(code, name, start_date, end_date)
        if df is not None and len(df) >= 30:
            all_data.append(df)
            success += 1
        else:
            failed += 1

        if (i + 1) % 5 == 0:
            time.sleep(0.3)

    progress_bar.progress(1.0, text="数据获取完成！")
    status_text.text(f"获取完成: 成功 {success} 只, 失败 {failed} 只")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values(["股票代码", "日期"]).reset_index(drop=True)
        return combined_df, f"数据获取完成 | 共 {success} 只股票 | {len(combined_df)} 条记录"
    return None, "未能获取到有效数据"


# ===================== 基本面分析函数 =====================
def classify_industry(industry_name):
    """将行业归类到大类"""
    if not industry_name:
        return "其他"
    industry_name = str(industry_name)
    for category, keywords in INDUSTRY_CATEGORIES.items():
        for kw in keywords:
            if kw in industry_name:
                return category
    return "其他"


def get_stock_fundamentals_batch(stock_codes):
    """批量获取多只股票的基本面信息"""
    fundamentals = {}
    for code in stock_codes:
        code_str = str(code).zfill(6)
        try:
            info = get_stock_individual_info(code_str)
            fundamentals[code] = {
                "行业": info.get("行业", "未知"),
                "总股本": info.get("总股本", 0),
                "总市值": info.get("总市值", 0),
                "流通市值": info.get("流通市值", 0),
                "上市时间": info.get("上市时间", ""),
            }
        except Exception:
            fundamentals[code] = {"行业": "未知", "总股本": 0, "总市值": 0, "流通市值": 0, "上市时间": ""}
    return fundamentals


def calc_ddm_valuation(stock_code, stock_name, close_price, fin_df):
    """
    DDM股利贴现模型估值
    使用戈登增长模型: V = D1 / (r - g)
    其中: D1 = 下一年预期每股股利, r = 折现率, g = 永续增长率
    """
    try:
        # 从财务指标获取关键数据
        if fin_df is not None and not fin_df.empty:
            latest = fin_df.iloc[0] if isinstance(fin_df.iloc[0], pd.Series) else fin_df.iloc[-1]

            # 获取EPS和ROE
            eps = 0
            roe = 0
            pe = 0
            pb = 0
            dividend_yield = 0

            # 尝试从财务数据中提取
            for col in fin_df.columns:
                col_str = str(col).lower()
                if "每股收益" in str(col) or "eps" in col_str:
                    try:
                        eps = float(fin_df[col].iloc[0])
                    except:
                        pass
                if "净资产收益率" in str(col) or "roe" in col_str:
                    try:
                        roe = float(fin_df[col].iloc[0])
                    except:
                        pass
                if "市盈率" in str(col) or "pe" in col_str:
                    try:
                        pe = float(fin_df[col].iloc[0])
                    except:
                        pass
                if "市净率" in str(col) or "pb" in col_str:
                    try:
                        pb = float(fin_df[col].iloc[0])
                    except:
                        pass
                if "股息率" in str(col) or "dividend" in col_str:
                    try:
                        dividend_yield = float(fin_df[col].iloc[0])
                    except:
                        pass

            # 如果无法获取EPS，用股价/PE估算
            if eps <= 0 and pe > 0:
                eps = close_price / pe
            if eps <= 0:
                eps = close_price * 0.05  # 默认5%收益率

            # 分红率估算（沪深300平均约30%）
            payout_ratio = 0.30

            # 每股股利
            dps = eps * payout_ratio

            # 折现率 r = 无风险利率 + 风险溢价
            risk_free_rate = 0.025  # 10年期国债收益率约2.5%
            market_risk_premium = 0.06  # 市场风险溢价约6%
            beta = 1.0  # 沪深300成分股beta约1
            r = risk_free_rate + beta * market_risk_premium  # 约8.5%

            # 永续增长率 g = ROE * (1 - 分红率)
            if roe > 0:
                g = roe / 100 * (1 - payout_ratio)
            else:
                g = 0.03  # 默认3%增长率

            g = min(g, r * 0.8)  # 确保 g < r

            # 戈登增长模型
            if r > g and dps > 0:
                intrinsic_value = dps * (1 + g) / (r - g)
            else:
                intrinsic_value = close_price  # 无法计算时返回当前价

            # 估值判断
            premium = (close_price - intrinsic_value) / intrinsic_value * 100

            if premium < -20:
                valuation = "严重低估"
                valuation_color = "#16a34a"
            elif premium < -5:
                valuation = "低估"
                valuation_color = "#22c55e"
            elif premium < 5:
                valuation = "合理"
                valuation_color = "#f59e0b"
            elif premium < 20:
                valuation = "高估"
                valuation_color = "#f97316"
            else:
                valuation = "严重高估"
                valuation_color = "#dc2626"

            return {
                "内在价值": round(intrinsic_value, 2),
                "当前价格": round(close_price, 2),
                "估值溢价": round(premium, 2),
                "估值判断": valuation,
                "估值颜色": valuation_color,
                "每股股利": round(dps, 3),
                "折现率": round(r * 100, 2),
                "永续增长率": round(g * 100, 2),
                "ROE": round(roe, 2) if roe else "N/A",
                "EPS": round(eps, 2) if eps else "N/A",
                "PE": round(pe, 2) if pe else "N/A",
                "PB": round(pb, 2) if pb else "N/A",
                "股息率": round(dividend_yield, 2) if dividend_yield else "N/A"
            }
    except Exception:
        pass

    # 默认返回
    return {
        "内在价值": round(close_price, 2),
        "当前价格": round(close_price, 2),
        "估值溢价": 0,
        "估值判断": "数据不足",
        "估值颜色": "#64748b",
        "每股股利": "N/A",
        "折现率": "N/A",
        "永续增长率": "N/A",
        "ROE": "N/A",
        "EPS": "N/A",
        "PE": "N/A",
        "PB": "N/A",
        "股息率": "N/A"
    }


# ===================== 技术分析函数 =====================
def calc_technical_indicators(df):
    """计算技术指标"""
    df = df.copy()
    df["MA5"] = df["收盘"].rolling(5).mean()
    df["MA10"] = df["收盘"].rolling(10).mean()
    df["MA20"] = df["收盘"].rolling(20).mean()
    df["MA60"] = df["收盘"].rolling(60).mean()

    ema12 = df["收盘"].ewm(span=12, adjust=False).mean()
    ema26 = df["收盘"].ewm(span=26, adjust=False).mean()
    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["MACD"] = (df["DIF"] - df["DEA"]) * 2

    delta = df["收盘"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["BOLL_MID"] = df["收盘"].rolling(20).mean()
    boll_std = df["收盘"].rolling(20).std()
    df["BOLL_UPPER"] = df["BOLL_MID"] + 2 * boll_std
    df["BOLL_LOWER"] = df["BOLL_MID"] - 2 * boll_std

    df["VOL_MA5"] = df["成交量"].rolling(5).mean()
    df["VOL_MA10"] = df["成交量"].rolling(10).mean()
    return df


def calc_monte_carlo_prediction(df, predict_days=30):
    """蒙特卡洛模拟价格预测"""
    close_prices = df["收盘"].values
    returns = np.diff(np.log(close_prices))
    last_price = close_prices[-1]

    x = np.arange(len(returns))
    slope, intercept, _, _, _ = stats.linregress(x, returns)
    volatility = np.std(returns[-60:]) if len(returns) >= 60 else np.std(returns)

    n_simulations = 1000
    prediction_paths = np.zeros((n_simulations, predict_days))
    prediction_paths[:, 0] = last_price

    for t in range(1, predict_days):
        shock = np.random.normal(slope, volatility, n_simulations)
        prediction_paths[:, t] = prediction_paths[:, t-1] * np.exp(shock)

    median_prediction = np.median(prediction_paths, axis=0)
    lower_bound = np.percentile(prediction_paths, 10, axis=0)
    upper_bound = np.percentile(prediction_paths, 90, axis=0)

    last_date = df["日期"].iloc[-1]
    pred_dates = []
    d = last_date
    while len(pred_dates) < predict_days:
        d = d + timedelta(days=1)
        if d.weekday() < 5:
            pred_dates.append(d)

    pred_df = pd.DataFrame({
        "日期": pred_dates[:predict_days],
        "预测价格": median_prediction[:len(pred_dates)],
        "上界": upper_bound[:len(pred_dates)],
        "下界": lower_bound[:len(pred_dates)]
    })
    return pred_df


# ===================== 选股策略函数 =====================
def stock_filter_and_pick(df, strategy, code2name, fundamentals=None):
    """根据策略筛选5只个股 + 生成预测和估值数据"""
    stock_list = df["股票代码"].unique()
    metrics = []

    # 批量获取基本面信息
    if fundamentals is None:
        fundamentals = get_stock_fundamentals_batch(stock_list[:50])

    for code in stock_list:
        s_df = df[df["股票代码"] == code].sort_values("日期").reset_index(drop=True)
        if len(s_df) < 60:
            continue

        s_df = calc_technical_indicators(s_df)
        close = s_df["收盘"].iloc[-1]

        ret20 = (close - s_df["收盘"].iloc[-21]) / s_df["收盘"].iloc[-21] if len(s_df) >= 21 else 0
        ret60 = (close - s_df["收盘"].iloc[-61]) / s_df["收盘"].iloc[-61] if len(s_df) >= 61 else 0
        vol = s_df["换手率"].iloc[-20:].mean()
        std = s_df["涨跌幅"].iloc[-20:].std()

        ma5 = s_df["MA5"].iloc[-1]
        ma10 = s_df["MA10"].iloc[-1]
        ma20 = s_df["MA20"].iloc[-1]

        if ma5 > ma10 > ma20:
            trend = 1
        elif ma5 < ma10 < ma20:
            trend = -1
        else:
            trend = 0

        macd_signal = 1 if s_df["DIF"].iloc[-1] > s_df["DEA"].iloc[-1] else -1
        rsi = s_df["RSI"].iloc[-1] if not np.isnan(s_df["RSI"].iloc[-1]) else 50

        # 获取基本面
        fund = fundamentals.get(code, {})
        industry = fund.get("行业", "未知")
        category = classify_industry(industry)

        # 获取财务指标和DDM估值
        code_str = str(code).zfill(6)
        fin_df = get_stock_financial_indicators(code_str)
        ddm = calc_ddm_valuation(code, code2name.get(code, ""), close, fin_df)

        # 综合评分
        score = 0
        score += trend * 20
        score += macd_signal * 15
        if 30 <= rsi <= 70:
            score += 10
        elif rsi < 30:
            score += 15
        score -= std * 100

        # DDM估值加分
        if ddm["估值判断"] == "严重低估":
            score += 30
        elif ddm["估值判断"] == "低估":
            score += 20
        elif ddm["估值判断"] == "合理":
            score += 10

        metrics.append({
            "股票代码": code,
            "股票名称": code2name.get(code, "未知"),
            "最新价": close,
            "20日涨幅": ret20 * 100,
            "60日涨幅": ret60 * 100,
            "波动率": std,
            "均线趋势": trend,
            "MACD信号": macd_signal,
            "RSI": rsi,
            "综合评分": score,
            "行业": industry,
            "行业大类": category,
            "技术数据": s_df,
            "DDM估值": ddm,
            "财务数据": fin_df
        })

    if len(metrics) < 5:
        return None, "有效个股不足5只，无法完成选股"

    m_df = pd.DataFrame(metrics)

    if "价值投资" in strategy:
        # 价值投资：优先DDM低估+低波动+稳定
        m_df = m_df.sort_values(["综合评分", "波动率"], ascending=[False, True])
    elif "趋势追涨" in strategy:
        trend_df = m_df[m_df["均线趋势"] == 1]
        if len(trend_df) >= 5:
            m_df = trend_df.sort_values("综合评分", ascending=False)
        else:
            m_df = m_df.sort_values("综合评分", ascending=False)
    else:
        oversold = m_df[m_df["RSI"] < 40]
        if len(oversold) >= 5:
            m_df = oversold.sort_values("综合评分", ascending=False)
        else:
            m_df = m_df.sort_values("20日涨幅", ascending=True)

    selected = m_df.head(5).reset_index(drop=True)
    res = []
    today = datetime.now()
    cfg = STRATEGY_DICT[strategy]

    for i, row in selected.iterrows():
        s_df = row["技术数据"]
        cur_price = row["最新价"]
        ddm = row["DDM估值"]

        hist_vol = s_df["涨跌幅"].iloc[-60:].std() if len(s_df) >= 60 else s_df["涨跌幅"].std()
        avg_daily_ret = s_df["涨跌幅"].iloc[-20:].mean()

        hold = np.random.randint(*cfg["hold_days"])
        ret = min(max(np.random.normal(avg_daily_ret * hold, hist_vol * np.sqrt(hold)),
                       cfg["target_return"][0] * 0.5),
                   cfg["target_return"][1] * 1.2)
        ret = max(ret, 0.01)

        cap = round(np.random.uniform(*cfg["capital_ratio"]), 1)
        buy = round(cur_price * np.random.uniform(0.97, 1.00), 2)
        sell = round(buy * (1 + ret), 2)
        sell_dt = (today + timedelta(days=hold)).strftime("%Y-%m-%d")

        pred_df = calc_monte_carlo_prediction(s_df, predict_days=min(hold + 10, 60))

        # 生成投资建议
        advice = generate_investment_advice(row, ddm, strategy)

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
            "RSI": round(row["RSI"], 2),
            "均线趋势": "多头排列" if row["均线趋势"] == 1 else ("空头排列" if row["均线趋势"] == -1 else "震荡"),
            "MACD信号": "金叉" if row["MACD信号"] == 1 else "死叉",
            "综合评分": round(row["综合评分"], 2),
            "行业": row["行业"],
            "行业大类": row["行业大类"],
            "DDM估值": ddm,
            "投资建议": advice,
            "K线数据": s_df.tail(120),
            "预测数据": pred_df
        })

    total = sum(x["资金占比"] for x in res)
    for x in res:
        x["资金占比"] = round(x["资金占比"] / total * 100, 1)

    return res, f"选股完成，共筛选出5只优质个股（策略: {strategy}）"


def generate_investment_advice(row, ddm, strategy):
    """根据股票特征生成个性化投资建议"""
    advice_list = []
    risk_notes = []

    # 基于DDM估值的建议
    if ddm["估值判断"] in ["严重低估", "低估"]:
        advice_list.append(f"DDM估值显示该股票{ddm['估值判断']}，内在价值约{ddm['内在价值']}元，当前价格具备安全边际")
    elif ddm["估值判断"] in ["高估", "严重高估"]:
        risk_notes.append(f"DDM估值显示该股票{ddm['估值判断']}，内在价值约{ddm['内在价值']}元，注意估值回调风险")
    else:
        advice_list.append("DDM估值显示当前价格合理，建议关注基本面变化")

    # 基于技术面的建议
    rsi = row["RSI"]
    if rsi < 30:
        advice_list.append(f"RSI={rsi:.1f}处于超卖区域，短期可能存在反弹机会")
    elif rsi > 70:
        risk_notes.append(f"RSI={rsi:.1f}处于超买区域，注意短期回调风险")

    trend = row["均线趋势"]
    if trend == 1:
        advice_list.append("均线多头排列，中期趋势向好")
    elif trend == -1:
        risk_notes.append("均线空头排列，中期趋势偏弱")

    macd = row["MACD信号"]
    if macd == 1:
        advice_list.append("MACD金叉信号，动能转强")
    else:
        risk_notes.append("MACD死叉信号，动能转弱")

    # 基于策略的建议
    if "价值投资" in strategy:
        advice_list.append("建议分批建仓，长期持有，关注季度财报和分红政策变化")
        advice_list.append("设置10%-15%止损线，若基本面恶化及时止损")
    elif "趋势追涨" in strategy:
        advice_list.append("建议在回调至支撑位时介入，避免追高")
        advice_list.append("设置8%-10%止损线，趋势反转时果断离场")
    else:
        advice_list.append("快进快出，严格按目标价止盈")
        advice_list.append("设置5%-8%止损线，控制单笔亏损")

    # 行业建议
    category = row["行业大类"]
    industry = row["行业"]
    if category in ["金融", "能源"]:
        advice_list.append(f"{industry}板块受宏观经济政策和利率影响较大，建议关注政策面变化")
    elif category in ["科技", "新能源"]:
        advice_list.append(f"{industry}板块成长性高但波动大，建议控制仓位")
    elif category in ["消费", "医药"]:
        advice_list.append(f"{industry}板块防御性强，适合作为组合底仓")

    return {
        "买入建议": advice_list,
        "风险提示": risk_notes,
        "行业": row["行业"],
        "行业大类": category
    }


# ===================== 可视化函数 =====================
def draw_index_overview(df):
    """绘制沪深300整体市场概览"""
    daily_stats = df.groupby("日期").agg({
        "收盘": "mean", "成交量": "sum", "涨跌幅": "mean",
        "振幅": "mean", "换手率": "mean"
    }).reset_index()
    daily_stats["累计收益"] = (daily_stats["收盘"] / daily_stats["收盘"].iloc[0] - 1) * 100

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )

    fig.add_trace(go.Scatter(
        x=daily_stats["日期"], y=daily_stats["收盘"],
        mode="lines", line=dict(color="#1e40af", width=2), name="平均收盘价",
        fill="tozeroy", fillcolor="rgba(30,64,175,0.05)"
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=daily_stats["日期"], y=daily_stats["累计收益"],
        mode="lines", line=dict(color="#06b6d4", width=1.5, dash="dot"), name="累计收益(%)"
    ), row=1, col=1, secondary_y=True)

    vol_colors = ["#16a34a" if r >= 0 else "#dc2626" for r in daily_stats["涨跌幅"]]
    fig.add_trace(go.Bar(x=daily_stats["日期"], y=daily_stats["成交量"],
                         marker_color=vol_colors, name="成交量", opacity=0.7), row=2, col=1)

    fig.add_trace(go.Bar(x=daily_stats["日期"], y=daily_stats["涨跌幅"],
                         marker_color=vol_colors, name="平均涨跌幅(%)", opacity=0.8), row=3, col=1)

    fig.update_layout(
        title="沪深300成分股整体走势概览",
        height=700, template="plotly_white",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_rangeslider_visible=False
    )
    fig.update_yaxes(title_text="平均价格", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="累计收益(%)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    fig.update_yaxes(title_text="涨跌幅(%)", row=3, col=1)
    return fig


def draw_stock_kline(s_df, name, code):
    """绘制个股K线+技术指标"""
    s_df = calc_technical_indicators(s_df)
    up_color = "#dc2626"
    down_color = "#16a34a"

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.45, 0.15, 0.2, 0.2]
    )

    fig.add_trace(go.Candlestick(
        x=s_df["日期"], open=s_df["开盘"], high=s_df["最高"],
        low=s_df["最低"], close=s_df["收盘"],
        increasing_line_color=up_color, decreasing_line_color=down_color, name="日K线"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=s_df["日期"], y=s_df["MA5"], line=dict(color="#1e40af", width=1.2), name="MA5"), row=1, col=1)
    fig.add_trace(go.Scatter(x=s_df["日期"], y=s_df["MA10"], line=dict(color="#f59e0b", width=1.2), name="MA10"), row=1, col=1)
    fig.add_trace(go.Scatter(x=s_df["日期"], y=s_df["MA20"], line=dict(color="#16a34a", width=1.2), name="MA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=s_df["日期"], y=s_df["BOLL_UPPER"], line=dict(color="#8b5cf6", width=1, dash="dot"), name="布林上轨", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=s_df["日期"], y=s_df["BOLL_LOWER"], line=dict(color="#8b5cf6", width=1, dash="dot"), name="布林下轨", showlegend=False), row=1, col=1)

    vol_colors = [up_color if o <= c else down_color for o, c in zip(s_df["开盘"], s_df["收盘"])]
    fig.add_trace(go.Bar(x=s_df["日期"], y=s_df["成交量"], marker_color=vol_colors, name="成交量", opacity=0.7), row=2, col=1)

    macd_colors = ["#16a34a" if v >= 0 else "#dc2626" for v in s_df["MACD"]]
    fig.add_trace(go.Bar(x=s_df["日期"], y=s_df["MACD"], marker_color=macd_colors, name="MACD柱", opacity=0.7), row=3, col=1)
    fig.add_trace(go.Scatter(x=s_df["日期"], y=s_df["DIF"], line=dict(color="#1e40af", width=1.5), name="DIF"), row=3, col=1)
    fig.add_trace(go.Scatter(x=s_df["日期"], y=s_df["DEA"], line=dict(color="#f59e0b", width=1.5), name="DEA"), row=3, col=1)

    fig.add_trace(go.Scatter(x=s_df["日期"], y=s_df["RSI"], line=dict(color="#8b5cf6", width=1.5), name="RSI(14)"), row=4, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#dc2626", line_width=1, row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#16a34a", line_width=1, row=4, col=1)

    fig.update_layout(
        title=f"{name}({code}) 技术分析图",
        height=900, template="plotly_white",
        legend=dict(orientation="h", y=1.02, font=dict(size=10)),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_rangeslider_visible=False
    )
    fig.update_yaxes(title_text="价格(元)", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="RSI(14)", row=4, col=1, range=[0, 100])
    return fig


def draw_prediction_chart(s_df, pred_df, name, code):
    """绘制价格预测图"""
    fig = go.Figure()
    hist_data = s_df.tail(60)

    fig.add_trace(go.Scatter(
        x=hist_data["日期"], y=hist_data["收盘"],
        mode="lines", line=dict(color="#1e40af", width=2), name="历史价格"
    ))
    fig.add_trace(go.Scatter(
        x=pred_df["日期"], y=pred_df["预测价格"],
        mode="lines", line=dict(color="#f59e0b", width=2, dash="dash"), name="预测价格"
    ))
    fig.add_trace(go.Scatter(
        x=pred_df["日期"].tolist() + pred_df["日期"].tolist()[::-1],
        y=pred_df["上界"].tolist() + pred_df["下界"].tolist()[::-1],
        fill="toself", fillcolor="rgba(245,158,11,0.15)",
        line=dict(color="rgba(245,158,11,0.3)", width=1),
        name="90%置信区间"
    ))

    last_price = hist_data["收盘"].iloc[-1]
    fig.add_hline(y=last_price, line_dash="dot", line_color="#1e40af",
                  annotation_text=f"当前价: {last_price:.2f}")

    fig.update_layout(
        title=f"{name}({code}) 价格预测走势",
        height=450, template="plotly_white",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_rangeslider_visible=False,
        xaxis_title="日期", yaxis_title="价格(元)"
    )
    return fig


def draw_correlation_heatmap(df, code2name, top_n=20):
    """绘制个股涨跌幅相关性热力图"""
    stock_counts = df.groupby("股票代码").size().sort_values(ascending=False).head(top_n)
    top_codes = stock_counts.index.tolist()

    pivot_df = df[df["股票代码"].isin(top_codes)].pivot_table(
        index="日期", columns="股票代码", values="涨跌幅"
    )
    corr_matrix = pivot_df.corr()
    labels = [f"{code2name.get(c, str(c))}" for c in corr_matrix.columns]

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values, x=labels, y=labels,
        colorscale="RdBu_r", zmin=-1, zmax=1,
        colorbar=dict(title="相关系数")
    ))

    fig.update_layout(
        title=f"沪深300成分股涨跌幅相关性热力图（Top {top_n}）",
        height=700, template="plotly_white",
        margin=dict(l=100, r=20, t=50, b=100),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9))
    )
    return fig


def draw_sector_analysis(df, code2name):
    """绘制板块/行业分布分析"""
    def classify_stock(code):
        code_str = str(code).zfill(6)
        prefix = code_str[0]
        if prefix == "6":
            return "上海主板(60xxxx)"
        elif prefix == "0":
            return "深圳主板(00xxxx)"
        elif prefix == "3":
            return "创业板(30xxxx)"
        return "其他"

    sector_data = df.copy()
    sector_data["板块"] = sector_data["股票代码"].apply(classify_stock)

    sector_stats = sector_data.groupby("板块").agg({
        "股票代码": "nunique", "收盘": "mean", "涨跌幅": "mean",
        "成交量": "sum", "换手率": "mean"
    }).reset_index()
    sector_stats.columns = ["板块", "股票数量", "平均价格", "平均涨跌幅(%)", "总成交量", "平均换手率(%)"]

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "pie"}, {"type": "bar"}, {"type": "bar"}]],
        subplot_titles=("股票数量分布", "平均涨跌幅对比", "平均换手率对比")
    )

    fig.add_trace(go.Pie(
        labels=sector_stats["板块"], values=sector_stats["股票数量"],
        hole=0.4, marker_colors=["#1e40af", "#16a34a", "#f59e0b", "#64748b"],
        textinfo="label+percent"
    ), row=1, col=1)

    colors = ["#16a34a" if v >= 0 else "#dc2626" for v in sector_stats["平均涨跌幅(%)"]]
    fig.add_trace(go.Bar(
        x=sector_stats["板块"], y=sector_stats["平均涨跌幅(%)"],
        marker_color=colors, name="平均涨跌幅(%)"
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        x=sector_stats["板块"], y=sector_stats["平均换手率(%)"],
        marker_color=["#8b5cf6"] * len(sector_stats), name="平均换手率(%)"
    ), row=1, col=3)

    fig.update_layout(
        title="沪深300成分股板块分析",
        height=400, template="plotly_white",
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig, sector_stats


def draw_risk_analysis(df, code2name):
    """绘制风险分析图"""
    stock_metrics = []
    for code in df["股票代码"].unique():
        s_df = df[df["股票代码"] == code].sort_values("日期").tail(60)
        if len(s_df) < 30:
            continue
        returns = s_df["涨跌幅"].values
        stock_metrics.append({
            "股票代码": code,
            "股票名称": code2name.get(code, str(code)),
            "年化波动率": np.std(returns) * np.sqrt(252) * 100,
            "最大回撤": ((s_df["收盘"].cummax() - s_df["收盘"]) / s_df["收盘"].cummax()).max() * 100,
            "夏普比率": (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0,
            "平均日收益(%)": np.mean(returns)
        })

    metrics_df = pd.DataFrame(stock_metrics)
    metrics_df = metrics_df.sort_values("夏普比率", ascending=False).head(20)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scatter"}, {"type": "bar"}]],
        subplot_titles=("风险-收益散点图", "Top20夏普比率排名")
    )

    fig.add_trace(go.Scatter(
        x=metrics_df["年化波动率"],
        y=metrics_df["平均日收益(%)"] * 252,
        mode="markers+text", text=metrics_df["股票名称"],
        textposition="top center", textfont=dict(size=8),
        marker=dict(size=12, color=metrics_df["夏普比率"],
                    colorscale="RdYlGn", showscale=True,
                    colorbar=dict(title="夏普比率")),
        name="个股"
    ), row=1, col=1)

    colors = ["#16a34a" if v > 0 else "#dc2626" for v in metrics_df["夏普比率"]]
    fig.add_trace(go.Bar(
        y=metrics_df["股票名称"], x=metrics_df["夏普比率"],
        orientation="h", marker_color=colors, name="夏普比率"
    ), row=1, col=2)

    fig.update_layout(
        title="沪深300成分股风险分析",
        height=500, template="plotly_white",
        margin=dict(l=100, r=20, t=60, b=20),
        showlegend=False
    )
    return fig, metrics_df


# ===================== 主程序 =====================
def main():
    st.markdown("""
    <div class="hero-title">
        <h1>沪深300股票智能预测分析平台</h1>
        <p>实时数据 | DDM估值 | 基本面分析 | 智能选股 | 价格预测 | 风险评估</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    with st.sidebar:
        st.markdown("<div class='sidebar-header'><h3>控制面板</h3></div>", unsafe_allow_html=True)

        st.subheader("1. 数据获取方式")
        data_source = st.radio(
            "选择数据源",
            ["在线获取实时数据（推荐）", "上传本地CSV文件"],
            index=0,
            help="在线获取使用akshare实时数据，上传文件需自行准备数据"
        )

        st.subheader("2. 数据时间范围")
        col_a, col_b = st.columns(2)
        with col_a:
            start_date = st.date_input("开始日期", datetime(2025, 1, 1))
        with col_b:
            end_date = st.date_input("结束日期", datetime.now())
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        st.subheader("3. 投资策略选择")
        select_strategy = st.selectbox("选择策略", list(STRATEGY_DICT.keys()))
        strat_cfg = STRATEGY_DICT[select_strategy]

        st.markdown(f"""
        <div style="background: {strat_cfg['risk_color']}15; padding: 12px; border-radius: 8px;
                    border-left: 3px solid {strat_cfg['risk_color']}; margin-top: 8px;">
            <strong>风险等级: {strat_cfg['risk_level']}</strong><br>
            <small>{strat_cfg['desc']}</small>
        </div>
        """, unsafe_allow_html=True)

        run_btn = st.button("开始智能分析", type="primary", use_container_width=True)

        st.divider()
        st.subheader("文件上传（备选）")
        file_name_map = st.file_uploader("沪深300名单CSV", type=["csv"], key="name_file")
        file_data = st.file_uploader("股票行情CSV", type=["csv"], key="data_file")

    # ==================== 数据加载 ====================
    df_stock = None
    code2name = {}
    fundamentals = {}

    if data_source == "在线获取实时数据（推荐）":
        code2name, name_msg = get_hs300_constituents()
        if code2name:
            st.sidebar.success(name_msg)
            if run_btn:
                progress_bar = st.progress(0, text="准备获取数据...")
                status_text = st.empty()
                df_stock, data_msg = fetch_all_hs300_data(
                    code2name, start_date_str, end_date_str, progress_bar, status_text
                )
                if df_stock is not None:
                    st.success(data_msg)
                    with st.spinner("正在获取基本面信息..."):
                        fundamentals = get_stock_fundamentals_batch(df_stock["股票代码"].unique()[:50])
                else:
                    st.error(data_msg)
        else:
            st.error(name_msg)
    else:
        if file_name_map is not None:
            try:
                df_map = pd.read_csv(file_name_map)
                df_map["纯代码"] = df_map["code"].apply(lambda x: re.sub(r"^(sh\.|sz\.)", "", str(x)))
                df_map["纯代码"] = pd.to_numeric(df_map["纯代码"], errors="coerce")
                df_map = df_map.dropna(subset=["纯代码"])
                code2name = dict(zip(df_map["纯代码"].astype(int), df_map["code_name"]))
                st.sidebar.success(f"沪深300名单加载成功，共 {len(code2name)} 只")
            except Exception as e:
                st.sidebar.error(f"名单加载失败: {e}")

        if file_data is not None and code2name:
            try:
                df_stock = pd.read_csv(file_data)
                required_cols = ["股票代码", "日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌额", "换手率", "涨跌幅"]
                miss = [c for c in required_cols if c not in df_stock.columns]
                if miss:
                    st.error(f"缺少必填字段: {','.join(miss)}")
                    df_stock = None
                else:
                    df_stock["股票代码"] = pd.to_numeric(df_stock["股票代码"], errors="coerce").fillna(0).astype(int)
                    df_stock["日期"] = pd.to_datetime(df_stock["日期"], errors="coerce")
                    df_stock = df_stock.dropna(subset=["股票代码", "日期", "开盘", "收盘"])
                    df_stock = df_stock[df_stock["股票代码"] > 0]
                    df_stock = df_stock.sort_values(["股票代码", "日期"]).reset_index(drop=True)
                    df_stock["股票名称"] = df_stock["股票代码"].map(code2name)
                    st.success(f"行情数据加载完成 | {len(df_stock)} 条记录 | {df_stock['股票代码'].nunique()} 只股票")
            except Exception as e:
                st.error(f"行情数据加载失败: {e}")
                df_stock = None

    # ==================== 主页面内容 ====================
    if df_stock is not None and len(df_stock) > 0:
        st.subheader("数据概览")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("总数据行数", f"{len(df_stock):,}")
        c2.metric("覆盖个股数", df_stock["股票代码"].nunique())
        c3.metric("数据起始日", df_stock["日期"].min().strftime("%Y-%m-%d"))
        c4.metric("数据截止日", df_stock["日期"].max().strftime("%Y-%m-%d"))
        c5.metric("平均日涨跌幅", f"{df_stock.groupby('日期')['涨跌幅'].mean().iloc[-1]:.2f}%")

        with st.expander("查看数据样例（前20行）"):
            st.dataframe(df_stock.head(20), use_container_width=True, hide_index=True)
        st.divider()

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "市场概览", "技术分析", "智能选股", "价格预测", "风险分析"
        ])

        # ---- Tab1: 市场概览 ----
        with tab1:
            st.markdown("### 沪深300成分股整体走势")
            fig_overview = draw_index_overview(df_stock)
            st.plotly_chart(fig_overview, use_container_width=True)

            st.markdown("### 板块分布分析")
            fig_sector, sector_stats = draw_sector_analysis(df_stock, code2name)
            st.plotly_chart(fig_sector, use_container_width=True)
            st.dataframe(sector_stats, use_container_width=True, hide_index=True)

            st.markdown("### 个股涨跌幅相关性")
            fig_corr = draw_correlation_heatmap(df_stock, code2name, top_n=15)
            st.plotly_chart(fig_corr, use_container_width=True)

        # ---- Tab2: 技术分析 ----
        with tab2:
            st.markdown("### 选择个股进行技术分析")
            available_stocks = sorted(df_stock["股票代码"].unique())
            stock_options = {str(c): f"{code2name.get(c, '未知')}({c})" for c in available_stocks}
            selected_stock = st.selectbox(
                "选择股票", options=list(stock_options.keys()),
                format_func=lambda x: stock_options[x], key="tech_stock_select"
            )

            if selected_stock:
                code_int = int(selected_stock)
                s_df = df_stock[df_stock["股票代码"] == code_int].sort_values("日期").reset_index(drop=True)
                stock_name = code2name.get(code_int, "未知")

                if len(s_df) >= 30:
                    s_df_calc = calc_technical_indicators(s_df)
                    latest = s_df_calc.iloc[-1]

                    # 获取基本面信息
                    code_str = str(code_int).zfill(6)
                    info = get_stock_individual_info(code_str)
                    fin_df = get_stock_financial_indicators(code_str)
                    ddm = calc_ddm_valuation(code_int, stock_name, latest["收盘"], fin_df)

                    # 基本信息展示
                    st.markdown(f"""
                    <div class="info-box">
                        <strong>{stock_name}({code_int})</strong> |
                        <strong>行业:</strong> {info.get('行业', '未知')} |
                        <strong>总股本:</strong> {info.get('总股本', 'N/A')} |
                        <strong>总市值:</strong> {info.get('总市值', 'N/A')} |
                        <strong>上市时间:</strong> {info.get('上市时间', 'N/A')}
                    </div>
                    """, unsafe_allow_html=True)

                    # DDM估值卡片
                    st.markdown(f"""
                    <div class="valuation-box">
                        <h4>DDM股利贴现模型估值</h4>
                        <p><strong>内在价值:</strong> {ddm['内在价值']}元 |
                        <strong>当前价格:</strong> {ddm['当前价格']}元 |
                        <strong>估值溢价:</strong> <span style="color:{ddm['估值颜色']}">{ddm['估值溢价']}% ({ddm['估值判断']})</span></p>
                        <p><strong>每股股利:</strong> {ddm['每股股利']} |
                        <strong>折现率:</strong> {ddm['折现率']}% |
                        <strong>永续增长率:</strong> {ddm['永续增长率']}% |
                        <strong>ROE:</strong> {ddm['ROE']} |
                        <strong>PE:</strong> {ddm['PE']} |
                        <strong>PB:</strong> {ddm['PB']}</p>
                        <p><small>DDM模型公式: V = D1 / (r - g)，其中D1为预期每股股利，r为折现率，g为永续增长率</small></p>
                    </div>
                    """, unsafe_allow_html=True)

                    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
                    mc1.metric("最新收盘价", f"{latest['收盘']:.2f}元")
                    mc2.metric("RSI(14)", f"{latest['RSI']:.1f}",
                              "超买" if latest['RSI'] > 70 else ("超卖" if latest['RSI'] < 30 else "中性"))
                    mc3.metric("MACD", f"{latest['MACD']:.4f}",
                              "金叉" if latest['DIF'] > latest['DEA'] else "死叉")
                    mc4.metric("20日涨幅", f"{(latest['收盘']/s_df_calc['收盘'].iloc[-21]-1)*100:.2f}%")
                    mc5.metric("60日涨幅", f"{(latest['收盘']/s_df_calc['收盘'].iloc[-61]-1)*100:.2f}%"
                              if len(s_df_calc) >= 61 else "N/A")
                    mc6.metric("平均换手率", f"{s_df_calc['换手率'].iloc[-20:].mean():.2f}%")

                    fig_kline = draw_stock_kline(s_df_calc, stock_name, code_int)
                    st.plotly_chart(fig_kline, use_container_width=True)

                    st.markdown("#### 技术指标解读")
                    rsi_val = latest['RSI']
                    if rsi_val > 70:
                        rsi_msg = f"RSI={rsi_val:.1f}，处于**超买区域**，注意回调风险"
                    elif rsi_val < 30:
                        rsi_msg = f"RSI={rsi_val:.1f}，处于**超卖区域**，可能存在反弹机会"
                    else:
                        rsi_msg = f"RSI={rsi_val:.1f}，处于**中性区域**"

                    trend_msg = "多头排列（看涨）" if latest['MA5'] > latest['MA10'] > latest['MA20'] else \
                               "空头排列（看跌）" if latest['MA5'] < latest['MA10'] < latest['MA20'] else "震荡整理"

                    st.markdown(f"""
                    <div class="info-box">
                        <strong>RSI指标：</strong>{rsi_msg}<br>
                        <strong>均线排列：</strong>{trend_msg}<br>
                        <strong>MACD信号：</strong>{"DIF > DEA，金叉看涨" if latest['DIF'] > latest['DEA'] else "DIF < DEA，死叉看跌"}<br>
                        <strong>布林带：</strong>价格位于{"上轨附近，注意压力" if latest['收盘'] > latest['BOLL_MID'] else "下轨附近，关注支撑"}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning(f"{stock_name} 数据不足30条，无法进行技术分析")

        # ---- Tab3: 智能选股 ----
        with tab3:
            st.markdown(f"### 策略选股结果 — {select_strategy}")
            st.markdown(f"""
            <div class="info-box">
                <strong>策略说明：</strong>{strat_cfg['desc']}<br>
                <strong>选股规则：</strong>{strat_cfg['select_rule']}<br>
                <strong>风险等级：</strong>{strat_cfg['risk_level']} |
                <strong>建议持有：</strong>{strat_cfg['hold_days'][0]}-{strat_cfg['hold_days'][1]}天 |
                <strong>目标收益：</strong>{strat_cfg['target_return'][0]*100:.0f}%-{strat_cfg['target_return'][1]*100:.0f}%
            </div>
            """, unsafe_allow_html=True)

            stock_res, pick_msg = stock_filter_and_pick(df_stock, select_strategy, code2name, fundamentals)
            if stock_res is None:
                st.error(pick_msg)
            else:
                st.success(pick_msg)

                df_res = pd.DataFrame([{k: v for k, v in item.items()
                                        if k not in ["K线数据", "预测数据", "投资建议", "DDM估值"]}
                                       for item in stock_res])
                st.dataframe(df_res, use_container_width=True, hide_index=True)
                st.divider()

                st.markdown("### 个股详细分析")
                for item in stock_res:
                    ddm = item["DDM估值"]
                    advice = item["投资建议"]

                    with st.expander(
                        f"第{item['序号']}只: {item['股票名称']}({item['股票代码']}) | "
                        f"行业: {item['行业']} | 评分: {item['综合评分']:.1f} | "
                        f"DDM: {ddm['估值判断']}",
                        expanded=True
                    ):
                        # 基本信息
                        st.markdown(f"""
                        <div class="info-box">
                            <strong>行业分类：</strong>{item['行业']}（{item['行业大类']}）<br>
                            <strong>均线趋势：</strong>{item['均线趋势']} |
                            <strong>MACD信号：</strong>{item['MACD信号']} |
                            <strong>RSI：</strong>{item['RSI']}
                        </div>
                        """, unsafe_allow_html=True)

                        # DDM估值
                        st.markdown(f"""
                        <div class="valuation-box">
                            <h4>DDM股利贴现模型估值分析</h4>
                            <p><strong>内在价值:</strong> {ddm['内在价值']}元 |
                            <strong>当前价格:</strong> {ddm['当前价格']}元 |
                            <strong>估值溢价:</strong> <span style="color:{ddm['估值颜色']};font-weight:bold">{ddm['估值溢价']}% ({ddm['估值判断']})</span></p>
                            <p><strong>每股股利(DPS):</strong> {ddm['每股股利']} |
                            <strong>折现率(r):</strong> {ddm['折现率']}% |
                            <strong>永续增长率(g):</strong> {ddm['永续增长率']}%</p>
                            <p><strong>ROE:</strong> {ddm['ROE']}% |
                            <strong>EPS:</strong> {ddm['EPS']} |
                            <strong>PE:</strong> {ddm['PE']} |
                            <strong>PB:</strong> {ddm['PB']} |
                            <strong>股息率:</strong> {ddm['股息率']}%</p>
                            <p><small>戈登增长模型: V = D1 / (r - g)。折现率r = 无风险利率(2.5%) + Beta * 市场风险溢价(6%)</small></p>
                        </div>
                        """, unsafe_allow_html=True)

                        # 交易指标
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("建议买入价", f"{item['建议买入价']}元", f"现价 {item['最新收盘价']}元")
                        mc2.metric("资金占比", f"{item['资金占比']}%")
                        mc3.metric("预期卖出价", f"{item['预期卖出价']}元", f"收益 {item['预期收益率']}%")
                        mc4.metric("预期卖出日", item["预期卖出日"], f"持有 {item['持有天数']}天")

                        c_info1, c_info2 = st.columns(2)
                        c_info1.info(f"分批卖出: {item['分批卖出']}")
                        c_info2.info(f"卖出比例: {item['卖出比例']}")

                        # K线图
                        k_fig = draw_stock_kline(item["K线数据"], item["股票名称"], item["股票代码"])
                        st.plotly_chart(k_fig, use_container_width=True)

                        # 投资建议
                        st.markdown("#### 投资建议")
                        if advice["买入建议"]:
                            for adv in advice["买入建议"]:
                                st.markdown(f"<div class='info-box' style='padding:10px;margin:5px 0;'>✅ {adv}</div>", unsafe_allow_html=True)
                        if advice["风险提示"]:
                            for risk in advice["风险提示"]:
                                st.markdown(f"<div class='risk-box' style='padding:10px;margin:5px 0;'>⚠️ {risk}</div>", unsafe_allow_html=True)

                # 综合投资建议
                st.divider()
                st.markdown("""
                <div class="advice-box">
                    <h4>综合操作建议</h4>
                    <p>1. 严格按照推荐资金比例分配仓位，分散投资风险；</p>
                    <p>2. 尽量在建议买入价格区间内分批建仓，不追高；</p>
                    <p>3. 触及预期卖出价后，按规则止盈离场；</p>
                    <p>4. 统一设置5%~8%为止损线，控制亏损；</p>
                    <p>5. 每日跟踪大盘与个股走势，动态观察调整；</p>
                    <p>6. 关注DDM估值变化，若估值从低估转为高估，考虑减仓。</p>
                </div>
                <div class="risk-box">
                    <strong>风险提示：</strong>本系统仅基于历史数据做量化分析，DDM模型基于多项假设估算，不构成任何投资建议。股市有风险，入市需谨慎。
                </div>
                """, unsafe_allow_html=True)

        # ---- Tab4: 价格预测 ----
        with tab4:
            st.markdown("### 股票价格预测（蒙特卡洛模拟）")
            st.markdown("""
            <div class="info-box">
                基于历史收益率分布和波动率，采用蒙特卡洛模拟方法进行价格预测。
                预测结果仅供参考，不构成投资建议。
            </div>
            """, unsafe_allow_html=True)

            pred_stock = st.selectbox(
                "选择预测股票", options=list(stock_options.keys()),
                format_func=lambda x: stock_options[x], key="pred_stock_select"
            )

            if pred_stock:
                code_int = int(pred_stock)
                s_df = df_stock[df_stock["股票代码"] == code_int].sort_values("日期").reset_index(drop=True)
                stock_name = code2name.get(code_int, "未知")

                if len(s_df) >= 60:
                    pred_days = st.slider("预测天数", 10, 90, 30, key="pred_days_slider")
                    pred_df = calc_monte_carlo_prediction(s_df, predict_days=pred_days)

                    fig_pred = draw_prediction_chart(s_df, pred_df, stock_name, code_int)
                    st.plotly_chart(fig_pred, use_container_width=True)

                    last_price = s_df["收盘"].iloc[-1]
                    pred_final = pred_df["预测价格"].iloc[-1]
                    pred_return = (pred_final - last_price) / last_price * 100

                    pc1, pc2, pc3, pc4 = st.columns(4)
                    pc1.metric("当前价格", f"{last_price:.2f}元")
                    pc2.metric(f"{pred_days}日后预测价", f"{pred_final:.2f}元", f"{pred_return:+.2f}%")
                    pc3.metric("预测上界(90%)", f"{pred_df['上界'].iloc[-1]:.2f}元")
                    pc4.metric("预测下界(90%)", f"{pred_df['下界'].iloc[-1]:.2f}元")

                    with st.expander("查看预测详细数据"):
                        st.dataframe(pred_df, use_container_width=True, hide_index=True)
                else:
                    st.warning(f"{stock_name} 数据不足60条，无法进行可靠预测")

        # ---- Tab5: 风险分析 ----
        with tab5:
            st.markdown("### 沪深300成分股风险分析")
            fig_risk, risk_df = draw_risk_analysis(df_stock, code2name)
            st.plotly_chart(fig_risk, use_container_width=True)

            st.markdown("#### 风险指标排名（Top 20）")
            st.dataframe(risk_df, use_container_width=True, hide_index=True)

            st.markdown("""
            <div class="risk-box">
                <strong>风险提示：</strong>
                <p>1. 年化波动率越高，表示价格波动越大，风险越高；</p>
                <p>2. 最大回撤反映历史最大亏损幅度，是衡量风险的重要指标；</p>
                <p>3. 夏普比率衡量风险调整后收益，数值越高越好（>1为良好，>2为优秀）；</p>
                <p>4. 以上指标均基于历史数据计算，不代表未来表现。</p>
            </div>
            """, unsafe_allow_html=True)

    elif df_stock is None and data_source == "在线获取实时数据（推荐）":
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px;">
            <h2 style="color: #1e40af; margin-bottom: 20px;">欢迎使用沪深300股票智能预测分析平台</h2>
            <p style="color: #64748b; font-size: 1.1rem; max-width: 600px; margin: 0 auto 30px;">
                本平台基于akshare实时数据，集成DDM股利贴现模型估值，提供沪深300成分股的技术分析、基本面分析、智能选股、价格预测和风险评估功能。
            </p>
            <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
                <div style="background: white; padding: 24px; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); width: 200px;">
                    <div style="font-size: 2rem; margin-bottom: 8px;">📊</div>
                    <strong>实时数据</strong>
                    <p style="color: #64748b; font-size: 0.85rem;">akshare实时获取沪深300成分股行情</p>
                </div>
                <div style="background: white; padding: 24px; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); width: 200px;">
                    <div style="font-size: 2rem; margin-bottom: 8px;">💰</div>
                    <strong>DDM估值</strong>
                    <p style="color: #64748b; font-size: 0.85rem;">戈登增长模型计算股票内在价值</p>
                </div>
                <div style="background: white; padding: 24px; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); width: 200px;">
                    <div style="font-size: 2rem; margin-bottom: 8px;">📈</div>
                    <strong>技术分析</strong>
                    <p style="color: #64748b; font-size: 0.85rem;">K线、均线、MACD、RSI、布林带</p>
                </div>
                <div style="background: white; padding: 24px; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); width: 200px;">
                    <div style="font-size: 2rem; margin-bottom: 8px;">🎯</div>
                    <strong>智能选股</strong>
                    <p style="color: #64748b; font-size: 0.85rem;">三种策略+DDM估值筛选优质个股</p>
                </div>
                <div style="background: white; padding: 24px; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); width: 200px;">
                    <div style="font-size: 2rem; margin-bottom: 8px;">🔮</div>
                    <strong>价格预测</strong>
                    <p style="color: #64748b; font-size: 0.85rem;">蒙特卡洛模拟预测未来走势</p>
                </div>
            </div>
            <p style="color: #94a3b8; margin-top: 40px; font-size: 0.9rem;">
                请在左侧控制面板选择数据源和时间范围，然后点击"开始智能分析"按钮
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #94a3b8; font-size: 0.8rem; padding: 10px;">
        沪深300股票智能预测分析平台 V2.0 | 数据来源: akshare | DDM估值模型 | 本平台仅供学习研究，不构成任何投资建议
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

