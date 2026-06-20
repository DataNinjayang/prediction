"""
沪深300股票智能预测分析平台 V5.0
- 内置历史数据：预置50只沪深300成分股完整历史数据
- 可选实时更新：网络可用时自动补充最新数据
- 绝对估值法: DCF现金流折现模型 + DDM股利贴现模型
- 相对估值法: PE/PB/PS/EV/EBITDA 行业对比
- 多因子综合评分模型选股
- 精美K线图 + 估值信息整合展示
- 适配 Streamlit Cloud 部署（无网络依赖）
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
import re
import random
import os
from scipy import stats

warnings.filterwarnings("ignore")
np.random.seed(42)

# ===================== 内置数据加载 =====================
@st.cache_data(ttl=None)
def load_built_in_data():
    """加载内置历史数据CSV"""
    csv_path = os.path.join(os.path.dirname(__file__), "built_in_stock_data.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["日期"] = pd.to_datetime(df["日期"])
        for col in ["开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌额", "换手率", "涨跌幅"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    return None

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

.main .block-container {
    padding-top: 2rem; padding-bottom: 3rem;
    max-width: 1440px; margin: 0 auto;
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border-radius: 16px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.06);
}

.hero-title { text-align: center; padding: 30px 0 10px; }
.hero-title h1 {
    font-size: 2.5rem; font-weight: 700;
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #06b6d4 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 8px;
}
.hero-title p { color: #64748b; font-size: 1rem; letter-spacing: 1px; }

.sidebar-header {
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
    color: white; padding: 18px 15px; border-radius: 12px;
    margin-bottom: 20px; text-align: center;
    box-shadow: 0 4px 12px rgba(30,64,175,0.3);
}

.stButton > button {
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
    color: white; border: none; border-radius: 10px;
    font-weight: 600; padding: 12px 24px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(30,64,175,0.3); width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(30,64,175,0.4);
}

.advice-box {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    padding: 24px; border-radius: 12px;
    border-left: 5px solid #1e40af; margin: 15px 0;
    box-shadow: 0 2px 8px rgba(30,64,175,0.1);
}
.risk-box {
    background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    padding: 20px; border-radius: 12px;
    border-left: 5px solid #dc2626; margin: 10px 0;
    box-shadow: 0 2px 8px rgba(220,38,38,0.1);
}
.info-box {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    padding: 20px; border-radius: 12px;
    border-left: 5px solid #16a34a; margin: 10px 0;
    box-shadow: 0 2px 8px rgba(22,163,74,0.1);
}
.valuation-box {
    background: linear-gradient(135deg, #fefce8 0%, #fef9c3 100%);
    padding: 20px; border-radius: 12px;
    border-left: 5px solid #f59e0b; margin: 10px 0;
    box-shadow: 0 2px 8px rgba(245,158,11,0.1);
}
.factor-box {
    background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
    padding: 20px; border-radius: 12px;
    border-left: 5px solid #8b5cf6; margin: 10px 0;
    box-shadow: 0 2px 8px rgba(139,92,246,0.1);
}

hr, .stDivider {
    border: none; height: 2px;
    background: linear-gradient(90deg, transparent 0%, #cbd5e1 50%, transparent 100%);
    margin: 30px 0;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px; background: #f1f5f9;
    padding: 8px; border-radius: 12px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; padding: 10px 20px; font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ===================== 行业分类映射 =====================
INDUSTRY_CATEGORIES = {
    "金融": ["银行", "保险", "证券", "多元金融", "信托", "期货", "金融"],
    "消费": ["白酒", "食品", "饮料", "家电", "零售", "服装", "化妆品", "旅游", "酒店", "消费"],
    "医药": ["医药", "生物", "医疗器械", "医疗服务", "中药", "化学制药", "医疗保健"],
    "科技": ["半导体", "电子", "软件", "计算机", "通信", "互联网", "人工智能", "科技"],
    "新能源": ["光伏", "风电", "锂电池", "新能源汽车", "储能", "氢能", "新能源"],
    "制造": ["机械", "汽车", "军工", "航空航天", "船舶", "重工", "制造"],
    "材料": ["化工", "钢铁", "有色", "建材", "造纸", "塑料", "橡胶", "材料"],
    "基建": ["建筑", "房地产", "工程", "水泥", "基建"],
    "能源": ["煤炭", "石油", "天然气", "电力", "公用事业", "能源"],
    "物流": ["交通运输", "物流", "港口", "航运", "航空", "物流"]
}


def classify_industry(industry_name):
    if not industry_name:
        return "其他"
    industry_name = str(industry_name)
    for category, keywords in INDUSTRY_CATEGORIES.items():
        for kw in keywords:
            if kw in industry_name:
                return category
    return "其他"


# ===================== 内置沪深300成分股兜底列表 =====================
# 键使用字符串格式，避免0开头的整数被解释为八进制
FALLBACK_HS300 = {
    "600519": "贵州茅台", "600036": "招商银行", "601318": "中国平安", "000858": "五粮液",
    "600900": "长江电力", "601012": "隆基绿能", "002594": "比亚迪", "600276": "恒瑞医药",
    "000333": "美的集团", "601888": "中国中免", "300750": "宁德时代", "601398": "工商银行",
    "601288": "农业银行", "601939": "建设银行", "601988": "中国银行", "600030": "中信证券",
    "600887": "伊利股份", "000568": "泸州老窖", "000001": "平安银行", "600809": "山西汾酒",
    "002415": "海康威视", "601166": "兴业银行", "600309": "万华化学", "601899": "紫金矿业",
    "300059": "东方财富", "601668": "中国建筑", "603259": "药明康德", "002352": "顺丰控股",
    "600436": "片仔癀", "601088": "中国神华", "002714": "牧原股份", "601225": "陕西煤业",
    "600031": "三一重工", "000002": "万科A", "600048": "保利发展", "601728": "中国电信",
    "002230": "科大讯飞", "601857": "中国石油", "600028": "中国石化", "601728": "中国电信",
    "300760": "迈瑞医疗", "603288": "海天味业", "600438": "通威股份", "002142": "宁波银行",
    "601601": "中国太保", "601628": "中国人寿", "601668": "中国建筑", "601766": "中国中车",
    "601390": "中国中铁", "601186": "中国铁建", "601800": "中国交建", "601618": "中国中冶",
    "601669": "中国电建", "601117": "中国化学", "601868": "中国能建", "601618": "中国中冶",
    "601989": "中国重工", "601727": "上海电气", "600089": "特变电工", "601179": "中国西电",
    "601877": "正泰电器", "600406": "国电南瑞", "601126": "四方股份", "600312": "平高电气",
    "600550": "保变电气", "600268": "国电南自", "600885": "宏发股份", "002028": "思源电气",
    "002074": "国轩高科", "300014": "亿纬锂能", "002709": "天赐材料", "002460": "赣锋锂业",
    "603799": "华友钴业", "300433": "蓝思科技", "002475": "立讯精密", "000725": "京东方A",
    "601138": "工业富联", "002241": "歌尔股份", "603501": "韦尔股份", "688981": "中芯国际",
    "600584": "长电科技", "002371": "北方华创", "603893": "瑞芯微", "300782": "卓胜微",
    "688012": "中微公司", "688008": "澜起科技", "603986": "兆易创新", "300661": "圣邦股份",
    "688396": "华润微", "688981": "中芯国际", "600703": "三安光电", "002049": "紫光国微",
    "300223": "北京君正", "603160": "汇顶科技", "688536": "思瑞浦", "300408": "三环集团",
    "300433": "蓝思科技", "000063": "中兴通讯", "600498": "烽火通信", "600487": "亨通光电",
    "002281": "光迅科技", "300308": "中际旭创", "000938": "中芯国际", "600745": "闻泰科技",
    "002156": "通富微电", "600460": "士兰微", "300373": "扬杰科技", "002185": "华天科技",
    "603005": "晶方科技", "300046": "台基股份", "300623": "捷捷微电", "300373": "扬杰科技",
    "600460": "士兰微", "300223": "北京君正", "603893": "瑞芯微", "688595": "芯海科技",
    "688608": "恒玄科技", "688099": "晶晨股份", "688385": "复旦微电", "688123": "聚辰股份",
    "688019": "安集科技", "688126": "沪硅产业", "300666": "江丰电子", "688200": "华峰测控",
    "688012": "中微公司", "688082": "盛美上海", "688072": "拓荆科技", "688037": "芯源微",
    "688120": "华海清科", "688409": "富创精密", "688361": "中科飞测", "688147": "微导纳米",
    "688502": "茂莱光学", "688103": "国力股份", "688072": "拓荆科技", "688120": "华海清科",
}


# ===================== 数据获取函数（带重试和容错） =====================
def safe_ak_call(func, max_retries=3, delay_range=(1.0, 2.5), *args, **kwargs):
    """带重试机制的akshare调用，支持更长的延迟"""
    last_error = None
    for attempt in range(max_retries):
        try:
            result = func(*args, **kwargs)
            if result is not None and (isinstance(result, pd.DataFrame) and not result.empty):
                return result
            if attempt < max_retries - 1:
                time.sleep(random.uniform(*delay_range))
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                time.sleep(random.uniform(*delay_range))
    return None


@st.cache_data(ttl=3600, show_spinner="正在获取沪深300成分股列表...")
def get_hs300_constituents():
    """获取沪深300成分股列表，带多重备用方案"""
    errors = []

    # 方案1: index_stock_cons_weight_csindex (中证指数官网，最权威)
    try:
        df = ak.index_stock_cons_weight_csindex(symbol="000300")
        if "成分券代码" in df.columns and "成分券名称" in df.columns:
            code2name = dict(zip(
                df["成分券代码"].astype(str).str.strip(),
                df["成分券名称"].astype(str).str.strip()
            ))
            if len(code2name) >= 100:
                return code2name, f"通过中证指数官网获取沪深300成分股，共 {len(code2name)} 只"
    except Exception as e:
        errors.append(f"index_stock_cons_weight_csindex 失败: {e}")

    # 方案2: index_stock_cons_csindex (中证指数，无权重)
    try:
        df = ak.index_stock_cons_csindex(symbol="000300")
        if "成分券代码" in df.columns and "成分券名称" in df.columns:
            code2name = dict(zip(
                df["成分券代码"].astype(str).str.strip(),
                df["成分券名称"].astype(str).str.strip()
            ))
            if len(code2name) >= 100:
                return code2name, f"通过中证指数获取沪深300成分股，共 {len(code2name)} 只"
    except Exception as e:
        errors.append(f"index_stock_cons_csindex 失败: {e}")

    # 方案3: index_stock_cons (旧接口)
    try:
        df = ak.index_stock_cons(symbol="000300")
        code_col = "品种代码" if "品种代码" in df.columns else "code"
        name_col = "品种名称" if "品种名称" in df.columns else "code_name"
        df["纯代码"] = df[code_col].astype(str).str.strip().str.replace(r"^(sh\.|sz\.)", "", regex=True)
        df = df[df["纯代码"].str.len() == 6]
        code2name = dict(zip(df["纯代码"], df[name_col].astype(str).str.strip()))
        if len(code2name) >= 100:
            return code2name, f"成功获取沪深300成分股，共 {len(code2name)} 只"
    except Exception as e:
        errors.append(f"index_stock_cons 失败: {e}")

    # 方案4: 使用内置兜底列表
    st.warning(f"在线接口均不可用，使用内置沪深300成分股列表。")
    return FALLBACK_HS300.copy(), f"使用内置沪深300成分股列表，共 {len(FALLBACK_HS300)} 只"


@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_individual_info(symbol):
    try:
        df = ak.stock_individual_info_em(symbol=symbol)
        return dict(zip(df["item"], df["value"]))
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_financial_indicators(symbol):
    try:
        return ak.stock_financial_analysis_indicator_em(symbol=symbol)
    except Exception:
        return None


def _normalize_hist_df(df, stock_code, stock_name):
    """统一历史数据DataFrame格式"""
    df["股票代码"] = stock_code
    df["股票名称"] = stock_name
    df["日期"] = pd.to_datetime(df["日期"])
    for col in ["开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌额", "换手率", "涨跌幅"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["日期", "开盘", "收盘", "最高", "最低"])
    df = df.sort_values("日期").reset_index(drop=True)
    return df


def get_stock_history_data(stock_code, stock_name, start_date, end_date):
    """获取单只股票历史日线数据，支持东财/新浪/腾讯三个数据源"""
    code_str = str(stock_code).zfill(6)
    start_fmt = start_date.replace("-", "")
    end_fmt = end_date.replace("-", "")

    # 方案1: stock_zh_a_hist (东方财富) - 数据质量最高
    try:
        df = safe_ak_call(ak.stock_zh_a_hist, max_retries=2, delay_range=(1.5, 3.0),
                          symbol=code_str, period="daily",
                          start_date=start_fmt, end_date=end_fmt, adjust="qfq")
        if df is not None and not df.empty and len(df) >= 20:
            df = df.rename(columns={
                "日期": "日期", "开盘": "开盘", "收盘": "收盘",
                "最高": "最高", "最低": "最低", "成交量": "成交量",
                "成交额": "成交额", "振幅": "振幅", "涨跌额": "涨跌额",
                "换手率": "换手率", "涨跌幅": "涨跌幅"
            })
            return _normalize_hist_df(df, stock_code, stock_name)
    except Exception:
        pass

    # 方案2: stock_zh_a_daily (新浪财经) - 带市场前缀
    try:
        prefix = "sh" if code_str.startswith("6") else "sz"
        df = safe_ak_call(ak.stock_zh_a_daily, max_retries=2, delay_range=(1.5, 3.0),
                          symbol=f"{prefix}{code_str}", start_date=start_date, end_date=end_date, adjust="qfq")
        if df is not None and not df.empty and len(df) >= 20:
            df = df.rename(columns={
                "date": "日期", "open": "开盘", "close": "收盘",
                "high": "最高", "low": "最低", "volume": "成交量",
                "amount": "成交额", "amplitude": "振幅", "pct_change": "涨跌幅",
                "change": "涨跌额", "turnover": "换手率"
            })
            return _normalize_hist_df(df, stock_code, stock_name)
    except Exception:
        pass

    # 方案3: stock_zh_a_hist_tx (腾讯财经) - 带市场前缀
    try:
        prefix = "sh" if code_str.startswith("6") else "sz"
        df = safe_ak_call(ak.stock_zh_a_hist_tx, max_retries=2, delay_range=(1.5, 3.0),
                          symbol=f"{prefix}{code_str}", start_date=start_date, end_date=end_date, adjust="qfq")
        if df is not None and not df.empty and len(df) >= 20:
            df = df.rename(columns={
                "date": "日期", "open": "开盘", "close": "收盘",
                "high": "最高", "low": "最低", "amount": "成交额"
            })
            # 腾讯接口没有成交量、涨跌幅等字段，用0填充
            for col in ["成交量", "振幅", "涨跌额", "换手率", "涨跌幅"]:
                if col not in df.columns:
                    df[col] = 0.0
            return _normalize_hist_df(df, stock_code, stock_name)
    except Exception:
        pass

    return None


def fetch_all_hs300_data(code2name, start_date, end_date, progress_bar, status_text):
    """批量获取沪深300成分股历史数据，带智能降速"""
    all_data = []
    success = 0
    failed = 0
    failed_codes = []

    # 限制最多获取50只，保证稳定性
    codes_to_fetch = list(code2name.items())[:50]
    total = len(codes_to_fetch)

    for i, (code, name) in enumerate(codes_to_fetch):
        progress_bar.progress(min(i / total, 0.99),
                              text=f"正在获取: {name}({code}) [{i+1}/{total}]")
        status_text.text(f"进度: {i+1}/{total} | 成功: {success} | 失败: {failed}")

        df = get_stock_history_data(code, name, start_date, end_date)
        if df is not None and len(df) >= 20:
            all_data.append(df)
            success += 1
        else:
            failed += 1
            failed_codes.append(f"{name}({code})")

        # 智能降速：每3只股票后增加延迟
        if (i + 1) % 3 == 0:
            time.sleep(random.uniform(1.5, 3.0))
        else:
            time.sleep(random.uniform(0.5, 1.2))

    progress_bar.progress(1.0, text="数据获取完成！")
    status_text.text(f"获取完成: 成功 {success} 只, 失败 {failed} 只")

    if failed > 0:
        st.info(f"以下 {min(len(failed_codes), 10)} 只股票获取失败: {', '.join(failed_codes[:10])}")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values(["股票代码", "日期"]).reset_index(drop=True)
        return combined_df, f"数据获取完成 | 共 {success} 只股票 | {len(combined_df)} 条记录"
    return None, "未能获取到有效数据"


# ===================== 绝对估值法: DCF模型 =====================
def calc_dcf_valuation(symbol, close_price, fin_df):
    try:
        revenue = 0
        net_profit = 0
        operating_cashflow = 0
        total_assets = 0
        total_equity = 0
        total_liab = 0
        roe = 0
        debt_to_assets = 0

        if fin_df is not None and not fin_df.empty:
            for col in fin_df.columns:
                col_str = str(col).lower()
                if any(k in col_str for k in ["营业收入", "营业总收入", "revenue"]):
                    try:
                        revenue = float(fin_df[col].iloc[0])
                    except:
                        pass
                if any(k in col_str for k in ["净利润", "归母净利润", "net profit"]):
                    try:
                        net_profit = float(fin_df[col].iloc[0])
                    except:
                        pass
                if any(k in col_str for k in ["经营现金流", "经营活动现金流", "operating cash flow"]):
                    try:
                        operating_cashflow = float(fin_df[col].iloc[0])
                    except:
                        pass
                if any(k in col_str for k in ["总资产", "资产总计"]):
                    try:
                        total_assets = float(fin_df[col].iloc[0])
                    except:
                        pass
                if any(k in col_str for k in ["净资产", "所有者权益", "股东权益"]):
                    try:
                        total_equity = float(fin_df[col].iloc[0])
                    except:
                        pass
                if any(k in col_str for k in ["总负债", "负债合计"]):
                    try:
                        total_liab = float(fin_df[col].iloc[0])
                    except:
                        pass
                if "净资产收益率" in str(col) or "roe" in col_str:
                    try:
                        roe = float(fin_df[col].iloc[0])
                    except:
                        pass
                if "资产负债率" in str(col):
                    try:
                        debt_to_assets = float(fin_df[col].iloc[0])
                    except:
                        pass

        info = get_stock_individual_info(symbol)
        total_shares = 0
        try:
            total_shares_str = str(info.get("总股本", "0")).replace(",", "").replace("万股", "").replace("亿", "")
            total_shares = float(total_shares_str)
            if total_shares > 1000000:
                total_shares = total_shares / 10000
        except:
            total_shares = close_price * 100

        if operating_cashflow != 0:
            fcff = operating_cashflow * 0.7
        elif net_profit != 0:
            fcff = net_profit * 0.6
        else:
            fcff = close_price * total_shares * 0.05

        risk_free_rate = 0.025
        market_risk_premium = 0.06
        beta = 1.0
        cost_of_equity = risk_free_rate + beta * market_risk_premium

        if debt_to_assets > 0:
            debt_ratio = debt_to_assets / 100 if debt_to_assets > 1 else debt_to_assets
        else:
            debt_ratio = 0.4

        cost_of_debt = 0.04
        tax_rate = 0.25
        wacc = cost_of_equity * (1 - debt_ratio) + cost_of_debt * (1 - tax_rate) * debt_ratio

        if roe > 0:
            growth_rate_high = min(roe / 100 * 0.5, 0.15)
            growth_rate_stable = min(roe / 100 * 0.2, 0.03)
        else:
            growth_rate_high = 0.08
            growth_rate_stable = 0.025

        forecast_years = 5
        fcff_list = []
        pv_list = []
        current_fcff = fcff

        for year in range(1, forecast_years + 1):
            current_fcff = current_fcff * (1 + growth_rate_high)
            pv = current_fcff / ((1 + wacc) ** year)
            fcff_list.append(current_fcff)
            pv_list.append(pv)

        pv_forecast = sum(pv_list)
        terminal_fcff = fcff_list[-1] * (1 + growth_rate_stable)
        terminal_value = terminal_fcff / (wacc - growth_rate_stable)
        pv_terminal = terminal_value / ((1 + wacc) ** forecast_years)

        enterprise_value = pv_forecast + pv_terminal

        if total_liab > 0 and total_assets > 0:
            cash = total_assets - total_liab - total_equity if total_equity > 0 else 0
            net_debt = total_liab - max(cash, 0)
        else:
            net_debt = enterprise_value * 0.3

        equity_value = enterprise_value - net_debt

        if total_shares > 0:
            intrinsic_value_per_share = equity_value / total_shares
        else:
            intrinsic_value_per_share = close_price

        premium = (close_price - intrinsic_value_per_share) / intrinsic_value_per_share * 100

        if premium < -20:
            valuation = "严重低估"; color = "#16a34a"
        elif premium < -5:
            valuation = "低估"; color = "#22c55e"
        elif premium < 5:
            valuation = "合理"; color = "#f59e0b"
        elif premium < 20:
            valuation = "高估"; color = "#f97316"
        else:
            valuation = "严重高估"; color = "#dc2626"

        return {
            "内在价值": round(intrinsic_value_per_share, 2),
            "当前价格": round(close_price, 2),
            "估值溢价": round(premium, 2),
            "估值判断": valuation,
            "估值颜色": color,
            "企业价值": round(enterprise_value / 100000000, 2),
            "股权价值": round(equity_value / 100000000, 2),
            "WACC": round(wacc * 100, 2),
            "高增长期增长率": round(growth_rate_high * 100, 2),
            "永续增长率": round(growth_rate_stable * 100, 2),
            "预测期FCFF现值": round(pv_forecast / 100000000, 2),
            "终值现值": round(pv_terminal / 100000000, 2),
            "FCFF": round(fcff / 100000000, 2),
            "ROE": round(roe, 2) if roe else "N/A",
            "净利润": round(net_profit / 100000000, 2) if net_profit else "N/A",
            "经营现金流": round(operating_cashflow / 100000000, 2) if operating_cashflow else "N/A",
        }
    except Exception:
        pass

    return {
        "内在价值": round(close_price, 2), "当前价格": round(close_price, 2),
        "估值溢价": 0, "估值判断": "数据不足", "估值颜色": "#64748b",
        "企业价值": "N/A", "股权价值": "N/A", "WACC": "N/A",
        "高增长期增长率": "N/A", "永续增长率": "N/A",
        "预测期FCFF现值": "N/A", "终值现值": "N/A",
        "FCFF": "N/A", "ROE": "N/A", "净利润": "N/A", "经营现金流": "N/A"
    }


# ===================== 绝对估值法: DDM模型 =====================
def calc_ddm_valuation(close_price, fin_df):
    try:
        if fin_df is not None and not fin_df.empty:
            eps = 0; roe = 0; pe = 0; pb = 0; dividend_yield = 0
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

            if eps <= 0 and pe > 0:
                eps = close_price / pe
            if eps <= 0:
                eps = close_price * 0.05

            payout_ratio = 0.30
            dps = eps * payout_ratio

            risk_free_rate = 0.025
            market_risk_premium = 0.06
            beta = 1.0
            r = risk_free_rate + beta * market_risk_premium

            if roe > 0:
                g = roe / 100 * (1 - payout_ratio)
            else:
                g = 0.03
            g = min(g, r * 0.8)

            if r > g and dps > 0:
                intrinsic_value = dps * (1 + g) / (r - g)
            else:
                intrinsic_value = close_price

            premium = (close_price - intrinsic_value) / intrinsic_value * 100

            if premium < -20:
                valuation = "严重低估"; color = "#16a34a"
            elif premium < -5:
                valuation = "低估"; color = "#22c55e"
            elif premium < 5:
                valuation = "合理"; color = "#f59e0b"
            elif premium < 20:
                valuation = "高估"; color = "#f97316"
            else:
                valuation = "严重高估"; color = "#dc2626"

            return {
                "内在价值": round(intrinsic_value, 2),
                "当前价格": round(close_price, 2),
                "估值溢价": round(premium, 2),
                "估值判断": valuation,
                "估值颜色": color,
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

    return {
        "内在价值": round(close_price, 2), "当前价格": round(close_price, 2),
        "估值溢价": 0, "估值判断": "数据不足", "估值颜色": "#64748b",
        "每股股利": "N/A", "折现率": "N/A", "永续增长率": "N/A",
        "ROE": "N/A", "EPS": "N/A", "PE": "N/A", "PB": "N/A", "股息率": "N/A"
    }


# ===================== 相对估值法 =====================
def calc_relative_valuation(symbol, close_price, fin_df, industry):
    try:
        pe = 0; pb = 0; ps = 0; ev_ebitda = 0; roe = 0; revenue = 0; net_profit = 0

        if fin_df is not None and not fin_df.empty:
            for col in fin_df.columns:
                col_str = str(col).lower()
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
                if "市销率" in str(col) or "ps" in col_str:
                    try:
                        ps = float(fin_df[col].iloc[0])
                    except:
                        pass
                if "ev/ebitda" in col_str or "企业价值倍数" in str(col):
                    try:
                        ev_ebitda = float(fin_df[col].iloc[0])
                    except:
                        pass
                if "净资产收益率" in str(col) or "roe" in col_str:
                    try:
                        roe = float(fin_df[col].iloc[0])
                    except:
                        pass
                if any(k in col_str for k in ["营业收入", "营业总收入"]):
                    try:
                        revenue = float(fin_df[col].iloc[0])
                    except:
                        pass
                if any(k in col_str for k in ["净利润", "归母净利润"]):
                    try:
                        net_profit = float(fin_df[col].iloc[0])
                    except:
                        pass

        if ps <= 0 and revenue > 0:
            info = get_stock_individual_info(symbol)
            total_shares = 0
            try:
                total_shares_str = str(info.get("总股本", "0")).replace(",", "").replace("万股", "").replace("亿", "")
                total_shares = float(total_shares_str)
                if total_shares > 1000000:
                    total_shares = total_shares / 10000
            except:
                total_shares = close_price * 100
            market_cap = close_price * total_shares
            if market_cap > 0 and revenue > 0:
                ps = market_cap / revenue

        industry_benchmarks = {
            "金融": {"pe": [6, 12], "pb": [0.8, 1.5], "ps": [2, 5]},
            "消费": {"pe": [20, 40], "pb": [3, 8], "ps": [2, 6]},
            "医药": {"pe": [25, 50], "pb": [3, 8], "ps": [3, 10]},
            "科技": {"pe": [30, 60], "pb": [4, 10], "ps": [4, 15]},
            "新能源": {"pe": [20, 40], "pb": [2, 6], "ps": [2, 8]},
            "制造": {"pe": [12, 25], "pb": [1.5, 3.5], "ps": [1, 4]},
            "材料": {"pe": [10, 20], "pb": [1, 2.5], "ps": [0.5, 3]},
            "基建": {"pe": [8, 18], "pb": [0.8, 2], "ps": [0.3, 2]},
            "能源": {"pe": [8, 15], "pb": [1, 2], "ps": [1, 4]},
            "物流": {"pe": [10, 20], "pb": [1, 2.5], "ps": [0.5, 3]},
            "其他": {"pe": [15, 30], "pb": [1.5, 4], "ps": [1, 5]}
        }

        category = classify_industry(industry)
        bench = industry_benchmarks.get(category, industry_benchmarks["其他"])

        def judge_ratio(value, low, high):
            if value <= 0:
                return "N/A", "#64748b"
            if value < low * 0.7:
                return "显著低估", "#16a34a"
            elif value < low:
                return "低估", "#22c55e"
            elif value <= high:
                return "合理", "#f59e0b"
            elif value <= high * 1.3:
                return "高估", "#f97316"
            else:
                return "显著高估", "#dc2626"

        pe_judge, pe_color = judge_ratio(pe, bench["pe"][0], bench["pe"][1])
        pb_judge, pb_color = judge_ratio(pb, bench["pb"][0], bench["pb"][1])
        ps_judge, ps_color = judge_ratio(ps, bench["ps"][0], bench["ps"][1])

        judges = [j for j in [pe_judge, pb_judge, ps_judge] if j != "N/A"]
        if judges:
            under_count = sum(1 for j in judges if "低估" in j)
            over_count = sum(1 for j in judges if "高估" in j)
            if under_count >= 2:
                overall = "低估"; overall_color = "#16a34a"
            elif over_count >= 2:
                overall = "高估"; overall_color = "#dc2626"
            else:
                overall = "合理"; overall_color = "#f59e0b"
        else:
            overall = "数据不足"; overall_color = "#64748b"

        return {
            "PE": round(pe, 2) if pe > 0 else "N/A",
            "PE判断": pe_judge, "PE颜色": pe_color,
            "PB": round(pb, 2) if pb > 0 else "N/A",
            "PB判断": pb_judge, "PB颜色": pb_color,
            "PS": round(ps, 2) if ps > 0 else "N/A",
            "PS判断": ps_judge, "PS颜色": ps_color,
            "EV/EBITDA": round(ev_ebitda, 2) if ev_ebitda > 0 else "N/A",
            "行业": category,
            "行业PE区间": f"{bench['pe'][0]}-{bench['pe'][1]}",
            "行业PB区间": f"{bench['pb'][0]}-{bench['pb'][1]}",
            "行业PS区间": f"{bench['ps'][0]}-{bench['ps'][1]}",
            "综合判断": overall,
            "综合颜色": overall_color
        }
    except Exception:
        pass

    return {
        "PE": "N/A", "PE判断": "N/A", "PE颜色": "#64748b",
        "PB": "N/A", "PB判断": "N/A", "PB颜色": "#64748b",
        "PS": "N/A", "PS判断": "N/A", "PS颜色": "#64748b",
        "EV/EBITDA": "N/A", "行业": "其他",
        "行业PE区间": "N/A", "行业PB区间": "N/A", "行业PS区间": "N/A",
        "综合判断": "数据不足", "综合颜色": "#64748b"
    }


# ===================== 技术指标计算 =====================
def calc_technical_indicators(df):
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


# ===================== 多因子评分模型 =====================
def calc_multi_factor_score(s_df, ddm, dcf, relative, fin_df):
    scores = {}

    value_score = 50
    if ddm["估值判断"] in ["严重低估", "低估"]:
        value_score += 25
    elif ddm["估值判断"] == "合理":
        value_score += 10
    elif ddm["估值判断"] in ["高估", "严重高估"]:
        value_score -= 20

    if dcf["估值判断"] in ["严重低估", "低估"]:
        value_score += 15
    elif dcf["估值判断"] in ["高估", "严重高估"]:
        value_score -= 10

    if relative["综合判断"] in ["显著低估", "低估"]:
        value_score += 10
    elif relative["综合判断"] in ["高估", "显著高估"]:
        value_score -= 10

    scores["价值因子"] = min(max(value_score, 0), 100)

    close = s_df["收盘"].values
    ret_5d = (close[-1] / close[-6] - 1) * 100 if len(close) >= 6 else 0
    ret_20d = (close[-1] / close[-21] - 1) * 100 if len(close) >= 21 else 0

    momentum_score = 50
    if ret_20d > 10:
        momentum_score += 25
    elif ret_20d > 5:
        momentum_score += 15
    elif ret_20d > 0:
        momentum_score += 5
    elif ret_20d < -10:
        momentum_score -= 20
    elif ret_20d < -5:
        momentum_score -= 10

    if ret_5d > 5:
        momentum_score += 5
    elif ret_5d < -5:
        momentum_score -= 5

    scores["动量因子"] = min(max(momentum_score, 0), 100)

    quality_score = 50
    roe = 0
    if fin_df is not None and not fin_df.empty:
        for col in fin_df.columns:
            if "净资产收益率" in str(col) or "roe" in str(col).lower():
                try:
                    roe = float(fin_df[col].iloc[0])
                except:
                    pass

    if roe > 0:
        if roe > 20:
            quality_score += 30
        elif roe > 15:
            quality_score += 20
        elif roe > 10:
            quality_score += 10
        elif roe < 5:
            quality_score -= 20

    volatility = s_df["涨跌幅"].iloc[-60:].std() * np.sqrt(252) if len(s_df) >= 60 else 0
    if volatility > 50:
        quality_score -= 15
    elif volatility < 20:
        quality_score += 10

    scores["质量因子"] = min(max(quality_score, 0), 100)

    s_df_ind = calc_technical_indicators(s_df)
    ma5 = s_df_ind["MA5"].iloc[-1]
    ma10 = s_df_ind["MA10"].iloc[-1]
    ma20 = s_df_ind["MA20"].iloc[-1]
    ma60 = s_df_ind["MA60"].iloc[-1]

    trend_score = 50
    if ma5 > ma10 > ma20 > ma60:
        trend_score = 100
    elif ma5 > ma10 > ma20:
        trend_score = 80
    elif ma5 > ma10:
        trend_score = 60
    elif ma5 < ma10 < ma20 < ma60:
        trend_score = 20
    elif ma5 < ma10 < ma20:
        trend_score = 30
    elif ma5 < ma10:
        trend_score = 40

    if s_df_ind["DIF"].iloc[-1] > s_df_ind["DEA"].iloc[-1]:
        trend_score += 5
    else:
        trend_score -= 5

    scores["趋势因子"] = min(max(trend_score, 0), 100)

    total_score = (
        scores["价值因子"] * 0.30 +
        scores["动量因子"] * 0.25 +
        scores["质量因子"] * 0.25 +
        scores["趋势因子"] * 0.20
    )

    scores["综合评分"] = round(total_score, 2)
    scores["20日涨幅"] = round(ret_20d, 2)
    scores["60日涨幅"] = round((close[-1] / close[-61] - 1) * 100 if len(close) >= 61 else 0, 2)
    scores["年化波动率"] = round(volatility, 2)

    return scores


# ===================== 蒙特卡洛预测 =====================
def calc_monte_carlo_prediction(df, predict_days=30):
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


# ===================== 智能选股 =====================
def stock_filter_and_pick(df, code2name):
    stock_list = df["股票代码"].unique()
    metrics = []

    for code in stock_list:
        s_df = df[df["股票代码"] == code].sort_values("日期").reset_index(drop=True)
        if len(s_df) < 60:
            continue

        close = s_df["收盘"].iloc[-1]
        code_str = str(code).zfill(6)
        info = get_stock_individual_info(code_str)
        fin_df = get_stock_financial_indicators(code_str)
        ddm = calc_ddm_valuation(close, fin_df)
        dcf = calc_dcf_valuation(code_str, close, fin_df)
        relative = calc_relative_valuation(code_str, close, fin_df, info.get("行业", ""))
        factor_scores = calc_multi_factor_score(s_df, ddm, dcf, relative, fin_df)

        industry = info.get("行业", "未知")
        category = classify_industry(industry)

        metrics.append({
            "股票代码": code,
            "股票名称": code2name.get(code, "未知"),
            "最新价": close,
            "行业": industry,
            "行业大类": category,
            "综合评分": factor_scores["综合评分"],
            "价值因子": factor_scores["价值因子"],
            "动量因子": factor_scores["动量因子"],
            "质量因子": factor_scores["质量因子"],
            "趋势因子": factor_scores["趋势因子"],
            "20日涨幅": factor_scores["20日涨幅"],
            "60日涨幅": factor_scores["60日涨幅"],
            "年化波动率": factor_scores["年化波动率"],
            "DDM估值": ddm,
            "DCF估值": dcf,
            "相对估值": relative,
            "技术数据": s_df,
        })

    if len(metrics) < 5:
        return None, "有效个股不足5只，无法完成选股"

    m_df = pd.DataFrame(metrics)
    m_df = m_df.sort_values("综合评分", ascending=False)
    selected = m_df.head(5).reset_index(drop=True)

    res = []
    today = datetime.now()

    for i, row in selected.iterrows():
        s_df = row["技术数据"]
        cur_price = row["最新价"]
        ddm = row["DDM估值"]
        dcf = row["DCF估值"]
        relative = row["相对估值"]

        hist_vol = s_df["涨跌幅"].iloc[-60:].std() if len(s_df) >= 60 else s_df["涨跌幅"].std()
        avg_daily_ret = s_df["涨跌幅"].iloc[-20:].mean()

        expected_ret = avg_daily_ret * 30
        expected_ret = max(min(expected_ret, 0.25), -0.15)

        hold = 30
        buy = round(cur_price * 0.98, 2)
        sell = round(buy * (1 + expected_ret), 2)
        sell_dt = (today + timedelta(days=hold)).strftime("%Y-%m-%d")

        pred_df = calc_monte_carlo_prediction(s_df, predict_days=60)
        advice = generate_investment_advice(row, ddm, dcf, relative, s_df)

        res.append({
            "序号": i + 1,
            "股票代码": row["股票代码"],
            "股票名称": row["股票名称"],
            "最新收盘价": round(cur_price, 2),
            "建议买入价": buy,
            "预期卖出价": sell,
            "预期收益率": round(expected_ret * 100, 2),
            "预期卖出日": sell_dt,
            "持有天数": hold,
            "综合评分": row["综合评分"],
            "价值因子": row["价值因子"],
            "动量因子": row["动量因子"],
            "质量因子": row["质量因子"],
            "趋势因子": row["趋势因子"],
            "20日涨幅": row["20日涨幅"],
            "60日涨幅": row["60日涨幅"],
            "年化波动率": row["年化波动率"],
            "行业": row["行业"],
            "行业大类": row["行业大类"],
            "DDM估值": ddm,
            "DCF估值": dcf,
            "相对估值": relative,
            "投资建议": advice,
            "K线数据": s_df.tail(120),
            "预测数据": pred_df
        })

    return res, f"多因子选股完成，共筛选出5只优质个股"


def generate_investment_advice(row, ddm, dcf, relative, s_df):
    advice_list = []
    risk_notes = []

    score = row["综合评分"]
    if score >= 80:
        suitability = "强烈推荐"; suit_color = "#16a34a"
    elif score >= 65:
        suitability = "推荐"; suit_color = "#22c55e"
    elif score >= 50:
        suitability = "中性"; suit_color = "#f59e0b"
    else:
        suitability = "谨慎"; suit_color = "#dc2626"

    if ddm["估值判断"] in ["严重低估", "低估"]:
        advice_list.append(f"DDM股利贴现模型显示该股票{ddm['估值判断']}，内在价值约{ddm['内在价值']}元，当前价格具备安全边际")
    elif ddm["估值判断"] in ["高估", "严重高估"]:
        risk_notes.append(f"DDM模型显示该股票{ddm['估值判断']}，内在价值约{ddm['内在价值']}元，注意估值回调风险")

    if dcf["估值判断"] in ["严重低估", "低估"]:
        advice_list.append(f"DCF现金流折现模型显示该股票{dcf['估值判断']}，企业价值约{dcf['企业价值']}亿元，具备投资价值")
    elif dcf["估值判断"] in ["高估", "严重高估"]:
        risk_notes.append(f"DCF模型显示该股票{dcf['估值判断']}，企业价值约{dcf['企业价值']}亿元，注意估值风险")

    if relative["综合判断"] in ["显著低估", "低估"]:
        advice_list.append(f"相对估值法(PE/PB/PS)显示该股票{relative['综合判断']}，低于行业平均水平")
    elif relative["综合判断"] in ["高估", "显著高估"]:
        risk_notes.append(f"相对估值法显示该股票{relative['综合判断']}，高于行业平均水平")

    if row["价值因子"] >= 80:
        advice_list.append(f"价值因子得分{row['价值因子']:.0f}分，估值优势明显")
    if row["动量因子"] >= 80:
        advice_list.append(f"动量因子得分{row['动量因子']:.0f}分，近期走势强劲")
    elif row["动量因子"] <= 30:
        risk_notes.append(f"动量因子得分{row['动量因子']:.0f}分，近期走势偏弱")
    if row["质量因子"] >= 80:
        advice_list.append(f"质量因子得分{row['质量因子']:.0f}分，财务质量优良")
    if row["趋势因子"] >= 80:
        advice_list.append(f"趋势因子得分{row['趋势因子']:.0f}分，技术形态良好")
    elif row["趋势因子"] <= 30:
        risk_notes.append(f"趋势因子得分{row['趋势因子']:.0f}分，技术形态偏弱")

    if row["年化波动率"] > 40:
        risk_notes.append(f"年化波动率{row['年化波动率']:.1f}%较高，价格波动剧烈")
    elif row["年化波动率"] < 20:
        advice_list.append(f"年化波动率{row['年化波动率']:.1f}%较低，价格走势稳定")

    category = row["行业大类"]
    industry = row["行业"]
    if category in ["金融", "能源"]:
        advice_list.append(f"{industry}板块受宏观经济政策和利率影响较大，建议关注政策面变化")
    elif category in ["科技", "新能源"]:
        advice_list.append(f"{industry}板块成长性高但波动大，建议控制仓位")
    elif category in ["消费", "医药"]:
        advice_list.append(f"{industry}板块防御性强，适合作为组合底仓")

    advice_list.append("建议分批建仓，首次仓位不超过总资金的30%")
    advice_list.append("设置8%-10%为止损线，若跌破关键支撑位果断止损")
    advice_list.append("到达预期卖出价后分批止盈，锁定收益")

    if score >= 70 and (ddm["估值判断"] in ["严重低估", "低估"] or dcf["估值判断"] in ["严重低估", "低估"]):
        future_suit = "非常适合投资"; future_color = "#16a34a"
    elif score >= 60:
        future_suit = "适合投资"; future_color = "#22c55e"
    elif score >= 45:
        future_suit = "谨慎投资"; future_color = "#f59e0b"
    else:
        future_suit = "不建议投资"; future_color = "#dc2626"

    return {
        "适合度评级": suitability,
        "适合度颜色": suit_color,
        "未来适合投资": future_suit,
        "未来适合颜色": future_color,
        "买入建议": advice_list,
        "风险提示": risk_notes,
        "行业": row["行业"],
        "行业大类": category
    }


# ===================== 可视化函数 =====================
def draw_stock_kline(s_df, name, code):
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
        title=f"{name}({code}) K线技术分析图",
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


def draw_factor_radar(factor_scores):
    categories = ["价值因子", "动量因子", "质量因子", "趋势因子"]
    values = [factor_scores.get(k, 0) for k in categories]
    values += [values[0]]

    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(30,64,175,0.2)',
        line=dict(color='#1e40af', width=2),
        name='因子得分'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        title="多因子评分雷达图",
        height=400,
        template="plotly_white"
    )
    return fig


def draw_index_overview(df):
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


def draw_risk_analysis(df, code2name):
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
        <p>绝对估值(DCF+DDM) | 相对估值(PE/PB/PS) | 多因子模型 | K线图 | 智能选股</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    with st.sidebar:
        st.markdown("<div class='sidebar-header'><h3>控制面板</h3></div>", unsafe_allow_html=True)

        st.subheader("1. 数据时间范围")
        col_a, col_b = st.columns(2)
        with col_a:
            start_date = st.date_input("开始日期", datetime(2025, 1, 1))
        with col_b:
            end_date = st.date_input("结束日期", datetime.now())
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        st.subheader("2. 分析设置")
        st.markdown("""
        <div class="info-box" style="font-size:0.85rem;">
            <strong>多因子模型权重:</strong><br>
            价值因子 30% | 动量因子 25%<br>
            质量因子 25% | 趋势因子 20%<br><br>
            <strong>估值方法:</strong><br>
            绝对估值: DCF + DDM<br>
            相对估值: PE/PB/PS行业对比
        </div>
        """, unsafe_allow_html=True)

        run_btn = st.button("开始智能分析", type="primary", use_container_width=True)

    # ==================== 数据加载 ====================
    df_stock = None
    code2name = {}

    # 第一步：尝试获取成分股列表
    code2name, name_msg = get_hs300_constituents()
    if code2name:
        st.sidebar.success(name_msg)
    else:
        st.sidebar.error(name_msg)

    if run_btn:
        # 第二步：优先加载内置数据
        df_built_in = load_built_in_data()

        if df_built_in is not None:
            st.info(f"已加载内置历史数据：{df_built_in['股票代码'].nunique()} 只股票，{len(df_built_in)} 条记录")
            # 根据用户选择的时间范围过滤
            df_stock = df_built_in[
                (df_built_in["日期"] >= pd.Timestamp(start_date_str)) &
                (df_built_in["日期"] <= pd.Timestamp(end_date_str))
            ].copy()
            if len(df_stock) > 0:
                # 更新code2name为实际有数据的股票
                available_codes = df_stock["股票代码"].unique().tolist()
                code2name = {c: code2name.get(c, df_stock[df_stock["股票代码"]==c]["股票名称"].iloc[0])
                             for c in available_codes}
                st.success(f"内置数据加载成功 | {df_stock['股票代码'].nunique()} 只股票 | {len(df_stock)} 条记录")
            else:
                st.warning("选定时间范围内无内置数据，尝试实时获取...")
                df_stock = None
        else:
            st.warning("未找到内置数据文件，尝试实时获取...")

        # 第三步：如果内置数据不足，尝试实时获取
        if df_stock is None or len(df_stock) == 0:
            if code2name:
                progress_bar = st.progress(0, text="准备获取数据...")
                status_text = st.empty()
                df_stock, data_msg = fetch_all_hs300_data(
                    code2name, start_date_str, end_date_str, progress_bar, status_text
                )
                if df_stock is not None:
                    st.success(data_msg)
                else:
                    st.error(data_msg)
            else:
                st.error("无可用数据源，请检查网络连接或联系管理员")

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
            "市场概览", "个股深度分析", "智能选股", "价格预测", "风险分析"
        ])

        # ---- Tab1: 市场概览 ----
        with tab1:
            st.markdown("### 沪深300成分股整体走势")
            fig_overview = draw_index_overview(df_stock)
            st.plotly_chart(fig_overview, use_container_width=True)

        # ---- Tab2: 个股深度分析 ----
        with tab2:
            st.markdown("### 选择个股进行深度估值分析")
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

                    code_str = str(code_int).zfill(6)
                    info = get_stock_individual_info(code_str)
                    fin_df = get_stock_financial_indicators(code_str)
                    ddm = calc_ddm_valuation(latest["收盘"], fin_df)
                    dcf = calc_dcf_valuation(code_str, latest["收盘"], fin_df)
                    relative = calc_relative_valuation(code_str, latest["收盘"], fin_df, info.get("行业", ""))
                    factor_scores = calc_multi_factor_score(s_df, ddm, dcf, relative, fin_df)

                    # 基本信息
                    st.markdown(f"""
                    <div class="info-box">
                        <strong>{stock_name}({code_int})</strong> |
                        <strong>行业:</strong> {info.get('行业', '未知')}（{classify_industry(info.get('行业', ''))}） |
                        <strong>总股本:</strong> {info.get('总股本', 'N/A')} |
                        <strong>总市值:</strong> {info.get('总市值', 'N/A')} |
                        <strong>上市时间:</strong> {info.get('上市时间', 'N/A')}
                    </div>
                    """, unsafe_allow_html=True)

                    # 绝对估值: DCF
                    st.markdown(f"""
                    <div class="valuation-box">
                        <h4>绝对估值法 - DCF现金流折现模型</h4>
                        <p><strong>每股内在价值:</strong> {dcf['内在价值']}元 |
                        <strong>当前价格:</strong> {dcf['当前价格']}元 |
                        <strong>估值溢价:</strong> <span style="color:{dcf['估值颜色']}">{dcf['估值溢价']}% ({dcf['估值判断']})</span></p>
                        <p><strong>企业价值(EV):</strong> {dcf['企业价值']}亿元 |
                        <strong>股权价值:</strong> {dcf['股权价值']}亿元 |
                        <strong>WACC:</strong> {dcf['WACC']}%</p>
                        <p><strong>预测期FCFF现值:</strong> {dcf['预测期FCFF现值']}亿元 |
                        <strong>终值现值:</strong> {dcf['终值现值']}亿元 |
                        <strong>FCFF:</strong> {dcf['FCFF']}亿元</p>
                        <p><strong>高增长期增长率:</strong> {dcf['高增长期增长率']}% |
                        <strong>永续增长率:</strong> {dcf['永续增长率']}%</p>
                        <p><small>两阶段DCF模型: 预测期({dcf['高增长期增长率']}%增长) + 永续期({dcf['永续增长率']}%增长)，折现率WACC={dcf['WACC']}%</small></p>
                    </div>
                    """, unsafe_allow_html=True)

                    # 绝对估值: DDM
                    st.markdown(f"""
                    <div class="valuation-box">
                        <h4>绝对估值法 - DDM股利贴现模型</h4>
                        <p><strong>每股内在价值:</strong> {ddm['内在价值']}元 |
                        <strong>当前价格:</strong> {ddm['当前价格']}元 |
                        <strong>估值溢价:</strong> <span style="color:{ddm['估值颜色']}">{ddm['估值溢价']}% ({ddm['估值判断']})</span></p>
                        <p><strong>每股股利(DPS):</strong> {ddm['每股股利']} |
                        <strong>折现率(r):</strong> {ddm['折现率']}% |
                        <strong>永续增长率(g):</strong> {ddm['永续增长率']}%</p>
                        <p><strong>ROE:</strong> {ddm['ROE']}% |
                        <strong>EPS:</strong> {ddm['EPS']} |
                        <strong>PE:</strong> {ddm['PE']} |
                        <strong>PB:</strong> {ddm['PB']}</p>
                        <p><small>戈登增长模型: V = D1 / (r - g)。折现率r = 无风险利率(2.5%) + Beta * 市场风险溢价(6%)</small></p>
                    </div>
                    """, unsafe_allow_html=True)

                    # 相对估值法
                    st.markdown(f"""
                    <div class="factor-box">
                        <h4>相对估值法 - 行业对比分析</h4>
                        <p><strong>PE:</strong> {relative['PE']} <span style="color:{relative['PE颜色']}">({relative['PE判断']})</span> |
                        <strong>行业PE区间:</strong> {relative['行业PE区间']}</p>
                        <p><strong>PB:</strong> {relative['PB']} <span style="color:{relative['PB颜色']}">({relative['PB判断']})</span> |
                        <strong>行业PB区间:</strong> {relative['行业PB区间']}</p>
                        <p><strong>PS:</strong> {relative['PS']} <span style="color:{relative['PS颜色']}">({relative['PS判断']})</span> |
                        <strong>行业PS区间:</strong> {relative['行业PS区间']}</p>
                        <p><strong>EV/EBITDA:</strong> {relative['EV/EBITDA']}</p>
                        <p><strong>综合判断:</strong> <span style="color:{relative['综合颜色']};font-weight:bold">{relative['综合判断']}</span></p>
                        <p><small>相对估值法通过与行业平均水平对比，判断股票估值高低</small></p>
                    </div>
                    """, unsafe_allow_html=True)

                    # 多因子评分
                    st.markdown(f"""
                    <div class="factor-box">
                        <h4>多因子综合评分: {factor_scores['综合评分']:.1f}分</h4>
                        <p>价值因子: {factor_scores['价值因子']:.0f}分 |
                        动量因子: {factor_scores['动量因子']:.0f}分 |
                        质量因子: {factor_scores['质量因子']:.0f}分 |
                        趋势因子: {factor_scores['趋势因子']:.0f}分</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # 雷达图
                    fig_radar = draw_factor_radar(factor_scores)
                    st.plotly_chart(fig_radar, use_container_width=True)

                    # 技术指标
                    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
                    mc1.metric("最新收盘价", f"{latest['收盘']:.2f}元")
                    mc2.metric("RSI(14)", f"{latest['RSI']:.1f}",
                              "超买" if latest['RSI'] > 70 else ("超卖" if latest['RSI'] < 30 else "中性"))
                    mc3.metric("MACD", f"{latest['MACD']:.4f}",
                              "金叉" if latest['DIF'] > latest['DEA'] else "死叉")
                    mc4.metric("20日涨幅", f"{factor_scores['20日涨幅']:.2f}%")
                    mc5.metric("60日涨幅", f"{factor_scores['60日涨幅']:.2f}%")
                    mc6.metric("年化波动率", f"{factor_scores['年化波动率']:.1f}%")

                    # K线图
                    fig_kline = draw_stock_kline(s_df_calc, stock_name, code_int)
                    st.plotly_chart(fig_kline, use_container_width=True)
                else:
                    st.warning(f"{stock_name} 数据不足30条，无法进行技术分析")

        # ---- Tab3: 智能选股 ----
        with tab3:
            st.markdown("### 多因子模型智能选股")
            st.markdown("""
            <div class="info-box">
                <strong>选股模型说明：</strong><br>
                基于价值因子(30%) + 动量因子(25%) + 质量因子(25%) + 趋势因子(20%)进行综合评分。<br>
                价值因子整合DCF现金流折现、DDM股利贴现、相对估值(PE/PB/PS)三种估值方法。<br>
                筛选出沪深300中综合得分最高的5只优质个股。
            </div>
            """, unsafe_allow_html=True)

            stock_res, pick_msg = stock_filter_and_pick(df_stock, code2name)
            if stock_res is None:
                st.error(pick_msg)
            else:
                st.success(pick_msg)

                display_cols = ["序号", "股票代码", "股票名称", "最新收盘价", "综合评分",
                               "价值因子", "动量因子", "质量因子", "趋势因子",
                               "20日涨幅", "60日涨幅", "行业", "行业大类"]
                df_res = pd.DataFrame([{k: v for k, v in item.items() if k in display_cols}
                                       for item in stock_res])
                st.dataframe(df_res, use_container_width=True, hide_index=True)
                st.divider()

                st.markdown("### 个股详细分析与投资建议")
                for item in stock_res:
                    ddm = item["DDM估值"]
                    dcf = item["DCF估值"]
                    relative = item["相对估值"]
                    advice = item["投资建议"]

                    with st.expander(
                        f"第{item['序号']}只: {item['股票名称']}({item['股票代码']}) | "
                        f"综合评分: {item['综合评分']:.1f} | "
                        f"DCF: {dcf['估值判断']} | DDM: {ddm['估值判断']} | "
                        f"适合度: {advice['适合度评级']}",
                        expanded=True
                    ):
                        # 适合度评级
                        st.markdown(f"""
                        <div style="background: {advice['适合度颜色']}15; padding: 12px; border-radius: 8px;
                                    border-left: 4px solid {advice['适合度颜色']}; margin-bottom: 10px;">
                            <strong>当前适合度评级: {advice['适合度评级']}</strong> |
                            <strong>未来投资建议: <span style="color:{advice['未来适合颜色']}">{advice['未来适合投资']}</span></strong>
                        </div>
                        """, unsafe_allow_html=True)

                        # 行业信息
                        st.markdown(f"""
                        <div class="info-box">
                            <strong>行业分类：</strong>{item['行业']}（{item['行业大类']}）<br>
                            <strong>20日涨幅：</strong>{item['20日涨幅']:.2f}% |
                            <strong>60日涨幅：</strong>{item['60日涨幅']:.2f}% |
                            <strong>年化波动率：</strong>{item['年化波动率']:.1f}%
                        </div>
                        """, unsafe_allow_html=True)

                        # 多因子评分
                        st.markdown(f"""
                        <div class="factor-box">
                            <h4>多因子评分详情</h4>
                            <p>价值因子: {item['价值因子']:.0f}分 |
                            动量因子: {item['动量因子']:.0f}分 |
                            质量因子: {item['质量因子']:.0f}分 |
                            趋势因子: {item['趋势因子']:.0f}分</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # DCF估值
                        st.markdown(f"""
                        <div class="valuation-box">
                            <h4>绝对估值 - DCF现金流折现模型</h4>
                            <p><strong>每股内在价值:</strong> {dcf['内在价值']}元 |
                            <strong>当前价格:</strong> {dcf['当前价格']}元 |
                            <strong>估值溢价:</strong> <span style="color:{dcf['估值颜色']};font-weight:bold">{dcf['估值溢价']}% ({dcf['估值判断']})</span></p>
                            <p><strong>企业价值:</strong> {dcf['企业价值']}亿元 |
                            <strong>股权价值:</strong> {dcf['股权价值']}亿元 |
                            <strong>WACC:</strong> {dcf['WACC']}%</p>
                            <p><strong>预测期FCFF现值:</strong> {dcf['预测期FCFF现值']}亿元 |
                            <strong>终值现值:</strong> {dcf['终值现值']}亿元</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # DDM估值
                        st.markdown(f"""
                        <div class="valuation-box">
                            <h4>绝对估值 - DDM股利贴现模型</h4>
                            <p><strong>每股内在价值:</strong> {ddm['内在价值']}元 |
                            <strong>当前价格:</strong> {ddm['当前价格']}元 |
                            <strong>估值溢价:</strong> <span style="color:{ddm['估值颜色']};font-weight:bold">{ddm['估值溢价']}% ({ddm['估值判断']})</span></p>
                            <p><strong>每股股利(DPS):</strong> {ddm['每股股利']} |
                            <strong>折现率(r):</strong> {ddm['折现率']}% |
                            <strong>永续增长率(g):</strong> {ddm['永续增长率']}%</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # 相对估值
                        st.markdown(f"""
                        <div class="factor-box">
                            <h4>相对估值法 - 行业对比</h4>
                            <p><strong>PE:</strong> {relative['PE']} <span style="color:{relative['PE颜色']}">({relative['PE判断']})</span> |
                            <strong>PB:</strong> {relative['PB']} <span style="color:{relative['PB颜色']}">({relative['PB判断']})</span> |
                            <strong>PS:</strong> {relative['PS']} <span style="color:{relative['PS颜色']}">({relative['PS判断']})</span></p>
                            <p><strong>综合判断:</strong> <span style="color:{relative['综合颜色']};font-weight:bold">{relative['综合判断']}</span></p>
                        </div>
                        """, unsafe_allow_html=True)

                        # 交易指标
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("建议买入价", f"{item['建议买入价']}元", f"现价 {item['最新收盘价']}元")
                        mc2.metric("预期卖出价", f"{item['预期卖出价']}元", f"收益 {item['预期收益率']}%")
                        mc3.metric("预期卖出日", item["预期卖出日"], f"持有 {item['持有天数']}天")
                        mc4.metric("综合评分", f"{item['综合评分']:.1f}分")

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

                st.divider()
                st.markdown("""
                <div class="advice-box">
                    <h4>综合操作建议</h4>
                    <p>1. 严格按照综合评分排序分配仓位，高分个股优先配置；</p>
                    <p>2. 关注DCF和DDM估值，优先选择两种绝对估值均显示低估的个股；</p>
                    <p>3. 相对估值(PE/PB/PS)作为辅助验证，与行业均值对比；</p>
                    <p>4. 建议分批建仓，首次仓位不超过总资金的20%；</p>
                    <p>5. 设置8%-10%为止损线，跌破关键支撑位果断止损；</p>
                    <p>6. 到达预期卖出价后分批止盈，锁定收益；</p>
                    <p>7. 定期关注多因子评分和估值变化，动态调整持仓。</p>
                </div>
                <div class="risk-box">
                    <strong>风险提示：</strong>本系统基于历史数据和多因子模型进行量化分析，DCF和DDM模型基于多项假设估算，
                    相对估值法基于行业均值对比，均不构成任何投资建议。股市有风险，入市需谨慎。
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

    else:
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px;">
            <h2 style="color: #1e40af; margin-bottom: 20px;">欢迎使用沪深300股票智能预测分析平台</h2>
            <p style="color: #64748b; font-size: 1.1rem; max-width: 700px; margin: 0 auto 30px;">
                本平台基于akshare实时数据，采用绝对估值法(DCF+DDM)和相对估值法(PE/PB/PS)进行多维度估值分析，
                结合多因子模型进行综合评分选股，提供精美K线图、价格预测和风险评估功能。
            </p>
            <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
                <div style="background: white; padding: 24px; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); width: 200px;">
                    <div style="font-size: 2rem; margin-bottom: 8px;">💰</div>
                    <strong>绝对估值</strong>
                    <p style="color: #64748b; font-size: 0.85rem;">DCF现金流折现 + DDM股利贴现</p>
                </div>
                <div style="background: white; padding: 24px; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); width: 200px;">
                    <div style="font-size: 2rem; margin-bottom: 8px;">📊</div>
                    <strong>相对估值</strong>
                    <p style="color: #64748b; font-size: 0.85rem;">PE/PB/PS行业对比分析</p>
                </div>
                <div style="background: white; padding: 24px; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); width: 200px;">
                    <div style="font-size: 2rem; margin-bottom: 8px;">🎯</div>
                    <strong>多因子模型</strong>
                    <p style="color: #64748b; font-size: 0.85rem;">价值+动量+质量+趋势综合评分</p>
                </div>
                <div style="background: white; padding: 24px; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); width: 200px;">
                    <div style="font-size: 2rem; margin-bottom: 8px;">📈</div>
                    <strong>K线图</strong>
                    <p style="color: #64748b; font-size: 0.85rem;">精美K线+均线+MACD+RSI+布林带</p>
                </div>
                <div style="background: white; padding: 24px; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); width: 200px;">
                    <div style="font-size: 2rem; margin-bottom: 8px;">🔮</div>
                    <strong>价格预测</strong>
                    <p style="color: #64748b; font-size: 0.85rem;">蒙特卡洛模拟预测未来走势</p>
                </div>
            </div>
            <p style="color: #94a3b8; margin-top: 40px; font-size: 0.9rem;">
                请在左侧控制面板选择数据时间范围，然后点击"开始智能分析"按钮
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #94a3b8; font-size: 0.8rem; padding: 10px;">
        沪深300股票智能预测分析平台 V4.1 | 数据来源: akshare | 绝对估值(DCF+DDM) | 相对估值(PE/PB/PS) | 本平台仅供学习研究，不构成任何投资建议
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
