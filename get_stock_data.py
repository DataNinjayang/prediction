#!/usr/bin/env python3
"""
沪深300数据提取器 - Streamlit可视化网页版 V1.0
- 登录baostock获取沪深300成分股
- 获取历史量价数据 + PE/PB/PS/PCF估值指标
- 支持增量更新
- 导出CSV供后续预测分析使用
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
from io import BytesIO

st.set_page_config(
    page_title="沪深300数据提取器",
    page_icon="📥",
    layout="wide"
)

st.markdown("""
<style>
.main {font-family: 'Microsoft YaHei', sans-serif;}
.block-container {padding-top: 1rem; max-width: 1200px;}
.stButton>button {background-color: #1e40af; color: white; border-radius: 8px; font-weight: 600; width: 100%;}
.success-box {background: #ecfdf5; padding: 15px; border-radius: 8px; border-left: 4px solid #10b981; margin: 10px 0;}
.info-box {background: #eff6ff; padding: 15px; border-radius: 8px; border-left: 4px solid #1e40af; margin: 10px 0;}
.warning-box {background: #fffbeb; padding: 15px; border-radius: 8px; border-left: 4px solid #f59e0b; margin: 10px 0;}
.error-box {background: #fef2f2; padding: 15px; border-radius: 8px; border-left: 4px solid #dc2626; margin: 10px 0;}
.metric-card {background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center'>📥 沪深300数据提取器</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#64748b;'>使用baostock获取成分股历史数据 + 估值指标 | 导出CSV供预测分析使用</p>", unsafe_allow_html=True)
st.divider()

# ===================== 核心函数 =====================
def get_hs300_stocks():
    """获取沪深300成分股列表"""
    try:
        import baostock as bs
        lg = bs.login()
        if lg.error_code != '0':
            return None, f"登录失败: {lg.error_msg}"
        
        rs = bs.query_hs300_stocks()
        if rs.error_code != '0':
            bs.logout()
            return None, f"获取成分股失败: {rs.error_msg}"
        
        stocks = []
        while (rs.error_code == '0') & rs.next():
            stocks.append(rs.get_row_data())
        
        df = pd.DataFrame(stocks, columns=rs.fields)
        bs.logout()
        return df, f"成功获取 {len(df)} 只沪深300成分股"
    except Exception as e:
        return None, f"获取失败: {str(e)}"


def get_stock_history(bs_code, start_date, end_date):
    """获取单只股票历史数据（含估值指标）"""
    try:
        import baostock as bs
        # 价格数据 + 估值指标
        fields = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM"
        rs = bs.query_history_k_data_plus(
            bs_code, fields,
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="1"
        )
        
        if rs.error_code != '0':
            return None
        
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            return None
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        # 转换数据类型
        numeric_cols = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 
                        'turn', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 计算振幅和涨跌额
        df['振幅'] = ((df['high'] - df['low']) / df['preclose'] * 100).round(2)
        df['涨跌额'] = (df['close'] - df['preclose']).round(2)
        
        # 转换日期格式 YYYY/M/D
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y/%-m/%-d')
        
        # 提取纯数字股票代码
        df['code'] = df['code'].str.replace('sh.', '').str.replace('sz.', '')
        df['code'] = df['code'].str.zfill(6)
        
        # 重命名列
        df = df.rename(columns={
            'code': '股票代码',
            'date': '日期',
            'open': '开盘',
            'close': '收盘',
            'high': '最高',
            'low': '最低',
            'volume': '成交量',
            'amount': '成交额',
            'turn': '换手率',
            'pctChg': '涨跌幅',
            'peTTM': 'PE_TTM',
            'pbMRQ': 'PB_MRQ',
            'psTTM': 'PS_TTM',
            'pcfNcfTTM': 'PCF_TTM'
        })
        
        columns = ['股票代码', '日期', '开盘', '收盘', '最高', '最低', 
                   '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
                   'PE_TTM', 'PB_MRQ', 'PS_TTM', 'PCF_TTM']
        return df[columns]
    except Exception:
        return None


# ===================== 页面布局 =====================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("⚙️ 提取设置")
    
    start_date = st.date_input("开始日期", datetime(2024, 1, 1))
    end_date = st.date_input("结束日期", datetime.now())
    
    st.markdown("""
    <div class="info-box" style="font-size:0.85rem;">
    <strong>获取字段说明:</strong><br>
    • 基础数据: 开/收/高/低/成交量/成交额<br>
    • 衍生指标: 振幅/涨跌额/换手率/涨跌幅<br>
    • 估值指标: PE_TTM / PB_MRQ / PS_TTM / PCF_TTM<br><br>
    <strong>用途:</strong><br>
    • 绝对估值: DCF现金流折现 + DDM股利贴现<br>
    • 相对估值: PE/PB/PS行业对比分析<br>
    • 多因子选股: 价值/动量/质量/趋势
    </div>
    """, unsafe_allow_html=True)
    
    extract_btn = st.button("🚀 开始提取数据", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 提取进度")
    progress_bar = st.progress(0, text="等待开始...")
    status_text = st.empty()
    log_container = st.container()

st.divider()

# ===================== 数据提取逻辑 =====================
if extract_btn:
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    status_text.text("正在登录baostock...")
    progress_bar.progress(0.05, text="登录baostock...")
    
    hs300_df, msg = get_hs300_stocks()
    
    if hs300_df is None:
        st.markdown(f"""
        <div class="error-box">
        <strong>连接失败:</strong> {msg}<br><br>
        1. 请检查网络连接是否正常<br>
        2. baostock服务器可能暂时不可用，请稍后再试<br>
        3. 如果多次失败，可以使用之前提取的CSV文件继续分析
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success(msg)
        
        total = len(hs300_df)
        all_data = []
        failed_stocks = []
        
        status_text.text(f"开始获取 {total} 只股票的历史数据...")
        
        log_lines = []
        log_placeholder = log_container.empty()
        
        for idx, row in hs300_df.iterrows():
            bs_code = row.get('code', '')
            stock_name = row.get('code_name', '')
            
            progress = min((idx + 1) / total, 0.99)
            progress_bar.progress(progress, text=f"正在获取: {stock_name} ({bs_code}) [{idx+1}/{total}]")
            
            try:
                import baostock as bs
                lg = bs.login()
                stock_data = get_stock_history(bs_code, start_str, end_str)
                bs.logout()
                
                if stock_data is not None and len(stock_data) > 0:
                    all_data.append(stock_data)
                    log_lines.append(f"✓ {stock_name} ({bs_code}): {len(stock_data)}条")
                else:
                    failed_stocks.append((bs_code, stock_name))
                    log_lines.append(f"✗ {stock_name} ({bs_code}): 无数据")
            except Exception as e:
                failed_stocks.append((bs_code, stock_name))
                log_lines.append(f"✗ {stock_name} ({bs_code}): {str(e)[:50]}")
            
            if (idx + 1) % 5 == 0 or idx == total - 1:
                log_text = "\n".join(log_lines[-20:])
                log_placeholder.markdown(f"""
                <div style="background:#f8fafc; padding:10px; border-radius:5px; font-size:0.8rem; max-height:300px; overflow-y:auto;">
                <pre>{log_text}</pre>
                </div>
                """)
            
            time.sleep(0.3)
        
        progress_bar.progress(1.0, text="数据提取完成！")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            st.divider()
            st.subheader("📈 提取结果")
            
            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.markdown(f"""
                <div class="metric-card">
                <h2 style="color:#1e40af; margin:0;">{len(combined_df):,}</h2>
                <p style="color:#64748b; margin:5px 0;">总数据行数</p>
                </div>
                """, unsafe_allow_html=True)
            with mc2:
                st.markdown(f"""
                <div class="metric-card">
                <h2 style="color:#16a34a; margin:0;">{combined_df['股票代码'].nunique()}</h2>
                <p style="color:#64748b; margin:5px 0;">覆盖股票数</p>
                </div>
                """, unsafe_allow_html=True)
            with mc3:
                st.markdown(f"""
                <div class="metric-card">
                <h2 style="color:#f59e0b; margin:0;">{len(all_data)}/{total}</h2>
                <p style="color:#64748b; margin:5px 0;">成功获取</p>
                </div>
                """, unsafe_allow_html=True)
            with mc4:
                st.markdown(f"""
                <div class="metric-card">
                <h2 style="color:#dc2626; margin:0;">{len(failed_stocks)}</h2>
                <p style="color:#64748b; margin:5px 0;">失败数量</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="success-box">
            <strong>数据时间范围:</strong> {combined_df['日期'].min()} 至 {combined_df['日期'].max()}<br>
            <strong>估值指标覆盖:</strong> PE_TTM非空 {combined_df['PE_TTM'].notna().sum()} 条 | 
            PB_MRQ非空 {combined_df['PB_MRQ'].notna().sum()} 条 | 
            PS_TTM非空 {combined_df['PS_TTM'].notna().sum()} 条 | 
            PCF_TTM非空 {combined_df['PCF_TTM'].notna().sum()} 条
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("查看数据样例"):
                st.dataframe(combined_df.head(20), use_container_width=True, hide_index=True)
            
            # CSV下载
            csv_buffer = BytesIO()
            combined_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            csv_buffer.seek(0)
            
            st.download_button(
                label="📥 下载 stock_data.csv (供预测分析使用)",
                data=csv_buffer,
                file_name="stock_data.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # 同时下载成分股列表
            hs300_buffer = BytesIO()
            hs300_df.to_csv(hs300_buffer, index=False, encoding='utf-8-sig')
            hs300_buffer.seek(0)
            
            st.download_button(
                label="📥 下载 hs300_stock_list.csv (成分股列表)",
                data=hs300_buffer,
                file_name="hs300_stock_list.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.markdown("""
            <div class="info-box">
            <strong>下一步操作:</strong><br>
            1. 下载上面两个CSV文件<br>
            2. 打开「沪深300股票预测分析」网页<br>
            3. 上传这两个文件进行分析
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="error-box">
            <strong>未能获取到任何数据</strong><br>
            请检查网络连接或baostock服务状态
            </div>
            """, unsafe_allow_html=True)

st.divider()
st.markdown("<p style='text-align:center; color:#94a3b8; font-size:0.8rem;'>数据来源: baostock | 仅供学习研究使用</p>", unsafe_allow_html=True)
