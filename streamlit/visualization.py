import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# 필요한 함수들 먼저 정의
# 광고의 매체별 성과 분석하는 함수
def analyze_ads_performance(ads_idx, click_data, media_portfolio=None):
    """
    특정 광고의 매체별 성과를 분석하는 함수
    """
    
    # 1. 해당 광고의 데이터가 있는지 확인
    ads_data = click_data[click_data['ads_idx'] == ads_idx]
    if len(ads_data) == 0:
        print(f"광고 {ads_idx}에 대한 데이터가 없습니다.")
        return pd.DataFrame()
    
    # 2. 기본 성과 데이터 추출
    ads_performance = ads_data.groupby(['ads_idx', 'mda_idx']).agg({
        'click_key': 'count',
        'conversion': 'sum',
        'contract_price': 'first',
        'media_price': 'first',
        'domain': 'first',
        'ads_category': 'first'
    }).reset_index()
    
    # 컬럼명 변경
    ads_performance.columns = ['ads_idx', 'mda_idx', 'total_clicks', 'total_conversions', 
                              'contract_price', 'media_price', 'domain', 'ads_category']
    
    # 전환율 및 수익 계산
    ads_performance['cvr'] = (
        ads_performance['total_conversions'] / ads_performance['total_clicks']
    ).round(4)
    
    ads_performance['profit_per_conversion'] = (
        ads_performance['contract_price'] - ads_performance['media_price']
    )
    ads_performance['total_profit'] = (
        ads_performance['total_conversions'] * ads_performance['profit_per_conversion']
    )
    
    # 3. 날짜 범위 및 활동일 계산
    click_data_copy = click_data.copy()
    if not pd.api.types.is_datetime64_any_dtype(click_data_copy['click_date']):
        click_data_copy['click_date'] = pd.to_datetime(click_data_copy['click_date'])
    
    ads_activity = (
        click_data_copy.loc[click_data_copy['ads_idx'] == ads_idx]
                      .groupby('mda_idx')['click_date']
                      .agg(first_click='min', last_click='max')
                      .reset_index()
    )
    
    ads_activity['days_active_calc'] = (
        (ads_activity['last_click'] - ads_activity['first_click']).dt.days + 1
    )
    
    # 4. 데이터 병합
    merged = ads_performance.merge(
        ads_activity[['mda_idx', 'first_click', 'last_click', 'days_active_calc']],
        on='mda_idx', how='left'
    )
    
    # 5. 일평균 지표 계산
    merged['daily_clicks'] = merged['total_clicks'] / merged['days_active_calc']
    merged['daily_conversions'] = merged['total_conversions'] / merged['days_active_calc']
    merged['daily_profit'] = merged['total_profit'] / merged['days_active_calc']
    
    # 6. 배분 그룹 분류 (데이터가 충분한 경우에만)
    if len(merged) > 1:  # 최소 2개 이상의 매체가 있어야 중앙값 계산이 의미있음
        profit_median = merged['daily_profit'].median()
        conv_median = merged['daily_conversions'].median()
        
        merged['배분그룹'] = np.where(
            (merged['daily_profit'] >= profit_median) & (merged['daily_conversions'] >= conv_median),
            '잘 배분',
            '잘못 배분'
        )
        # 결과 정렬
        result = merged.sort_values(['배분그룹', 'daily_profit'], ascending=[True, False]).reset_index(drop=True)
    else:
        merged['배분그룹'] = '분류불가'
        result = merged.reset_index(drop=True)
    
    return result


# 전환 수/클릭 수 매체별 pie chart를 그리는 함수
def plot_share(ads_analysis, metric_col="total_conversions", top_n=5, title_prefix="전환"):
    """
    특정 광고의 매체사별 전환수/클릭수 점유율 파이 차트를 그리는 함수
    """
    if ads_analysis.empty:
        return None
    
    df_sorted = ads_analysis[["mda_idx", metric_col]].sort_values(metric_col, ascending=False)
    df_top = df_sorted.head(top_n)
    df_etc = pd.DataFrame({
        "mda_idx": ["기타"],
        metric_col: [df_sorted[metric_col][top_n:].sum()]
    })
    df_final = pd.concat([df_top, df_etc], ignore_index=True)

    fig = px.pie(
        df_final,
        values=metric_col,
        names="mda_idx",
        title=f"매체사별 {title_prefix} 점유율 (Top {top_n} + 기타)",
        hole=0.3
    )
    return fig



# def plot_conversion_share(selection, ads_analysis, top_n=5):
#     """전환수 비율 차트"""
#     # selection에 따른 정렬 ('conversions', 'cost', 'roas' 등)
#     top_media = ads_analysis.nlargest(top_n, selection)
    
#     fig = px.pie(
#         top_media, 
#         values=selection, 
#         names='media_name',
#         title=f'Top {top_n} 매체별 {selection} 비율'
#     )
    
#     return fig

def create_performance_chart(ads_analysis):
    """성과 비교 차트"""
    fig = px.bar(
        ads_analysis.head(10),
        x='media_name',
        y=['conversions', 'cost'],
        title='매체별 성과 비교',
        barmode='group'
    )
    
    fig.update_xaxes(tickangle=45)
    return fig

def display_kpi_metrics(row):
    """KPI 메트릭 표시"""
    # 숫자 변환 (안전하게)
    row = row.copy()
    row["contract_price"] = pd.to_numeric(row["contract_price"], errors="coerce")
    row["media_price"] = pd.to_numeric(row["media_price"], errors="coerce")
    total_conversions = pd.to_numeric(row["total_conversions"], errors="coerce")
    total_clicks = pd.to_numeric(row["total_clicks"], errors="coerce")
    ads_cvr = total_conversions / total_clicks if total_clicks != 0 else 0
    total_net_return, days_active = row[['total_net_return', 'days_active']]
    revenue = row["contract_price"] * total_conversions
    spend = row["media_price"] * total_conversions
    ads_roas = revenue / spend if spend != 0 else 0

    with st.container(horizontal=True, gap="medium"):
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("전체 전환 수", f"{int(total_conversions):,}", border=True)
        with col2:
            st.metric("CVR(클릭대비전환율)", f"{float(ads_cvr):.2%}",  border=True)
        with col3:
            st.metric("총 순수익", f"{int(total_net_return):,} 원",  border=True)
        with col4:
            st.metric("ROAS", f"{ads_roas:.2f}",  border=True)
        with col5:
            st.metric("광고 활성화 일수", f"{int(days_active):,} 일",  border=True)
