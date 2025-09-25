import streamlit as st
import os
import pandas as pd
import numpy as np
# import joblib

st.set_page_config(
    page_title="광고 성과 분석 및 매체사 추천",
    # page_icon=":chart_with_upwards_trend:",
    layout="wide"
)


ENV = os.getenv("APP_ENV", "local")  # 기본은 local, 배포 시 cloud로 세팅

if ENV == "local":
    # 유저테이블 불러오기
    click = pd.read_csv("/Users/t2023-m0052/Documents/GitHub/final_project/data/유저테이블.csv")

    # media portfolio 불러오기
    media_portfolio = pd.read_csv("/Users/t2023-m0052/Documents/GitHub/final_project/data/media_performance_classification.csv")
    media_portfolio = media_portfolio.iloc[:, 2:]

    # ads_pool 불러오기
    ads_pool = pd.read_csv('/Users/t2023-m0052/Documents/GitHub/final_project/data/ads_pool.csv')
    ads_pool = ads_pool.iloc[:, 2:]
    
elif ENV == "streamlit":
    # Streamlit Cloud용 예시 (Google Drive)
    url = "https://drive.google.com/uc?id=FILE_ID"
    df = pd.read_csv(url)
elif ENV == "gcp":
    # GCP GCS 예시
    df = pd.read_csv("gs://bucket_name/data.csv")


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


st.title("🔖 광고 성과 분석 및 매체사 추천")


ads_index = st.number_input("광고 인덱스", value=None, placeholder="매체별 성과를 알고 싶은 광고 idx를 입력해주세요...")

# 광고 정보로 보고 싶은 정보들
if ads_index in ads_pool['ads_idx']:
    exist = True
    data = ads_pool.loc[ads_pool['ads_idx'] == ads_index, ['ads_name', 'ads_category', 'domain', 'ads_os_type', 'ctit_mean', 'ads_size']]
    total_conversions, ads_cvr, margin, roi = ads_pool.loc[ads_pool['ads_idx'] == ads_index, ['total_conversions', 'cvr', 'margin', 'roi']]
else:
    exist = False


st.dataframe(data, use_container_width=True) # column_config=config

# 첫 번째 줄: 성과 중심 지표
col1, col2 = st.columns(2)
with col1:
    st.metric("전체 전환 수", f"{total_conversions:,}")
with col2:
    st.metric("CVR(클릭대비전환)", f"{ads_cvr:.2%}")

# 두 번째 줄: 재무 중심 지표
col3, col4 = st.columns(2)
with col3:
    st.metric("Margin", f"{margin:,} 원")
with col4:
    st.metric("ROI", f"{roi:.2f}")

ads_analysis = analyze_ads_performance(ads_index)


