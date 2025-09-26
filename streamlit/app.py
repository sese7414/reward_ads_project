import streamlit as st
from utils import (load_click, load_ads_pool, load_ads_list, 
                    load_media_portfolio, load_media_pf_cl, load_ads_time, 
                    load_ads_segment, load_mda_enriched_date)
from visualization import analyze_ads_performance, plot_share, display_kpi_metrics
import plotly.express as px
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
    click = load_click()
    # media portfolio 불러오기
    media_portfolio = load_media_pf_cl()
    # ads_pool 불러오기
    ads_pool = load_ads_pool()
    # ads_list 불러오기
    ads_list = load_ads_list()
    # ads_segment 불러오기
    ads_segment = load_ads_segment()
    # ads_time 불러오기
    ads_time = load_ads_time()
    
elif ENV == "streamlit":
    # Streamlit Cloud용 예시 (Google Drive)
    url = "https://drive.google.com/uc?id=FILE_ID"
    df = pd.read_csv(url)
elif ENV == "gcp":
    # GCP GCS 예시
    df = pd.read_csv("gs://bucket_name/data.csv")





# 실제 타이틀부터 웹페이지 구성

st.title("🔖 광고 성과 분석 및 매체사 추천")

# 📌 광고 인덱스 입력 (정수 전용)
ads_index = st.number_input(
    "광고 인덱스",
    min_value=0,
    step=1,
    format="%d",
    placeholder="매체별 성과를 알고 싶은 광고 idx를 입력해주세요..."
)

# 📌 광고 인덱스가 입력된 경우만 실행
if ads_index is not None:   # 또는 ads_index is not None
    if ads_index in ads_pool['ads_idx'].values:
        exist = True
        row = ads_pool.loc[ads_pool['ads_idx'] == ads_index].iloc[0]
        data = row[['ads_name', 'ads_category', 'domain', 'ads_os_type', 'ctit_mean', 'ads_size']]

    else:
        exist = False

    if exist:
        st.subheader("📋 광고 기본 정보")
        st.dataframe(pd.DataFrame([data]), use_container_width=True)

        st.subheader("📊 KPI 지표")
        display_kpi_metrics()

        st.subheader("📑 매체사 성과 분석")
        col5, col6 = st.columns([0.7, 2])
        ads_analysis = analyze_ads_performance(ads_index, click)

        with col5:
            selection = st.segmented_control("기준", ['전환 수', '클릭 수'], selection_mode='single')
            if selection == "전환 수":
                fig = plot_share(ads_analysis, "total_conversions", title_prefix="전환")
            else:
                fig = plot_share(ads_analysis, "total_clicks", title_prefix="클릭")

            st.plotly_chart(fig) #  `use_container_width=False`, use `width='content'`
        with col6:
            st.markdown('\n')
            st.markdown('\n')
            st.dataframe(ads_analysis.drop(columns=['domain', 'ads_category'], axis = 1), width='stretch')

    else:
        st.warning("해당 광고 인덱스에 대한 데이터가 없습니다.")