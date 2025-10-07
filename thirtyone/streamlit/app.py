import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import (load_click, load_ads_segment, load_media_pf_cl, 
                   load_mda_enriched_data, load_new_ads_pool, load_model_bundle)
from visualization import analyze_ads_performance, plot_share, display_kpi_metrics
from ml_prediction import recommend_top_media, predict_evaluate_all, get_new_media_for_ad
from exist_similarity import recommend_with_weighted_similarity
from new_similarity import run_new_ads_batch


st.set_page_config(
    page_title="광고 성과 분석 및 매체사 추천",
    # page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

os_map = {
    1: "AppStore",
    2: "GooglePlay",
    3: "원스토어",
    7: "Web",
    8: "갤럭시 스토어"
}

ads_category_map = {
    0: "카테고리 선택안함",
    1: "앱(간편적립)",
    2: "경험하기(게임적립)/앱",
    3: "구독(간편적립)",
    4: "간편미션-퀴즈",
    5: "경험하기(게임적립)",
    6: "멀티보상(게임적립)",
    7: "금융(참여적립)",
    8: "무료참여(참여적립)",
    10: "유료참여(참여적립)",
    11: "쇼핑-상품별카테고리",
    12: "제휴몰(쇼핑적립)",
    13: "간편미션(간편적립)"
}

ads_type_map = {
    1:'설치형', 2:'실행형', 3:'참여형', 4:'클릭형', 5:'페북', 6:'트위터', 7:'인스타', 8:'노출형', 9:'퀘스트', 10:'유튜브', 11:'네이버', 12:'CPS(물건구매)'
}

ads_rejoin_type_map = {
    'NONE': '재참여불가', 'ADS_CODE_DAILY_UPDATE': '매일 재참여가능', 'REJOINABLE': '계속 재참여 가능'
}

@st.cache_data
def load_all_required_data():
    """모든 필수 데이터 로딩"""
    try:
        data = {
            'click': load_click(),
            'media_portfolio': load_media_pf_cl(),
            'ads_segment': load_ads_segment(),
            'new_ads_pool': load_new_ads_pool()
        }
        
        # 필수 데이터 검증
        missing_data = [key for key, value in data.items() if value is None]
        if missing_data:
            st.error(f"❌ 다음 데이터를 로딩할 수 없습니다: {missing_data}")
            return None
            
        return data
    except Exception as e:
        st.error(f"❌ 데이터 로딩 중 오류: {e}")
        return None


# 사용자 입력 받기
def get_user_input():
    """사용자 입력 받기"""
    st.title("🔖 광고 성과 분석 및 매체사 추천")
    col1, _ = st.columns([1, 2])
    
    with col1:
        ads_index = st.number_input( # 📌 광고 인덱스 입력 (정수 전용)
            "광고 인덱스",
            min_value=0,
            step=1,
            format="%d",
            placeholder="매체별 성과를 알고 싶은 광고 idx를 입력해주세요..."
        )
    
    return ads_index


def display_ml_recommendations(ads_index, model_bundle):
    """ML 기반 추천 표시"""
    st.markdown("#### 전환 수 예측 기반 추천 매체사")
    
    preds_with_counts = model_bundle['preds_with_counts']
    tgt_large = model_bundle['tgt_large']
    
    top_recommend = get_new_media_for_ad(preds_with_counts, ads_index, top=10)
    eval_df = predict_evaluate_all(preds_with_counts, tgt_large)
    
    st.dataframe(top_recommend[['mda_idx', 'yhat_turn', 'score']], 
                 column_config={
                "mda_idx": "매체사 ID",
                "yhat_turn": "예측 전환수 (5일)",  
                "score":"추천 점수" }, width='stretch')
    
    with st.expander("모델 성능지표"):
        # st.markdown("#### 📋 성능 지표 비교")
        # st.dataframe(eval_df.style.format("{:.3f}"), width='stretch')
        # 성능 지표 차트
        plot_df = eval_df.reset_index().melt(id_vars="index", var_name="metric", value_name="score")
        fig = px.bar(plot_df, x="metric", y="score", color="index", barmode="group",
                    title="Precision/Recall/MAP/HitRate 비교 (@5 vs @10)")
        st.plotly_chart(fig, width='stretch')

def display_similarity_recommendations(ads_index, click, media_portfolio):
    """유사도 기반 추천 표시"""
    st.markdown("#### 지금 진행중인 매체사와 유사한 매체사")
    
    ads_analysis = analyze_ads_performance(ads_index, click)
    mda_pf_enriched = load_mda_enriched_data(media_portfolio, click)
    
    out, anchors, feats, w = recommend_with_weighted_similarity(
        ad_df=ads_analysis,
        mda_pf=mda_pf_enriched,
        use_clr=True,
        weight_power=0.5,
        prior_mix=0.2,
        prior_from="mda_mean",
        n_anchor=5
    )
    
    if out[out['similarity'] > 0.5].shape[0] > 0:
        st.dataframe(out[out['similarity'] > 0.5].head(10),
                     column_config={
                "mda_idx": "매체사 ID",
                "similarity": "유사도", 
                "basic_classification": "총 전환 수",
                "conversion_rate": "전환율",
                "expected_total_profit": "예측 총 이익",
                "days_active": "활동 일수",
                "total_ads": "전체 광고 수"
            },
            width='stretch')
    else:
        st.warning('유사한 매체사가 존재하지 않습니다.')


def show_existing_ad_analysis(ads_index, data):
    """기존 광고 분석 화면"""
    ads_segment = data['ads_segment']
    click = data['click']
    media_portfolio = data['media_portfolio']
    model_bundle = load_model_bundle()
    st.markdown("\n")

    # 기본 정보 표시
    row = ads_segment.loc[ads_segment['ads_idx'] == ads_index].iloc[0]
    st.subheader("📋 기존 광고 기본 정보")
    data = row[['ads_name','ads_category','domain','ads_os_type','ctit_median','ads_size']].to_frame().T
    data["ads_os_type"] = data["ads_os_type"].map(os_map).fillna("기타")
    data["ads_category"] = data["ads_category"].map(ads_category_map).fillna("기타")
    data.columns = ['광고 이름','광고 카테고리','도메인','타겟 os 타입','CTIT 중앙값','광고 규모']
    st.dataframe(data, width='stretch')
    st.markdown("\n")

    # KPI 지표
    st.subheader("📊 KPI 지표")
    display_kpi_metrics(row)
    st.markdown("\n")

    # 매체사 성과 분석
    st.subheader("📑 매체사 성과 분석")
    col3, col4 = st.columns([0.7, 2])
    ads_analysis = analyze_ads_performance(ads_index, click)

    with col3:
        selection = st.segmented_control("기준", ['전환 수', '클릭 수'], selection_mode='single', default='전환 수')
        if selection == "전환 수":
            fig = plot_share(ads_analysis, "total_conversions", title_prefix="전환")
        else:
            fig = plot_share(ads_analysis, "total_clicks", title_prefix="클릭")
        st.plotly_chart(fig)
    with col4:
        st.markdown('\n\n')
        st.dataframe(
            ads_analysis.drop(columns=['ads_idx', 'domain', 'ads_category'], axis=1),
            column_config={
                "mda_idx": "매체사 ID",
                "total_clicks": "총 클릭 수", 
                "total_conversions": "총 전환 수",
                "contract_price": "계약 단가",
                "media_price": "매체 단가",
                "cvr": "전환율",
                "profit_per_conversion": "전환당 이익",
                "total_profit": "총 이익",
                "first_click": "최초 클릭일",
                "last_click": "마지막 클릭일",
                "days_active_calc": "활동 일수",
                "daily_clicks": "일평균 클릭 수",
                "daily_conversions": "일평균 전환 수",
                "daily_profit": "일평균 이익",
                "배분그룹": "배분 그룹"
            },
            width='stretch'
        )

    # 추천 매체사
    st.subheader("추천 매체사")
    display_ml_recommendations(ads_index, model_bundle)
    display_similarity_recommendations(ads_index, click, media_portfolio)




def display_new_machesa(pred):
    """신규 광고 매체사 테이블 및 그래프 보이기"""
    # st.markdown("#### 추천 매체사")

    # pred = 추천 모델이 반환한 DataFrame
    top10 = pred.head(10).copy()
    top10['mda_idx'] = top10['mda_idx'].astype(str)

    # 탭 생성
    tab1, tab2, tab3 = st.tabs(["📊 메인 (표 + 전환율)", "🥧 전환 점유율", "📈 클릭 vs 전환율"])

    # =============================
    # 탭1: 메인 (표 + 전환율)
    # =============================
    with tab1:
        st.subheader("⭐ 추천 매체사 (Top 20)")
        st.caption("예측 전환율 기준 상위 20개 매체사 추천 결과")

        col1, col2 = st.columns([1, 1])

        with col1:
            # 표 (Top 20)
            # cvr_mc 컬럼 존재 확인
            if 'cvr_mc' in pred.columns:
                view = pred.loc[pred['cvr_mc'].notna(), ["mda_idx","pred_cvr","scenarioB_clicks","scenarioB_conv"]].head(20)
            else:
                # cvr_mc가 없으면 전체 데이터 사용
                view = pred[["mda_idx","pred_cvr", "scenarioB_clicks","scenarioB_conv"]].head(20)
            
            st.dataframe(view.style.format({
                "pred_cvr":"{:.6f}", 
                "scenarioB_clicks":"{:.3f}", "scenarioB_conv":"{:.3f}"
            }), column_config={"mda_idx": "매체사 ID",
                "pred_cvr": "예측 전환율",
                "scenarioB_clicks": "예상 클릭 수",
                "scenarioB_conv": "예상 전환 수"})

        with col2:
            # 가로 막대그래프가 더 보기 좋을 수 있음
            # top10_sorted = top10.sort_values('pred_cvr', ascending=True)  # 오름차순 정렬
            # top10_sorted['mda_label'] = "매체 " + top10_sorted['mda_idx'].astype(str)
            
            fig1 = px.bar(
                top10.sort_values('pred_cvr', ascending=True),
                x="pred_cvr",
                y="mda_idx",
                orientation='h',
                text="pred_cvr",
                labels={"mda_idx": "매체사 ID", "pred_cvr": "예측 전환율"},
                title="상위 10개 매체사 예측 전환율",
                color="pred_cvr",
                color_continuous_scale="Blues"
            )
            fig1.update_traces(texttemplate="%{text:.2%}", textposition="outside")
            fig1.update_yaxes(type='category')  # ← 핵심: y축을 범주형으로 설정
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)

    # =============================
    # 탭2: 전환 점유율
    # =============================
    with tab2:
        st.subheader("🥧 전환 점유율 (Top 10)")
        st.caption("예상 전환수 기준 매체사 점유율")

        fig2 = px.pie(
            top10,
            values="scenarioB_conv",
            names="mda_idx",
            title="상위 10개 매체사 예상 전환 점유율",
            hole=0.3
        )
        st.plotly_chart(fig2, use_container_width=True)

    # =============================
    # 탭3: 클릭 vs 전환율
    # =============================
    with tab3:
        st.subheader("📈 클릭 vs 전환율")
        st.caption("예상 클릭수와 예측 전환율을 함께 비교")

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=top10["mda_idx"],
            y=top10["scenarioB_clicks"],
            name="예상 클릭수"
        ))
        fig3.add_trace(go.Scatter(
            x=top10["mda_idx"],
            y=top10["pred_cvr"],
            name="예측 전환율",
            mode="lines+markers",
            yaxis="y2"
        ))
        fig3.update_layout(
            title="상위 10개 매체사 클릭 vs 전환율",
            xaxis=dict(title="매체사 ID"),
            yaxis=dict(title="예상 클릭수"),
            yaxis2=dict(title="예측 전환율", overlaying="y", side="right", tickformat=".0%"),
            legend=dict(x=0.01, y=0.99)
        )
        fig3.update_xaxes(type='category')  # ← 핵심: y축을 범주형으로 설정
        st.plotly_chart(fig3, use_container_width=True)


def display_new_ad_recommendations(ads_index, new_ads_pool, ads_segment):
    """신규 광고 추천 결과 표시"""
    try:
        results = run_new_ads_batch(new_ads_pool, 30)
        
        if ads_index not in results:
            st.warning(f"⚠️ ads_idx {ads_index}에 대한 추천 결과가 없습니다.")
            return
            
        pred, cohort, info = results[ads_index]
        

        # 코호트 상위 10개
        if cohort is not None and not cohort.empty:
            st.markdown("#### 유사 광고 코호트")
            disp = cohort.reset_index()[["ads_idx","sim","weight"]].head(10)
            
            # 코호트 광고 특성
            cohort_ads = ads_segment.merge(disp, on='ads_idx', how='right')
            cohort_ads["ads_os_type"] = cohort_ads["ads_os_type"].map(os_map).fillna("기타")
            cohort_ads["ads_category"] = cohort_ads["ads_category"].map(ads_category_map).fillna("기타")
            cohort_ads["ads_rejoin_type"] = cohort_ads["ads_rejoin_type"].map(ads_rejoin_type_map).fillna("기타")

            st.dataframe(cohort_ads[['ads_idx', 'ads_name', 'media_count', 'user_count', 'total_clicks',
                        'total_conversions', 'ads_category', 'domain', 'ads_os_type',
                         'ctit_median', 'ads_rejoin_type', 'contract_price', 'media_price', 
                         'first_click', 'last_click', 'ads_sdate', 'expire', 'days_active',
                    'daily_avg_conversions', 'cvr', 'margin', 'roi', 'total_net_return',
                    'daily_clicks', 'daily_users', 'ads_size', 'cluster', 'sim']], column_config={
                    "ads_idx": "광고 ID",
                    "media_count": "매체사 수",
                    "user_count": "참여 사용자 수",
                    "total_clicks": "총 클릭 수",
                    "total_conversions": "총 전환 수",
                    "ads_category": "광고 카테고리",
                    "domain": "도메인",
                    "ads_os_type": "타겟 OS 타입",
                    "ctit_median": "CTIT 중앙값",
                    "ads_rejoin_type": "재참여 가능 타입",
                    "contract_price": "계약 단가",
                    "media_price": "매체 단가",
                    "first_click": "최초 클릭일",
                    "last_click": "마지막 클릭일",
                    "ads_name": "광고 이름",
                    "ads_sdate": "광고 시작일",
                    "expire": "만료일",
                    "days_active": "활동 일수",
                    "daily_avg_conversions": "일평균 전환 수",
                    "cvr": "전환율(CVR)",
                    "margin": "마진",
                    "roi": "ROI(투자수익률)",
                    "total_net_return": "총 순수익",
                    "daily_clicks": "일평균 클릭 수",
                    "daily_users": "일평균 사용자 수",
                    "ads_size": "광고 규모",
                    "cluster": "클러스터",
                    "sim":"유사도"
                    }, width='stretch')
        
        # 매체사 추천 테이블
        if pred is not None and not pred.empty:
            display_new_machesa(pred)
        else:
            st.warning("⚠️ 추천할 매체사가 없습니다.")
            
    except Exception as e:
        st.error(f"❌ 추천 처리 중 오류: {e}")



def show_new_ad_recommendation(ads_index, data):
    """신규 광고 추천 화면"""
    new_ads_pool = data['new_ads_pool']
    ads_segment = data['ads_segment']

    # 기본 정보 표시
    st.subheader("📋 신규 광고 기본 정보") # 🆕
    row = new_ads_pool.loc[new_ads_pool['ads_idx'] == ads_index].copy()
    data = row[['ads_name', 'ads_type', 'ads_category','domain','ads_os_type', 'ads_contract_price', 'ads_reward_price', 'ads_rejoin_type']]
    data["ads_type"] = data["ads_type"].map(ads_type_map).fillna("기타")
    data["ads_os_type"] = data["ads_os_type"].map(os_map).fillna("기타")
    data["ads_category"] = data["ads_category"].map(ads_category_map).fillna("기타")
    data["ads_rejoin_type"] = data["ads_rejoin_type"].map(ads_rejoin_type_map).fillna("기타")

    data.columns = ['광고 이름', '광고 타입', '광고 카테고리','도메인','타겟 os 타입','계약 단가','리워드 단가', '재참여 가능 타입']
    st.dataframe(data, width='stretch')

    # 추천 결과
    st.subheader("⭐ 추천 매체사")
    display_new_ad_recommendations(ads_index, new_ads_pool, ads_segment)




def main():
    """메인 함수"""
    ENV = os.getenv("APP_ENV", "local")  # 기본은 local, 배포 시 cloud로 세팅

    if ENV == "local":
        data = load_all_required_data()
        
        if data is None:
            st.error("❌ 필수 데이터를 로딩할 수 없습니다. 관리자에게 문의하세요.")
            return
        
        
    elif ENV == "streamlit":
        # Streamlit Cloud용 예시 (Google Drive)
        url = "https://drive.google.com/uc?id=FILE_ID"
        df = pd.read_csv(url)
    elif ENV == "gcp":
        # GCP GCS 예시
        df = pd.read_csv("gs://bucket_name/data.csv")

     
    
    # 2. UI 설정 및 사용자 입력
    ads_index = get_user_input()
    
    # 3. 입력값이 없으면 종료
    if ads_index is None or ads_index == 0:
        st.info("📌 광고 인덱스를 입력해주세요.")
        return
    
    # 4. 단순한 분기
    if ads_index in data['ads_segment']['ads_idx'].values:
        show_existing_ad_analysis(ads_index, data)
    elif ads_index in data['new_ads_pool']['ads_idx'].values:
        show_new_ad_recommendation(ads_index, data)
    else:
        st.warning(f"⚠️ ads_idx {ads_index} 결과가 없습니다. 올바른 광고 번호를 입력했는지 확인해주세요.")


if __name__ == "__main__":
    main()