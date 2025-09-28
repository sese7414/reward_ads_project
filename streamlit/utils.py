import os
import re
import joblib
import pandas as pd
import numpy as np
import streamlit as st


# 데이터 파일 경로 설정
DATA_PATHS = {
    'click': "/Users/t2023-m0052/Documents/GitHub/final_project/data/유저테이블.csv",
    'ads_pool': "/Users/t2023-m0052/Documents/GitHub/final_project/data/ads_pool.csv",
    'media_portfolio': "/Users/t2023-m0052/Documents/GitHub/final_project/data/media_performance_classification.csv",
    'media_performance_classification': "/Users/t2023-m0052/Documents/GitHub/final_project/data/media_performance_classification.csv",
    'ads_list': "/Users/t2023-m0052/Documents/GitHub/final_project/Sehee/수정2_광고목록.csv",
    'ads_time': "/Users/t2023-m0052/Documents/GitHub/final_project/data/수정_시간별적립보고서(최종).csv",
    'ads_segment': "/Users/t2023-m0052/Documents/GitHub/final_project/data/ads_segment.csv",
    'new_ads_pool':"/Users/t2023-m0052/Documents/GitHub/final_project/Sehee/신규가상광고.csv",
    'model_bundle':"/Users/t2023-m0052/Documents/GitHub/final_project/streamlit/model_lightgbm.pkl"
}

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

# 매체사 포트폴리오 각 매체사별 카테고리나 도메인 전환 비중 
# 컬럼 이름에서 공백/특수문자 → '_' 로 바꾸는 헬퍼
def _slug(s):
    return re.sub(r'[^0-9A-Za-z가-힣]+', '_', str(s)).strip('_')

def add_cat_domain_to_mda_pf(
    mda_pf: pd.DataFrame,
    clicks_df: pd.DataFrame,
    conv_col: str = "conversion",
    cat_col: str = "ads_category",
    dom_col: str = "domain",
    add_within_cat: bool = False,      # mda×카테고리 내부 도메인 구성비 추가 여부
    add_within_dom: bool = False       # mda×도메인 내부 카테고리 구성비 추가 여부
):
    """
    반환: (enriched_mda_pf, new_columns)
    - conv_cat{카테고리}_{도메인} : 해당 mda의 (카테고리×도메인) 전환수
    - share_cat{카테고리}_{도메인}: 해당 mda 전체 전환 대비 구성비(0~1)
    - (옵션) shareWithinCat_*, shareWithinDomain_* 도 함께 추가 가능
    """
    df = clicks_df.copy()

    # 전환수 정리 (0/1이 아니면 그대로 합산, 0/1이면 1 합산)
    df[conv_col] = pd.to_numeric(df[conv_col], errors="coerce").fillna(0)
    if df[conv_col].max() <= 1:
        df["conv"] = (df[conv_col] > 0).astype(int)
    else:
        df["conv"] = df[conv_col]

    # 전환 있는 행만
    conv = df[df["conv"] > 0].copy()
    if conv.empty:
        enriched = mda_pf.copy()
        return enriched, []

    # mda × category × domain 전환수 집계
    g = (conv.groupby(["mda_idx", cat_col, dom_col], as_index=False)["conv"]
              .sum())

    # mda 전체 전환 합 → mda 대비 구성비
    total_mda = (g.groupby("mda_idx", as_index=False)["conv"]
                   .sum()
                   .rename(columns={"conv":"total_mda"}))
    g = g.merge(total_mda, on="mda_idx", how="left")
    g["share_mda"] = g["conv"] / g["total_mda"].replace(0, np.nan)
    g["share_mda"] = g["share_mda"].fillna(0.0)

    # ---- 피벗: 전환수 / mda-구성비
    piv_cnt = (g.pivot(index="mda_idx",
                       columns=[cat_col, dom_col],
                       values="conv")
                 .fillna(0))
    piv_shr = (g.pivot(index="mda_idx",
                       columns=[cat_col, dom_col],
                       values="share_mda")
                 .fillna(0.0))

    # 컬럼 평탄화
    piv_cnt.columns = [f"conv_cat{c}_{_slug(d)}" for c, d in piv_cnt.columns]
    piv_shr.columns = [f"share_cat{c}_{_slug(d)}" for c, d in piv_shr.columns]

    out = (mda_pf.merge(piv_cnt, on="mda_idx", how="left")
                 .merge(piv_shr, on="mda_idx", how="left"))

    new_cols = list(piv_cnt.columns) + list(piv_shr.columns)
    out[new_cols] = out[new_cols].fillna(0)

    # ---- (옵션) mda×카테고리 내부 도메인 구성비
    if add_within_cat:
        tot_cat = (g.groupby(["mda_idx", cat_col], as_index=False)["conv"]
                     .sum()
                     .rename(columns={"conv":"_tot_cat"}))
        g2 = g.merge(tot_cat, on=["mda_idx", cat_col], how="left")
        g2["share_within_cat"] = g2["conv"] / g2["_tot_cat"].replace(0, np.nan)
        piv_wc = (g2.pivot(index="mda_idx",
                           columns=[cat_col, dom_col],
                           values="share_within_cat")
                    .fillna(0.0))
        piv_wc.columns = [f"shareWithinCat_cat{c}_{_slug(d)}" for c, d in piv_wc.columns]
        out = out.merge(piv_wc, on="mda_idx", how="left")
        out[piv_wc.columns] = out[piv_wc.columns].fillna(0.0)
        new_cols += list(piv_wc.columns)

    # ---- (옵션) mda×도메인 내부 카테고리 구성비
    if add_within_dom:
        tot_dom = (g.groupby(["mda_idx", dom_col], as_index=False)["conv"]
                     .sum()
                     .rename(columns={"conv":"_tot_dom"}))
        g3 = g.merge(tot_dom, on=["mda_idx", dom_col], how="left")
        g3["share_within_domain"] = g3["conv"] / g3["_tot_dom"].replace(0, np.nan)
        piv_wd = (g3.pivot(index="mda_idx",
                           columns=[cat_col, dom_col],
                           values="share_within_domain")
                    .fillna(0.0))
        piv_wd.columns = [f"shareWithinDomain_cat{c}_{_slug(d)}" for c, d in piv_wd.columns]
        out = out.merge(piv_wd, on="mda_idx", how="left")
        out[piv_wd.columns] = out[piv_wd.columns].fillna(0.0)
        new_cols += list(piv_wd.columns)

    return out, new_cols

@st.cache_data
def load_click():
    """유저테이블(click 데이터) 로딩"""
    return pd.read_csv(DATA_PATHS["click"])

@st.cache_data
def load_ads_pool():
    """ads_pool 로딩"""
    df = pd.read_csv(DATA_PATHS["ads_pool"])
    df["ads_os_type"] = df["ads_os_type"].map(os_map).fillna("기타")
    df["ads_category"] = df["ads_category"].map(ads_category_map).fillna("기타")
    df['ads_rejoin_type'] = df['ads_rejoin_type'].map(ads_rejoin_type_map).fillna("기타")
    return df.iloc[:, 2:]


@st.cache_data
def load_ads_list():
    """ads_list 로딩"""
    return pd.read_csv(DATA_PATHS["ads_list"])

@st.cache_data
def load_media_portfolio():
    """media portfolio 로딩"""
    df = pd.read_csv(DATA_PATHS["media_portfolio"])
    return df.iloc[:, 1:]

@st.cache_data
def load_media_pf_cl():
    """media performance classification 로딩"""
    df = pd.read_csv(DATA_PATHS["media_performance_classification"])
    return df.iloc[:, 2:]

@st.cache_data
def load_ads_time():
    """ads_time 로딩"""
    return pd.read_csv(DATA_PATHS["ads_time"])

@st.cache_data
def load_ads_segment():
    """ads_segment 로딩"""
    df = pd.read_csv(DATA_PATHS["ads_segment"])
    df["ads_os_type"] = df["ads_os_type"].map(os_map).fillna("기타")
    df["ads_category"] = df["ads_category"].map(ads_category_map).fillna("기타")
    df['ads_rejoin_type'] = df['ads_rejoin_type'].map(ads_rejoin_type_map).fillna("기타")
    return df.iloc[:, 1:]

@st.cache_data
def load_new_ads_pool():
    """new_ads_pool 로딩"""
    df = pd.read_csv(DATA_PATHS["new_ads_pool"])
    df['ads_type'] = df["ads_type"].map(ads_type_map).fillna("기타")
    df["ads_os_type"] = df["ads_os_type"].map(os_map).fillna("기타")
    df["ads_category"] = df["ads_category"].map(ads_category_map).fillna("기타")
    df['ads_rejoin_type'] = df['ads_rejoin_type'].map(ads_rejoin_type_map).fillna("기타")
    return df


def load_model_bundle():
    """model_bundle 로딩"""
    return joblib.load(DATA_PATHS["model_bundle"])

@st.cache_data
def load_mda_enriched_data(mda_pf, click):
    # clicks_df: 원본 클릭/전환 테이블 (mda_idx, ads_category, domain, conversion 포함)
    # mda_pf: 매체 프로필 테이블 (mda_idx 기준)

    mda_pf_enriched, added_cols = add_cat_domain_to_mda_pf(
        mda_pf, click,
        add_within_cat=False,     # 필요하면 True
        add_within_dom=False      # 필요하면 True
    )

    return mda_pf_enriched