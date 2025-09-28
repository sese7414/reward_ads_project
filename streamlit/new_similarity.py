# ==============================================================
# 신규(가상) 광고 → 유사 광고 코호트 기반 매체별 CVR/전환수 예측 (원샷 셀)
# ==============================================================

import numpy as np
import pandas as pd
import streamlit as st

# ----------------- 파일 경로 -----------------
PERF_CSV = "/Users/t2023-m0052/Documents/GitHub/final_project/data/수정_시간별적립보고서(최종).csv"   # 시간별 집계(또는 로그)
META_CSV = "/Users/t2023-m0052/Documents/GitHub/final_project/data/광고도메인리스트.csv"             # 기존 광고 메타(ads_idx 존재)
NEW_ADS_CSV = "/Users/Jiwon/Documents/GitHub/final_project/Jiwon/신규가상광고.csv"                                                    # 신규 가상광고 목록
EXIST_ADS_CSV = "/Users/t2023-m0052/Documents/GitHub/final_project/data/ads_pool.csv"

# ----------------- 파일 경로 -----------------
# PERF_CSV = "/Users/Jiwon/Documents/GitHub/final_project/Jiwon/수정_시간별적립보고서(최종).csv"   # 시간별 집계(또는 로그)
# META_CSV = "/Users/Jiwon/Documents/GitHub/final_project/Jiwon/광고도메인리스트.csv"             # 기존 광고 메타(ads_idx 존재)
# NEW_ADS_CSV = "/Users/Jiwon/Documents/GitHub/final_project/Jiwon/신규가상광고.csv"                                                    # 신규 가상광고 목록

# ----------------- 하이퍼파라미터 -----------------
# L_DAYS = 30                 # 예측에 사용할 과거 창 길이 # 이거를 아예 받아버릴까?
H_DAYS = 30                 # 시나리오B(향후 H일) 클릭/전환 예측 길이
K = 10                      # 유사 광고 코호트 크기
BETA_SIM = 1.0              # 유사도 가중 지수 (cos sim^beta)
ALPHA_PRIOR = 2.0           # 베타-바이노믹 스무딩 alpha
BETA_PRIOR = 120.0          # 베타-바이노믹 스무딩 beta
BLEND_KAPPA = 15.0          # 블렌딩 전환: eff/(eff+kappa)
DOMAIN_WEIGHT = 1.0         # 도메인 가중치(영향 키우려면 2~3)
RESTRICT_SAME_DOMAIN = False# True면 같은 도메인 후보만 코호트로
DROP_RARE_MIN_ADS = 3       # 희귀 원-핫 열 제거(3개 미만 광고에서만 등장)

CAT_COLS = ["domain", "ads_category", "ads_os_type", "ads_type", "ads_rejoin_type"]
PRICE_CANDIDATES = ("ads_media_price", "media_price", "contract_price")

@st.cache_data
def load_ads_pool():
    """ads_pool 로딩"""
    df = pd.read_csv(EXIST_ADS_CSV)
    return df.iloc[:, 2:]

# ----------------- 작은 유틸 -----------------
def _norm_meta(df):
    df = df.copy()
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

def _pick_date_col(df):
    for c in ["rpt_time_date","click_day","click_date"]:
        if c in df.columns:
            return c
    raise ValueError("날짜 컬럼(rpt_time_date / click_day / click_date) 없음")

def _clicks_convs_cols(df):
    clicks = None; convs = None
    if "rpt_time_clk" in df.columns: clicks = "rpt_time_clk"
    elif "clicks" in df.columns:     clicks = "clicks"
    elif "click_key" in df.columns:  clicks = None           # 로그면 size()로 산출

    if "rpt_time_turn" in df.columns: convs = "rpt_time_turn"
    elif "conversions" in df.columns: convs = "conversions"
    elif "conversion" in df.columns:  convs = "conversion"
    return clicks, convs

def _z(s):
    s = pd.to_numeric(s, errors="coerce")
    mu, sd = np.nanmean(s), np.nanstd(s)
    sd = 1.0 if (sd is None or sd == 0 or np.isnan(sd)) else sd
    return (s - mu) / sd, float(mu), float(sd)

# ----------------- 1) 기존 광고로 피처 공간 학습 -----------------
def build_feature_space(ad_meta, drop_rare_min_ads=3, domain_weight=1.0):
    meta = _norm_meta(ad_meta).drop_duplicates("ads_idx").copy()

    X_list = []
    group_cols = {}

    # (a) 범주형 원-핫
    for c in CAT_COLS:
        if c in meta.columns:
            one = pd.get_dummies(meta[c].astype(str), prefix=c, dtype=float)
            if c == "domain" and domain_weight != 1.0:
                one = one * float(domain_weight)
            X_list.append(one)
            group_cols[c] = list(one.columns)

    # (b) 수치형(가격) - 후보 중 존재하는 첫 컬럼 사용
    price_col = next((c for c in PRICE_CANDIDATES if c in meta.columns), None)
    price_mu = price_sd = None
    if price_col is not None:
        price_log = np.log1p(pd.to_numeric(meta[price_col], errors="coerce"))
        price_z, price_mu, price_sd = _z(price_log)
        X_list.append(price_z.to_frame("price_z"))

    if not X_list:
        raise ValueError("ad_meta에서 만들 수 있는 피처가 없습니다.")

    X = pd.concat(X_list, axis=1).fillna(0.0)
    # 희귀 원-핫 열 제거
    if drop_rare_min_ads and drop_rare_min_ads > 1:
        nz = (X != 0).sum(0)
        keep = nz[nz >= float(drop_rare_min_ads)].index
        X = X[keep]
        # 그룹 열 목록 갱신
        for g in list(group_cols.keys()):
            group_cols[g] = [col for col in group_cols[g] if col in X.columns]

    # z-score (열 단위)
    mu = X.mean()
    sd = X.std(ddof=0).replace(0, 1.0)
    A_z = (X - mu) / (sd + 1e-9)
    A_z.index = meta["ads_idx"].astype(int).values

    store = dict(
        A_z=A_z.astype(np.float32),
        mu=mu.astype(np.float32), sd=sd.astype(np.float32),
        cols=A_z.columns.tolist(),
        group_cols=group_cols,
        price_col=price_col,
        price_mu=price_mu, price_sd=price_sd,
        meta_small=meta[["ads_idx"] + [c for c in CAT_COLS if c in meta.columns]]
    )
    return store

# ----------------- 2) 신규 광고 1건 인코딩(기존 공간에 맞춤) -----------------
def encode_new_ad_row(row, store):
    cols = store["cols"]
    x = pd.Series(0.0, index=cols, dtype=float)

    # 범주형: 기존에 있던 열만 1로 세움(새로운 카테고리는 정보 없음 → 0)
    for c in CAT_COLS:
        if c in row.index:
            val = str(row[c]).strip()
            one_col = f"{c}_{val}"
            if one_col in x.index:
                x[one_col] = 1.0

    # 가격: log1p 후 기존 z스케일 사용
    pcol = store["price_col"]
    if pcol and pcol in row.index:
        val = pd.to_numeric(row[pcol], errors="coerce")
        if pd.notnull(val):
            z = (np.log1p(val) - store["price_mu"]) / (store["price_sd"] + 1e-9)
            if "price_z" in x.index:
                x["price_z"] = float(z)

    # L2 정규화용 벡터 반환
    a = x.values.astype(np.float32)
    a = a / (np.linalg.norm(a) + 1e-12)
    return x, a

# ----------------- 3) 신규 광고 → 유사 광고 코호트 추출 -----------------
def cohort_for_new_ad(row, store, K=50, beta=BETA_SIM, restrict_same_domain=False):
    A = store["A_z"]
    # 후보 제한(같은 도메인)
    cand = A
    if restrict_same_domain and "domain" in row.index and "domain" in store["meta_small"].columns:
        dom = str(row["domain"]).strip()
        ok_ids = store["meta_small"].loc[
            store["meta_small"]["domain"].astype(str).str.strip() == dom, "ads_idx"
        ].astype(int)
        cand = A.loc[A.index.intersection(ok_ids)]

    # 인코딩
    x, a = encode_new_ad_row(row, store)
    M = cand.values
    norms = np.sqrt((M*M).sum(1)) + 1e-12
    sims = (M @ a) / norms

    if len(sims) == 0:
        return pd.DataFrame(columns=["weight","sim"])

    k = min(K, len(sims))
    top = np.argpartition(-sims, k-1)[:k]
    top = top[np.argsort(-sims[top])]

    sim_vals = sims[top]
    w = np.power(np.clip(sim_vals, 0, 1), beta); w = w / (w.sum() + 1e-12)

    out = pd.DataFrame({"ads_idx": cand.index.values[top].astype(int),
                        "sim": sim_vals, "weight": w})
    out = out.set_index("ads_idx")
    return out

# ----------------- 4) 코호트 기반 매체사 CVR/전환수 예측 -----------------
def predict_media_from_cohort(perf_df, ad_meta_df, cohort_df, new_row,
                              L_days=30, H_days=30,
                              alpha_prior=ALPHA_PRIOR, beta_prior=BETA_PRIOR,
                              blend_kappa=BLEND_KAPPA):
    if cohort_df.empty:
        return pd.DataFrame(), {"window_end": None, "L_days": L_days, "H_days": H_days}

    perf = perf_df.copy()
    date_col = _pick_date_col(perf)
    perf[date_col] = pd.to_datetime(perf[date_col])
    wend = perf[date_col].max().normalize()
    start = wend - pd.Timedelta(days=L_days-1)
    hist = perf[(perf[date_col]>=start) & (perf[date_col]<=wend)].copy()

    clk_col, cv_col = _clicks_convs_cols(hist)

    # ads_category를 히스토리에 붙임(매체×카테 베이스라인용)
    if "ads_category" in ad_meta_df.columns and "ads_category" not in hist.columns:
        cat_map = ad_meta_df.drop_duplicates("ads_idx").set_index("ads_idx")["ads_category"]
        hist = hist.merge(cat_map.rename("ads_category"), left_on="ads_idx", right_index=True, how="left")

    # 코호트 가중 집계
    sub = hist[hist["ads_idx"].isin(cohort_df.index)].copy()
    if sub.empty:
        return pd.DataFrame(), {"window_end": str(wend.date()), "L_days": L_days, "H_days": H_days}

    if clk_col is None:
        g = sub.groupby(["ads_idx","mda_idx"]).agg(
            clicks=("ads_idx","size"), convs=("conversion","sum")
        ).reset_index()
    else:
        g = sub.groupby(["ads_idx","mda_idx"]).agg(
            clicks=(clk_col,"sum"), convs=(cv_col,"sum")
        ).reset_index()

    w_map = cohort_df["weight"].to_dict()
    g["w"] = g["ads_idx"].map(w_map).fillna(0.0)
    g["w_clicks"] = g["w"] * g["clicks"]
    g["w_convs"]  = g["w"] * g["convs"]

    agg = g.groupby("mda_idx").agg(
        cohort_eff_clicks=("w_clicks","sum"),
        cohort_eff_convs=("w_convs","sum"),
        coverage_ads=("ads_idx","nunique")
    )

    # 베이스라인 (매체 전체)
    if clk_col is None:
        base_m = hist.groupby("mda_idx").agg(
            clicks=("ads_idx","size"), convs=("conversion","sum")
        )
    else:
        base_m = hist.groupby("mda_idx").agg(
            clicks=(clk_col,"sum"), convs=(cv_col,"sum")
        )
    base_m["cvr_m"] = (base_m["convs"] + alpha_prior) / (base_m["clicks"] + alpha_prior + beta_prior)

    # 베이스라인 (매체×카테고리: 신규 광고의 카테고리 사용)
    tcat = None
    if "ads_category" in new_row.index:
        try:
            tcat = int(pd.to_numeric(new_row["ads_category"], errors="coerce"))
        except Exception:
            tcat = None

    base_mc = pd.DataFrame()
    if (tcat is not None) and ("ads_category" in hist.columns):
        subcat = hist[hist["ads_category"]==tcat]
        if not subcat.empty:
            if clk_col is None:
                base_mc = subcat.groupby("mda_idx").agg(
                    clicks=("ads_idx","size"), convs=("conversion","sum")
                )
            else:
                base_mc = subcat.groupby("mda_idx").agg(
                    clicks=(clk_col,"sum"), convs=(cv_col,"sum")
                )
            base_mc["cvr_mc"] = (base_mc["convs"] + alpha_prior) / (base_mc["clicks"] + alpha_prior + beta_prior)

    out = agg.join(base_m[["cvr_m"]], how="left").join(base_mc[["cvr_mc"]], how="left").fillna({"cvr_m":0.0})
    out["cvr_cohort"] = (out["cohort_eff_convs"] + alpha_prior) / (out["cohort_eff_clicks"] + alpha_prior + beta_prior)
    base = out["cvr_mc"].fillna(out["cvr_m"])
    eff = out["cohort_eff_clicks"]
    w1 = eff / (eff + float(blend_kappa))
    out["pred_cvr"] = w1 * out["cvr_cohort"] + (1.0 - w1) * base
    out["per_1000_clicks_conv"] = out["pred_cvr"] * 1000.0

    # 시나리오: 코호트 일평균 클릭 × H_days
    if clk_col is None:
        per_day = (sub.groupby(["mda_idx", sub[date_col].dt.normalize()])["ads_idx"]
                   .size().rename("clk").reset_index())
    else:
        per_day = (sub.groupby(["mda_idx", sub[date_col].dt.normalize()])[clk_col]
                   .sum().rename("clk").reset_index())
    daily = per_day.groupby("mda_idx")["clk"].mean()
    out["scenarioB_clicks"] = daily.reindex(out.index).fillna(0.0).values * float(H_DAYS)
    out["scenarioB_conv"]   = out["pred_cvr"] * out["scenarioB_clicks"]

    out = out.reset_index().sort_values("per_1000_clicks_conv", ascending=False).reset_index(drop=True)
    info = {"window_end": str(wend.date()), "L_days": L_days, "H_days": H_DAYS}
    return out, info

# ----------------- 5) 배치 실행: 신규 광고 목록 전체 처리 -----------------
def run_new_ads_batch(new_ads_df, L_DAYS): # 신규 광고 데이터 프레임, 예측에 사용할 과거 창 길이
    # 1) 데이터 로드
    perf_df = pd.read_csv(PERF_CSV, encoding="utf-8-sig")
    ad_meta_df = pd.read_csv(META_CSV, encoding="utf-8-sig")

    store = build_feature_space(ad_meta_df, drop_rare_min_ads=DROP_RARE_MIN_ADS,
                                domain_weight=DOMAIN_WEIGHT)

    results = {}  # key: new_ad_key → (pred_df, cohort_df, info)
    key_col = "ads_idx" if "ads_idx" in new_ads_df.columns else None

    for i, row in new_ads_df.iterrows():
        new_key = int(row[key_col]) if key_col else int(i)
        cohort = cohort_for_new_ad(row, store, K=K, beta=BETA_SIM,
                                   restrict_same_domain=RESTRICT_SAME_DOMAIN)
        pred, info = predict_media_from_cohort(perf_df, ad_meta_df, cohort, row,
                                               L_days=L_DAYS, H_days=H_DAYS,
                                               alpha_prior=ALPHA_PRIOR, beta_prior=BETA_PRIOR,
                                               blend_kappa=BLEND_KAPPA)
        results[new_key] = (pred, cohort, info)


    return results

