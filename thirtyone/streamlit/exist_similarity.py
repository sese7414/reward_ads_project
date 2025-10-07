import numpy as np
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import streamlit as st

# === 통합 셀: 유사도 추천기 (CLR + prior + power + IDF + 추가 비율 피처) ===

# --- 유틸 ---
def _slug(s): 
    return re.sub(r'[^0-9A-Za-z가-힣]+', '_', str(s)).strip('_')

def _cosine(a, B):
    a = a.reshape(1, -1)
    num = (B * a).sum(axis=1)
    den = (np.sqrt((B**2).sum(axis=1)) * np.sqrt((a**2).sum()))
    den = np.where(den == 0, 1e-12, den)
    return (num / den).ravel()

def cosine_vec(a, B):
    return _cosine(np.asarray(a, dtype=float), np.asarray(B, dtype=float))

# --- 광고 데이터로 (카테고리×도메인) 가중치 + prior 스무딩 ---
def make_ad_pair_weights_from_ad_df(
    ad_df, cat_col='ads_category', dom_col='domain',
    conv_col='total_conversions', power=1.0, min_frac=0.0,
    prior_mix=0.0, prior_bg=None
):
    """
    (카테×도메인) 전환 분포 -> share_cat{c}_{slug} 가중치 dict
    power<1: 퍼짐, >1: 집중 / prior_mix: 배경분포 섞기
    """
    t = ad_df.copy()
    t[conv_col] = pd.to_numeric(t[conv_col], errors='coerce').fillna(0.0)
    g = (t.groupby([cat_col, dom_col])[conv_col].sum()
           .rename('conv').reset_index())
    tot = g['conv'].sum()
    if tot <= 0:
        return {}
    g['frac'] = g['conv'] / tot
    if min_frac > 0:
        g = g[g['frac'] >= min_frac].copy()
    g['w'] = (g['frac'] ** power)
    s = g['w'].sum()
    if s > 0:
        g['w'] = g['w'] / s
    g['key'] = [f"share_cat{int(c)}_{_slug(d)}" for c,d in g[[cat_col, dom_col]].itertuples(index=False)]
    w = dict(zip(g['key'], g['w']))

    # prior 스무딩
    if prior_mix and prior_mix > 0:
        if (prior_bg is None) or (len(prior_bg) == 0):
            prior_bg = {k: 1.0/len(w) for k in w.keys()}
        keys = set(w) | set(prior_bg)
        out = {}
        for k in keys:
            pw = w.get(k, 0.0)
            q  = prior_bg.get(k, 0.0)
            out[k] = (1.0 - prior_mix) * pw + prior_mix * q
        Z = sum(out.values()) or 1.0
        w = {k: v/Z for k,v in out.items()}
    return w

# --- 구성비 CLR 변환 ---
def _clr_block(df_block, eps=1e-6):
    Z = df_block.clip(lower=eps)
    g = np.exp(np.log(Z).mean(axis=1))
    return np.log(Z.div(g, axis=0))

# --- 피처 행렬 (share + 볼륨 + 각종 비율 + CLR + 가중 + z-score) ---
def build_feature_matrix_plus(
    mda_pf,
    share_cols,                 
    volume_cols=None,           
    size_ratio_cols=None,       
    os_ratio_cols=None,         
    category_ratio_cols=None,   
    domain_ratio_cols=None,     
    use_clr=True,               
    col_weights=None,           
    zscore=True
):
    volume_cols         = list(volume_cols or [])
    size_ratio_cols     = list(size_ratio_cols or [])
    os_ratio_cols       = list(os_ratio_cols or [])
    category_ratio_cols = list(category_ratio_cols or [])
    domain_ratio_cols   = list(domain_ratio_cols or [])

    all_cols = (list(share_cols) + volume_cols + size_ratio_cols +
                os_ratio_cols + category_ratio_cols + domain_ratio_cols)

    X = mda_pf.set_index('mda_idx')[all_cols].astype(float).copy()

    # 결측
    X[volume_cols] = X[volume_cols].fillna(0.0)
    X[size_ratio_cols + os_ratio_cols + category_ratio_cols + domain_ratio_cols + share_cols] = \
        X[size_ratio_cols + os_ratio_cols + category_ratio_cols + domain_ratio_cols + share_cols].fillna(0.0)

    # 볼륨: log1p
    if volume_cols:
        X[volume_cols] = np.log1p(X[volume_cols])

    # 비율: CLR
    if use_clr:
        if size_ratio_cols:
            X[size_ratio_cols] = _clr_block(X[size_ratio_cols])
        if os_ratio_cols:
            X[os_ratio_cols] = _clr_block(X[os_ratio_cols])
        if category_ratio_cols:
            X[category_ratio_cols] = _clr_block(X[category_ratio_cols])
        if domain_ratio_cols:
            X[domain_ratio_cols] = _clr_block(X[domain_ratio_cols])

    # 열 가중
    if col_weights:
        w = pd.Series({c: col_weights.get(c, 1.0) for c in all_cols}, index=all_cols, dtype=float)
        X = X.mul(w, axis=1)

    # 표준화
    if zscore:
        X = (X - X.mean()) / (X.std() + 1e-9)

    return X, all_cols

# --- 메인 추천 ---
def recommend_with_weighted_similarity(
    ad_df,
    mda_pf,
    top_anchor_by='total_conversions',
    n_anchor=5,
    topN=20,
    weight_power=0.5,
    min_pair_frac=0.0,
    top_weight_feats=None,
    exclude_classes=('계약종료형','품질관리형'),
    min_days_active=7,
    blend_pred_table=None,
    blend_ad_id=None,
    blend_alpha=0.7,
    sort_by="final",

    # 피처 세트(있으면 자동 사용)
    volume_cols=("user_count","total_clicks","total_conversions","daily_avg_conversions","total_ads"),
    size_ratio_cols=("MEGA_ratio","LARGE_ratio","MEDIUM_ratio","SMALL_ratio"),
    os_ratio_cols=("ads_os_type_1_pct","ads_os_type_2_pct","ads_os_type_3_pct","ads_os_type_7_pct"),
    category_ratio_cols=("ads_category_0_pct","ads_category_1_pct","ads_category_2_pct","ads_category_3_pct",
                         "ads_category_4_pct","ads_category_5_pct","ads_category_6_pct","ads_category_7_pct",
                         "ads_category_8_pct","ads_category_10_pct","ads_category_11_pct","ads_category_13_pct"),
    domain_ratio_cols=("domain_게임_pct","domain_교육_pct","domain_금융_pct","domain_기타_pct","domain_미디어/컨텐츠_pct",
                       "domain_뷰티_pct","domain_비영리/공공_pct","domain_생활_pct","domain_식당/카페_pct","domain_식음료_pct",
                       "domain_운동/스포츠_pct","domain_운세_pct","domain_의료/건강_pct","domain_채용_pct","domain_커머스_pct"),

    use_clr=True,
    extra_col_weights=None,

    # 안정화 옵션
    prior_mix=0.2,
    prior_from="mda_mean",   # "mda_mean" | "uniform" | "none"
    prior_bg_dict=None,
    use_idf=False,
    idf_smooth=1.0,
    min_similarity=None
):
    # share 피처
    share_cols = [c for c in mda_pf.columns if c.startswith('share_cat')]
    if not share_cols:
        raise ValueError("mda_pf에 share_cat* 컬럼이 없습니다. 먼저 enrichment를 수행하세요.")

    # 존재하는 컬럼만 사용
    def _keep_exist(cols): return [c for c in cols if c in mda_pf.columns]
    volume_cols         = _keep_exist(volume_cols)
    size_ratio_cols     = _keep_exist(size_ratio_cols)
    os_ratio_cols       = _keep_exist(os_ratio_cols)
    category_ratio_cols = _keep_exist(category_ratio_cols)
    domain_ratio_cols   = _keep_exist(domain_ratio_cols)

    # prior 배경 분포
    prior_bg = None
    if prior_bg_dict is not None:
        prior_bg = dict(prior_bg_dict)
    elif prior_from == "mda_mean":
        avg = mda_pf[share_cols].fillna(0.0).mean(axis=0)
        s = avg.sum()
        if s > 0:
            prior_bg = (avg / s).to_dict()
    elif prior_from == "uniform":
        prior_bg = {c: 1.0/len(share_cols) for c in share_cols}

    # 가중치(광고 분포) 생성
    col_w = make_ad_pair_weights_from_ad_df(
        ad_df, power=weight_power, min_frac=min_pair_frac,
        prior_mix=prior_mix if prior_mix else 0.0,
        prior_bg=prior_bg
    )
    if top_weight_feats:
        top_keys = set(pd.Series(col_w).sort_values(ascending=False).head(top_weight_feats).index)
        col_w = {k: (v if k in top_keys else 0.0) for k,v in col_w.items()}

    # IDF 보정(옵션)
    if use_idf:
        df_share = (mda_pf[share_cols].fillna(0) != 0).sum(axis=0)
        N = len(mda_pf)
        idf = np.log((N + 1.0) / (df_share + idf_smooth))
        idf = idf / (idf.mean() + 1e-12)
        for k in list(col_w.keys()):
            if k in idf.index:
                col_w[k] *= float(idf[k])

    if extra_col_weights:
        col_w.update(extra_col_weights)

    # 피처 행렬
    X, all_feat_cols = build_feature_matrix_plus(
        mda_pf,
        share_cols=share_cols,
        volume_cols=volume_cols,
        size_ratio_cols=size_ratio_cols,
        os_ratio_cols=os_ratio_cols,
        category_ratio_cols=category_ratio_cols,
        domain_ratio_cols=domain_ratio_cols,
        use_clr=use_clr,
        col_weights=col_w,
        zscore=True
    )

    # 앵커/센트로이드
    used = set(ad_df['mda_idx'].astype(int))
    anchors = (ad_df.sort_values(top_anchor_by, ascending=False)
                   .drop_duplicates('mda_idx')
                   .head(n_anchor)['mda_idx']
                   .astype(int).tolist())
    anchors = [m for m in anchors if m in X.index]
    if not anchors:
        raise ValueError("anchor가 없습니다. ad_df에 상위 매체가 있는지 확인하세요.")
    centroid = X.loc[anchors].mean(axis=0).values

    # 후보 & 필터
    cand = mda_pf[~mda_pf['mda_idx'].isin(used)].copy()
    if 'basic_classification' in cand.columns and exclude_classes:
        cand = cand[~cand['basic_classification'].isin(exclude_classes)]
    if 'days_active' in cand.columns:
        cand = cand[cand['days_active'] >= min_days_active]
    if cand.empty:
        return pd.DataFrame(columns=['mda_idx','similarity']), anchors, all_feat_cols, col_w

    # 유사도
    B = X.loc[cand['mda_idx']].values
    cand['similarity'] = cosine_vec(centroid, B)
    if (min_similarity is not None):
        cand = cand[cand['similarity'] >= float(min_similarity)]
        if cand.empty:
            return pd.DataFrame(columns=['mda_idx','similarity']), anchors, all_feat_cols, col_w

    # 예측 블렌딩(옵션)
    has_pred = (blend_pred_table is not None) and (blend_ad_id is not None)
    if has_pred:
        pt = blend_pred_table[blend_pred_table['ads_idx']==blend_ad_id][['mda_idx','pred_turn']].copy()
        cand = cand.merge(pt, on='mda_idx', how='left')
        cand['pred_turn'] = cand['pred_turn'].fillna(0.0)
        maxv = cand['pred_turn'].max()
        cand['pred_norm'] = cand['pred_turn'] / (maxv + 1e-9)
        cand['final_score'] = blend_alpha*cand['similarity'] + (1.0-blend_alpha)*cand['pred_norm']

    # 정렬
    if sort_by == "pred" and has_pred:
        sort_key = "pred_turn"
    elif sort_by == "sim":
        sort_key = "similarity"
    else:
        sort_key = "final_score" if has_pred else "similarity"

    keep = [c for c in [
        'mda_idx','similarity','final_score','pred_turn','pred_norm',
        'basic_classification','days_active','conversion_rate',
        'expected_total_profit','total_ads'
    ] if c in cand.columns]
    out = cand[keep].sort_values(sort_key, ascending=False).head(topN).reset_index(drop=True)
    return out, anchors, all_feat_cols, col_w
# === /통합 셀 끝 ===
