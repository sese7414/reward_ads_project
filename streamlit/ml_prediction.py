import pandas as pd
import numpy as np
import streamlit as st
import joblib
from typing import Dict, Optional, Set
from sklearn.metrics import roc_auc_score

# -------------------------------
# 8) 평가
# -------------------------------
def _precision_recall_hit_at_k(preds_dict, gt_dict, k=10):
    P,R,H=[],[],[]
    for ad, df in preds_dict.items():
        if ad not in gt_dict: continue
        pred = df["mda_idx"].astype(int).tolist()[:k]
        true = set(gt_dict[ad]); if_not = (not true)
        if if_not: continue
        inter = len(set(pred)&true)
        P.append(inter/max(1,len(pred))); R.append(inter/len(true)); H.append(1.0 if inter>0 else 0.0)
    return (float(np.mean(P)) if P else 0.0,
            float(np.mean(R)) if R else 0.0,
            float(np.mean(H)) if H else 0.0)

def _map_at_k(preds_dict, gt_dict, k=10):
    APs=[]
    for ad, df in preds_dict.items():
        if ad not in gt_dict: continue
        true=set(gt_dict[ad]); 
        if not true: continue
        ranked = df["mda_idx"].astype(int).tolist()[:k]
        hits=cum_prec=0
        for i,m in enumerate(ranked, start=1):
            if m in true:
                hits+=1; cum_prec += hits/i
        APs.append(cum_prec/min(len(true),k) if hits>0 else 0.0)
    return float(np.mean(APs)) if APs else 0.0

def _auc_flat(preds_dict, gt_pairs_dict, score_col="score"):
    y_true, y_score = [], []
    for ad, df in preds_dict.items():
        true_mdas = set(gt_pairs_dict.get(ad, []))
        for _, r in df.iterrows():
            y_true.append(1 if int(r["mda_idx"]) in true_mdas else 0)
            y_score.append(float(r[score_col]))
    if len(set(y_true))<2: return 0.5
    try: return float(roc_auc_score(y_true, y_score))
    except Exception: return 0.5

def evaluate_all(preds_with_counts: Dict[int, pd.DataFrame], tgt: pd.DataFrame, k=10):
    gt_pairs = tgt.groupby("ads_idx")["mda_idx"].apply(lambda s: list(pd.Series(s).dropna().astype(int).unique())).to_dict()
    prec, rec, hit = _precision_recall_hit_at_k(preds_with_counts, gt_pairs, k=k)
    mapk = _map_at_k(preds_with_counts, gt_pairs, k=k)
    auc_rank = _auc_flat(preds_with_counts, gt_pairs, score_col="score")
    auc_reg  = _auc_flat(preds_with_counts, gt_pairs, score_col="yhat_turn")
    return {f"precision@{k}":prec, f"recall@{k}":rec, f"map@{k}":mapk,
            f"hit_rate@{k}":hit, "auc_ranker":auc_rank, "auc_regressor":auc_reg}

def predict_evaluate_all(preds_with_counts, tgt_large):
    metrics_k5  = evaluate_all(preds_with_counts, tgt_large, k=5)
    metrics_k10 = evaluate_all(preds_with_counts, tgt_large, k=10)

    # key 이름에서 @숫자 부분 제거
    def normalize_keys(metrics: dict):
        clean = {}
        for k, v in metrics.items():
            if "precision@" in k: clean["precision"] = v
            elif "recall@" in k: clean["recall"] = v
            elif "map@" in k: clean["map"] = v
            elif "hit_rate@" in k: clean["hit_rate"] = v
            else: clean[k] = v
        return clean

    metrics_k5_clean = normalize_keys(metrics_k5)
    metrics_k10_clean = normalize_keys(metrics_k10)

    eval_df = pd.DataFrame([metrics_k5_clean, metrics_k10_clean], index=["@5", "@10"])
    return eval_df


def get_expected_conversions_for_ad(preds_with_counts: Dict[int, pd.DataFrame],
                                    ad_id: int, sort_by="yhat_turn", top=20) -> pd.DataFrame:
    df = preds_with_counts.get(int(ad_id))
    if df is None or df.empty:
        return pd.DataFrame(columns=["mda_idx","score","yhat_turn"])
    sort_by = "yhat_turn" if sort_by not in ("yhat_turn","score") else sort_by
    out = df.sort_values(sort_by, ascending=False).copy()
    return out.head(top) if top is not None else out


def recommend_top_media(ad_id, preds_with_counts, top_n):
    # (G) 예: 특정 광고 top-5 (예상 전환수 기준)
    top_recommend = get_expected_conversions_for_ad(preds_with_counts, ad_id, sort_by="yhat_turn", top=top_n)
    return top_recommend [["mda_idx","yhat_turn","score"]]


# -------------------------------
# 9-추가) 신규 매체사만 필터링 헬퍼
# -------------------------------
def _build_excluded_pairs_for_ad(
    ad_id: int,
    ads_time: pd.DataFrame,
    running_pairs: Optional[pd.DataFrame] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    exclude_click_ge1: bool = True,
    exclude_any_history: bool = True
) -> Set[int]:
    """
    특정 광고(ad_id)에 대해 '제외해야 할 mda_idx' 집합을 만든다.
      - 과거 로그에 등장한 모든 페어(ads_idx=ad_id)  [exclude_any_history]
      - 과거 클릭수가 1 이상인 페어                   [exclude_click_ge1]
      - 현재 진행중 페어(running_pairs)
    """
    t = ads_time.copy()
    # 날짜 필터
    if date_from or date_to:
        t["date"] = pd.to_datetime(t["rpt_time_date"], errors="coerce")
        if date_from: t = t[t["date"] >= pd.to_datetime(date_from)]
        if date_to:   t = t[t["date"] <= pd.to_datetime(date_to)]
    # ad 고정
    t = t[pd.to_numeric(t["ads_idx"], errors="coerce").fillna(-1).astype(int) == int(ad_id)].copy()
    t["mda_idx"] = pd.to_numeric(t["mda_idx"], errors="coerce").fillna(-1).astype(int)

    excluded = set()
    if exclude_any_history and not t.empty:
        excluded.update(t["mda_idx"].dropna().astype(int).tolist())
    if exclude_click_ge1 and not t.empty:
        clk = pd.to_numeric(t.get("rpt_time_clk", 0), errors="coerce").fillna(0)
        excluded.update(t.loc[clk >= 1, "mda_idx"].dropna().astype(int).tolist())

    if running_pairs is not None and not running_pairs.empty:
        rp = running_pairs.copy()
        rp["ads_idx"] = pd.to_numeric(rp["ads_idx"], errors="coerce").fillna(-1).astype(int)
        rp["mda_idx"] = pd.to_numeric(rp["mda_idx"], errors="coerce").fillna(-1).astype(int)
        excluded.update(rp.loc[rp["ads_idx"] == int(ad_id), "mda_idx"].tolist())

    return {int(m) for m in excluded if m != -1}

def get_new_media_for_ad(
    preds_with_counts: Dict[int, pd.DataFrame],
    ad_id: int,
    ads_time: pd.DataFrame,
    running_pairs: Optional[pd.DataFrame] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    sort_by: str = "yhat_turn",
    top: int = 20,
    media_meta: Optional[pd.DataFrame] = None  # 예: media_portfolio[["mda_idx","mda_name"]]
) -> pd.DataFrame:
    """
    예측 결과에서 해당 광고의 '신규 매체사'만 반환.
      - 신규 정의: 이력 無, 클릭≥1 無, 현재 진행중 아님
    """
    # 1) 예측 상위 rows
    cand = get_expected_conversions_for_ad(preds_with_counts, ad_id, sort_by=sort_by, top=None)
    if cand.empty:
        return cand

    # 2) 제외할 mda 집합 구성
    excluded_mdas = _build_excluded_pairs_for_ad(
        ad_id=ad_id,
        ads_time=ads_time,
        running_pairs=running_pairs,
        date_from=date_from,
        date_to=date_to,
        exclude_click_ge1=True,
        exclude_any_history=True
    )

    # 3) 신규만 필터
    out = cand[~cand["mda_idx"].astype(int).isin(excluded_mdas)].copy()

    # 4) 이름 붙이기(옵션)
    if media_meta is not None:
        cols = [c for c in ["mda_idx","mda_name"] if c in media_meta.columns]
        if "mda_idx" in cols and len(cols) >= 1:
            out = out.merge(media_meta[cols].drop_duplicates("mda_idx"), on="mda_idx", how="left")

    # 5) top N
    if top is not None:
        out = out.head(int(top))
    return out
