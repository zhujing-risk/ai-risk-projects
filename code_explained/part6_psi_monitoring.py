# ============================================================
# 第6部分：PSI稳定性监控
# ============================================================
# 目的：监控变量分布是否随时间"漂移"了
#
# 什么是分布漂移？
#   举例：你用2024年1-6月的数据训练了模型
#   如果到了2025年，客户的收入分布完全变了（比如经济下行导致整体收入降低）
#   那模型的预测就不准了——因为它学到的"规律"已经不适用了
#
# PSI = Population Stability Index（人群稳定性指数）
#   对比"基准期"和"观察期"的分布差异
#   PSI越大 = 差异越大 = 模型越可能失效
#
# 判断标准：
#   PSI < 0.1  → 稳定，不用管
#   0.1~0.25   → 轻微漂移，需要关注
#   PSI > 0.25 → 严重漂移，必须重训模型！
# ============================================================

import numpy as np
import pandas as pd
from typing import List, Union


def calculate_feature_psi(
    df: pd.DataFrame,
    month_col: str,
    feature_list: List[str],
    base_month: Union[str, None] = None,
    bins: int = 10,
    verbose: bool = True
) -> pd.DataFrame:
    """
    计算一组特征在不同月份相对于基准月的PSI

    参数：
        df: 包含数据的DataFrame
        month_col: 月份列的列名（比如 'apply_month'）
        feature_list: 要计算PSI的特征列表
        base_month: 基准月（如果不指定，用数据中最早的月份）
        bins: 分箱数（默认10）

    返回：
        DataFrame，每行 = (月份, 变量名, PSI值)
    """

    def is_numeric_series(series):
        """判断是否为数值型数据"""
        return pd.api.types.is_numeric_dtype(series)

    def calculate_single_psi(expected, actual, bins):
        """
        计算单个特征的PSI

        公式：PSI = Σ (actual% - expected%) × ln(actual% / expected%)

        通俗理解：
            把特征分成10个箱
            比较"基准月"和"当前月"在每个箱里的人数占比
            如果每个箱的占比都差不多 → PSI小 → 分布没变
            如果某些箱人数大增/大减 → PSI大 → 分布变了！

        为什么加1e-6？
            防止某箱人数=0时log(0)报错
        """
        expected = expected.dropna()
        actual = actual.dropna()

        if len(expected) == 0 or len(actual) == 0:
            return np.nan

        # 用基准月的分位数作为分箱边界
        # 这样"箱"的定义是固定的，才能公平比较
        breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
        breakpoints[-1] += 1e-6  # 确保最大值不被遗漏

        # 统计每箱的样本占比
        expected_hist = np.histogram(expected, bins=breakpoints)[0] + 1e-6
        actual_hist = np.histogram(actual, bins=breakpoints)[0] + 1e-6

        expected_pct = expected_hist / expected_hist.sum()
        actual_pct = actual_hist / actual_hist.sum()

        # PSI公式
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return psi

    # ---- 主逻辑 ----

    # 确定基准月
    if base_month is None:
        base_month = df[month_col].min()
        if verbose:
            print(f"自动使用最早月份作为基准: {base_month}")

    base_data = df[df[month_col] == base_month]

    skipped_vars = set()
    psi_results = []

    # 遍历每个月份 × 每个特征
    for month in df[month_col].unique():
        if month == base_month:
            continue  # 不和自己比

        current_data = df[df[month_col] == month]

        for var in feature_list:
            # 非数值型变量跳过（PSI只对数值有意义）
            if not is_numeric_series(base_data[var]):
                skipped_vars.add(var)
                continue

            psi = calculate_single_psi(
                expected=base_data[var],
                actual=current_data[var],
                bins=bins
            )
            psi_results.append({
                'month': month,
                'variable': var,
                'psi': psi,
                'base_month': base_month
            })

    if verbose and skipped_vars:
        print(f"已跳过 {len(skipped_vars)} 个非数值型变量")

    return pd.DataFrame(psi_results)


# ============================================================
# 实际使用
# ============================================================

# 计算PSI
psi_df = calculate_feature_psi(
    df=df[(df['is_perform_dob2_ever30'] > 0) & (df['trans_month'] >= '202501')].copy(),
    month_col='trans_month',
    feature_list=pboc_feature,  # 要监控的特征列表
    bins=5
)

# 筛选稳定变量：取每个变量在所有月份中PSI的最大值 < 0.1
stable_vars = psi_df.groupby('variable')['psi'].max().reset_index()
stable_vars = stable_vars[stable_vars['psi'] < 0.1]['variable'].tolist()

print(f"稳定变量数量: {len(stable_vars)}")
# 只有稳定的变量才能放进最终模型，不稳定的变量上线后会出问题


# ============================================================
# 补充：Kendall Tau趋势稳定性检验
# ============================================================
# 除了PSI看分布漂移，还要看"排序是否稳定"
# 比如：信用分分5箱，每月的坏账率排序应该都是"低分>高分"
# 如果某月突然反转了，说明这个变量的"区分逻辑"变了

from scipy.stats import kendalltau


def check_trend_stability(pivot_table, threshold=0.4):
    """
    检查分箱坏账率的排序是否跨月稳定

    做法：
        1. 每月按分箱排序坏账率
        2. 计算月份间排序的Kendall Tau相关
        3. 平均Tau > 0.4 → 稳定

    Tau值含义：
        τ ≥ 0.7  → 强稳定（排序高度一致）
        0.4~0.7  → 中等稳定
        τ < 0.4  → 不稳定（排序会翻转，变量不可靠）
    """
    monthly_rankings = {}

    for month in pivot_table.columns:
        monthly_rankings[month] = pivot_table[month].rank(method='dense').values

    # 两两月份比较Kendall Tau
    similarity_scores = []
    months = list(monthly_rankings.keys())

    for i in range(len(months)):
        for j in range(i + 1, len(months)):
            tau, _ = kendalltau(monthly_rankings[months[i]], monthly_rankings[months[j]])
            similarity_scores.append(tau)

    avg_similarity = np.mean(similarity_scores)
    is_stable = avg_similarity >= threshold

    return {
        'avg_similarity': avg_similarity,
        'is_stable': is_stable
    }
