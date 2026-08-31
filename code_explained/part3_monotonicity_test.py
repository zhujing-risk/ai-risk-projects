# ============================================================
# 第3部分：单调性检验
# ============================================================
# 目的：检查变量是否满足"单调性" —— 评分卡的核心要求
#
# 什么是单调性？
#   举例：信用分越高 → 违约率越低，这就是"单调递减"
#   如果信用分500违约率5%，600违约率8%，700违约率3% → 不单调！
#   不单调的变量不能直接放进评分卡（会导致"分数高反而更危险"的荒谬结果）
#
# 检验方法：
#   1. Cochran-Armitage趋势检验 → 统计学方法，看"趋势是否显著"
#   2. Spearman相关系数 → 看排序是否一致
#   3. 违反比例（violation_ratio）→ 有多少箱是"反方向"的
# ============================================================

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal, chi2_contingency
from joblib import Parallel, delayed
from tqdm import tqdm


# ---- 检验方法1：Cochran-Armitage趋势检验 ----

def cochran_armitage_test(bad_rates, bin_counts):
    """
    Cochran-Armitage趋势检验

    通俗理解：
        问：随着分箱从低到高，坏客户率的变化趋势是否"统计上显著"？
        答：p-value < 0.05 → 趋势显著（单调性成立）
            p-value >= 0.05 → 趋势不显著（可能是随机波动）

    参数：
        bad_rates: 每箱的坏账率（array）
        bin_counts: 每箱的样本数（array）

    返回：
        p-value（越小越好，<0.05表示趋势显著）
    """
    try:
        # 构建列联表：行=分箱，列=[好客户数, 坏客户数]
        contingency_table = np.array([
            bin_counts * (1 - bad_rates),  # 每箱好客户数
            bin_counts * bad_rates         # 每箱坏客户数
        ]).T.astype(int)

        # 每箱至少5个样本才有统计意义
        if np.any(contingency_table < 5):
            return np.nan

        # 加权卡方检验（权重=线性序号，检测线性趋势）
        n_rows = contingency_table.shape[0]
        weights = np.arange(1, n_rows + 1)  # 线性权重 [1, 2, 3, ...]
        weighted_table = contingency_table * weights[:, None]

        chi2, pval, dof, expected = chi2_contingency(weighted_table)
        return pval

    except Exception as e:
        print(e)
        return np.nan


# ---- 辅助函数：自动判断趋势方向 ----

def detect_trend_direction(bad_rates):
    """
    自动判断坏账率是递增还是递减

    做法：计算相邻箱的差值，看"升"多还是"降"多
        差值 > 0 的次数多 → 递增趋势
        差值 < 0 的次数多 → 递减趋势
    """
    if len(bad_rates) < 2:
        return 'none'

    differences = bad_rates.diff().dropna()
    sign_sum = (differences > 0).sum() - (differences < 0).sum()

    if sign_sum > 0:
        return 'increase'   # 坏账率随分箱增大而升高
    elif sign_sum < 0:
        return 'decrease'   # 坏账率随分箱增大而降低
    else:
        return 'stable'


# ---- 核心函数：分析单个变量的单调性 ----

def analyze_variable(df, x_col, y_col, n_bins=10, trend='auto', min_bin_size=0.02):
    """
    对单个变量进行完整的单调性分析

    步骤：
        1. 等频分箱（每箱人数差不多）
        2. 计算每箱坏账率
        3. 判断趋势方向（递增/递减）
        4. 计算Spearman相关（排序一致性）
        5. 计算违反比例（有多少箱"方向反了"）
        6. 做Cochran-Armitage检验（趋势是否显著）

    参数：
        df: 数据
        x_col: 要检验的特征列名
        y_col: 目标变量列名（0/1）
        n_bins: 分多少箱（默认10箱）
        min_bin_size: 每箱最少占总样本的比例（防止极端小箱）

    返回：
        字典，包含各项单调性指标
    """
    try:
        # 第1步：等频分箱
        df['bin'], bins = pd.qcut(df[x_col], q=n_bins, duplicates='drop', retbins=True)
        bad_rates = df.groupby('bin')[y_col].mean().sort_index()
        bin_counts = df.groupby('bin').size()

        # 过滤掉样本太少的箱（占比<2%的箱不可靠）
        valid_mask = (bin_counts / len(df)) >= min_bin_size
        bad_rates = bad_rates[valid_mask]
        bin_counts = bin_counts[valid_mask]

        if len(bad_rates) < 2:
            return None

        # 第2步：判断趋势方向
        actual_trend = detect_trend_direction(bad_rates) if trend == 'auto' else trend
        if actual_trend == 'none':
            return None

        # 第3步：计算Spearman相关系数
        # Spearman衡量"排序是否一致"
        # |rho|接近1 = 强单调，接近0 = 不单调
        bin_ranks = np.arange(len(bad_rates))
        rho, _ = spearmanr(bin_ranks, bad_rates)

        # 第4步：计算违反比例
        # 如果趋势应该是"递增"，那么每对相邻箱都应该是"后一箱 > 前一箱"
        # 违反 = "后一箱 < 前一箱"的次数 / 总比较次数
        if actual_trend == 'increase':
            violations = sum(np.diff(bad_rates) < 0)  # 应该升但降了
        else:
            violations = sum(np.diff(bad_rates) > 0)  # 应该降但升了

        return {
            'variable': x_col,
            'n_bins': len(bad_rates),
            'bad_rate_min': bad_rates.min(),
            'bad_rate_max': bad_rates.max(),
            'trend_direction': actual_trend,
            'trend_strength': abs(rho),           # |Spearman| → 越大越单调
            'violation_ratio': violations / (len(bad_rates) - 1),  # 越小越好，0=完美单调
            'cochran_armitage_p': cochran_armitage_test(bad_rates, bin_counts),
            'bins': bins.tolist()
        }

    except Exception as e:
        print(f"Error in {x_col}: {str(e)}")
        return None


# ---- 批量分析所有变量 ----

def batch_analyze(df, x_cols, y_col, nbins, trend='auto', min_bin_size=0.02, n_jobs=4):
    """
    并行批量分析多个变量的单调性

    n_jobs=4 表示同时用4个CPU核心并行计算
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(analyze_variable)(df, col, y_col, n_bins=nbins, trend=trend, min_bin_size=min_bin_size)
        for col in tqdm(x_cols, desc="单调性检验中...")
    )
    return pd.DataFrame([r for r in results if r is not None])


# ============================================================
# 实际使用
# ============================================================

# 对IV筛选后的变量做单调性检验
monotonicity_result = batch_analyze(
    df=raw_data,
    x_cols=iv_vars,           # IV筛选后的变量列表
    y_col='dob4_ever10_flg',  # 目标：DOB4（放款后第4个月）是否逾期10天以上
    nbins=10,
    n_jobs=4
)

# 筛选标准：
# 1. violation_ratio < 0.2 → 允许最多20%的箱违反趋势
# 2. cochran_armitage_p < 0.05 → 趋势统计显著
# 3. trend_strength > 0.5 → Spearman相关足够强
good_monotonic_vars = monotonicity_result[
    (monotonicity_result['violation_ratio'] < 0.2) &
    (monotonicity_result['cochran_armitage_p'] < 0.05) &
    (monotonicity_result['trend_strength'] > 0.5)
]['variable'].tolist()

print(f"通过单调性检验的变量数: {len(good_monotonic_vars)}")
