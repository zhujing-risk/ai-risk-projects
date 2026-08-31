# ============================================================
# 第2部分：IV值计算（变量初筛）
# ============================================================
# 目的：从几百个变量中，快速筛掉"没用的"变量
# IV (Information Value) = 一个变量对"好客户vs坏客户"的区分能力
#
# 通俗理解：
#   你有300个变量（年龄、收入、征信查询次数...）
#   不是每个都有用，有些变量对预测违约毫无帮助
#   IV就是用来衡量"这个变量到底有没有用"的数字
#
# IV判断标准：
#   IV < 0.02  → 没用，删掉
#   0.02~0.1   → 弱，勉强能用
#   0.1~0.3    → 不错，核心变量
#   0.3~0.5    → 很强，但要警惕是否有信息泄漏
#   > 0.5      → 太强了，可能有问题（比如用"是否被催收"预测"是否逾期"）
# ============================================================

import numpy as np
import pandas as pd
from joblib import Parallel, delayed  # 并行计算库，加速处理
from tqdm import tqdm  # 进度条库，显示处理进度


# ---- 步骤1：预处理，找出哪些列值得计算IV ----

def preprocess_features(df, target_col):
    """
    过滤掉明显没用的列

    为什么要先过滤？
    → 300个变量逐个算IV很慢，先排除掉"一眼就知道没用"的列

    哪些列没用？
    1. 缺失率>80%的列 → 大部分值都是空的，没有信息量
    2. 只有1个唯一值的列 → 比如某列全是0，无法区分好坏
    3. 目标变量本身 → 不能用答案来预测答案
    """
    valid_cols = []
    drop_cols = []
    dtypes = {}

    for col in df.columns:
        # 跳过目标变量（"是否违约"这一列）
        if col == target_col:
            drop_cols.append(col)
            continue

        # 缺失率>80%的列直接删
        if df[col].isnull().mean() > 0.8:
            drop_cols.append(col)
            continue

        # 只有1个值的列直接删（没有任何区分力）
        if df[col].nunique() <= 1:
            drop_cols.append(col)
            continue

        # 通过筛选的列，记录下来
        valid_cols.append(col)

        # 自动判断是数值型还是分类型
        if pd.api.types.is_numeric_dtype(df[col]):
            dtypes[col] = "numeric"
        else:
            dtypes[col] = "categorical"

    return valid_cols, dtypes, drop_cols


# ---- 步骤2：计算单个变量的IV ----

def calculate_iv_single(col_name, data):
    """
    计算一个变量的IV值

    核心公式：
        IV = Σ (好客户占比 - 坏客户占比) × ln(好客户占比 / 坏客户占比)

    步骤：
        1. 把变量分成若干箱（比如收入分10个档位）
        2. 算每箱里好客户和坏客户的比例
        3. 每箱的WOE = ln(好客户占比 / 坏客户占比)
        4. 每箱的IV贡献 = (好占比 - 坏占比) × WOE
        5. 总IV = 所有箱加起来
    """
    try:
        if len(data) < 50:
            return None

        feature_col = data.columns[0]  # 第一列是特征
        target_col = data.columns[1]   # 第二列是目标变量（0/1）

        feature_series = data[feature_col]
        target_series = data[target_col]

        # 数值型变量：等频分箱（每箱样本量差不多）
        if pd.api.types.is_numeric_dtype(feature_series):
            n_bins = min(10, len(data) // 50)  # 最多10箱，每箱至少50个样本
            try:
                bins = pd.qcut(feature_series, q=n_bins, duplicates='drop', labels=False)
            except:
                bins = pd.cut(feature_series, bins=n_bins, labels=False, duplicates='drop')
            dtype = 'numeric'
        else:
            # 分类变量：直接用类别编码
            bins = feature_series.astype('category').cat.codes
            dtype = 'categorical'

        # 计算IV
        iv_value = _fast_iv_calculation(target_series, bins)

        return {'variable': col_name, 'iv': iv_value, 'dtype': dtype}

    except Exception as e:
        print(f"计算列 {col_name} 时出错: {e}")
        return None


def _fast_iv_calculation(target, bins):
    """
    IV的核心计算逻辑

    为什么加了0.5？（平滑处理）
    → 如果某箱里坏客户数=0，ln(0)会报错
    → 加一个很小的数避免除零错误，这叫Laplace平滑
    """
    try:
        df_temp = pd.DataFrame({'target': target, 'bins': bins}).dropna()

        if len(df_temp) < 2:
            return 0

        # 按箱分组统计
        grouped = df_temp.groupby('bins')['target'].agg(['count', 'sum'])
        grouped['non_events'] = grouped['count'] - grouped['sum']  # 好客户数

        total_events = grouped['sum'].sum()       # 总坏客户数
        total_non_events = grouped['non_events'].sum()  # 总好客户数

        if total_events == 0 or total_non_events == 0:
            return 0

        # 加平滑，防止除零
        grouped['event_pct'] = (grouped['sum'] + 0.5) / (total_events + 1)
        grouped['non_event_pct'] = (grouped['non_events'] + 0.5) / (total_non_events + 1)

        # WOE = ln(好客户占比 / 坏客户占比)
        grouped['woe'] = np.log(grouped['event_pct'] / grouped['non_event_pct'])

        # IV = Σ (好占比 - 坏占比) × WOE
        iv = ((grouped['event_pct'] - grouped['non_event_pct']) * grouped['woe']).sum()

        return iv

    except:
        return 0


# ---- 步骤3：并行计算所有变量的IV ----

def compute_iv_all(df, target_col='dob4_ever10_flg'):
    """
    主函数：批量计算所有变量的IV

    为什么用并行（Parallel）？
    → 300个变量串行计算要10分钟，并行只要1分钟
    → n_jobs=-1 表示用所有CPU核心
    """
    # 先做预处理，过滤掉没用的列
    cols = [col for col in df.columns if col != target_col]
    missing_rates = df[cols].isnull().mean()
    nunique_vals = df[cols].nunique()

    valid_cols = [col for col in cols
                  if missing_rates[col] <= 0.8 and nunique_vals[col] > 1]

    # 并行计算每个变量的IV
    results = Parallel(n_jobs=-1)(
        delayed(calculate_iv_single)(col, df[[col, target_col]].dropna())
        for col in tqdm(valid_cols, desc="IV 并行计算中...")
    )

    # 整理结果
    iv_df = pd.DataFrame([r for r in results if r is not None])
    if not iv_df.empty:
        iv_df = iv_df.sort_values('iv', ascending=False)

        # 加一列中文说明
        iv_df['strength'] = np.select(
            [iv_df['iv'] <= 0.02, iv_df['iv'] <= 0.1,
             iv_df['iv'] <= 0.3, iv_df['iv'] > 0.3],
            ['无预测力', '弱', '中等', '强'],
            default='无预测力'
        )

    return iv_df


# ============================================================
# 实际使用
# ============================================================

# 定义要排除的列（这些是目标变量相关的列，不能参与计算）
to_drop = [
    'is_perform_fpb1', 'is_perform_fpb5', 'is_perform_fpb10',
    'fpd1_flag', 'fpd5_flag', 'fpd10_flag', 'fpd31_flag',
    'dob2_ever10_flg', 'dob3_ever10_flg', 'dob4_ever10_flg',
    # ... 所有表现期相关的列都要排除，否则就是"用答案预测答案"
]

# 计算IV
raw_data = df[df['is_perform_dob4_ever10'] > 0]  # 只看有表现期的样本
iv_result = compute_iv_all(raw_data.drop(to_drop, axis=1), 'dob4_ever10_flg')

# 筛选IV > 0.005的变量（非常宽松的阈值，先保留多一些）
iv_vars = list(iv_result[iv_result.iv > 0.005].variable)
print(f"IV筛选后剩余变量数: {len(iv_vars)}")
