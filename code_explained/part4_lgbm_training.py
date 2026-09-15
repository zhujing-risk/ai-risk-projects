# ============================================================
# 第4部分：LightGBM建模（特征精选）
# ============================================================
# 目的：用LightGBM树模型从剩余变量中找出"真正重要的TOP变量"
#
# 为什么不直接用LR？
#   LR只能发现线性关系（X增大→Y增大）
#   但很多变量的作用是非线性的（比如"年龄"对违约是U型的）
#   LGBM能自动发现这些复杂关系，找出真正有区分力的变量
#
# 为什么不直接用LGBM做最终模型？
#   因为监管要求评分卡"可解释" → LR可以，LGBM不行
#   所以：LGBM负责"选变量"，LR负责"建评分卡"
# ============================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# ============================================================
# 步骤1：准备训练数据
# ============================================================

# 选择要用的特征（排除目标变量和不该参与的列）
to_drop = ['channel_type_final', 'product_code', 'loan_apply_no',
           'apply_month', 'credit_apply_date', 'credit_amount',
           # ... 以及所有表现期相关的变量
           ]

# X = 特征矩阵（用来预测的输入）
train_data_x = dftmp[list(set(scoreVar) - set(to_drop))].copy()

# y = 目标变量（0=好客户, 1=坏客户）
train_data_y = dftmp['dob4_ever10_flg']

# 处理数据类型：确保所有列都是数值型
var_char = train_data_x.dtypes[train_data_x.dtypes == 'object'].index.tolist()
for col in var_char:
    try:
        train_data_x[col] = pd.to_numeric(train_data_x[col])
    except ValueError:
        train_data_x[col] = train_data_x[col].astype('category')


# ============================================================
# 步骤2：划分训练集和验证集
# ============================================================

# 为什么要分？
#   训练集：用来"学习"模型参数
#   验证集：用来"考试"，检验模型学得好不好
#   如果不分开，模型会"背题"（过拟合），看着表现好但遇到新数据就不行了
#
# stratify=train_data_y 的作用：
#   保证训练集和验证集中坏客户的比例一样（比如都是5%）
#   避免验证集中坏客户太少导致评估不准

X_train, X_val, y_train, y_val = train_test_split(
    train_data_x,
    train_data_y,
    test_size=0.3,          # 30%做验证
    random_state=42,        # 固定随机种子，结果可复现
    stratify=train_data_y   # 分层抽样
)

# 创建LightGBM专用的数据格式
# free_raw_data=False → 保留原始数据不释放（后面还要用来提取规则）
train_set = lgb.Dataset(X_train, y_train, free_raw_data=False)
val_set = lgb.Dataset(X_val, y_val, reference=train_set, free_raw_data=False)


# ============================================================
# 步骤3：设置参数
# ============================================================

base_params = {
    # ---- 基础设置 ----
    'boosting_type': 'gbdt',     # 梯度提升决策树（最稳定的方式）
    'objective': 'binary',       # 二分类任务（好/坏）
    'metric': ['auc', 'binary_error'],  # 评估指标

    # ---- 树结构参数（控制模型复杂度） ----
    'num_leaves': 31,            # 每棵树最多31个叶子（越大越复杂）
    'max_depth': 5,              # 树最深5层（防止过度细分）
    'min_data_in_leaf': 1000,    # 每个叶子至少1000个样本（防止极端小叶）
    # ↑ 这个参数很重要！如果设太小，模型会记住个别噪声点

    # ---- 学习参数 ----
    'learning_rate': 0.1,        # 学习率（每棵树的贡献权重）
    # ↑ 越小越保守：需要更多棵树，但不容易过拟合

    # ---- 正则化参数（防止过拟合） ----
    'feature_fraction': 0.8,     # 每棵树只随机用80%的特征
    'bagging_fraction': 0.8,     # 每棵树只随机用80%的样本
    'lambda_l1': 0.1,            # L1正则化（让不重要的特征权重→0）
    'lambda_l2': 0.1,            # L2正则化（让所有权重都小一些）

    # ---- 类别不平衡处理 ----
    'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
    # ↑ 好客户数/坏客户数。比如20:1，就让坏客户的权重×20
    # 为什么？因为坏客户很少（只有5%），不加权模型会"忽略"他们

    'verbose': -1  # 不输出训练过程的警告信息
}


# ============================================================
# 步骤4：两阶段训练
# ============================================================

# 为什么分两阶段？
#   第一阶段：learning_rate=0.1，快速找到大方向
#   第二阶段：learning_rate=0.02 + DART，精细调整
#   类比：先用大步走近目标，再用小步精确定位

# ---- 第一阶段：快速学习 ----
print("=== 第一阶段训练 ===")
stage1_params = base_params.copy()
stage1_params.update({
    'learning_rate': 0.1,
    'early_stopping_rounds': 50
})

lgbmodel = lgb.train(
    stage1_params,
    train_set,
    num_boost_round=200,          # 最多200棵树
    valid_sets=[train_set, val_set],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.log_evaluation(10),   # 每10棵树打印一次AUC
        lgb.early_stopping(50)    # 如果验证集AUC连续50轮不提升就停下
        # ↑ early_stopping是防过拟合的核心机制！
    ]
)

# ---- 第二阶段：精细调整 ----
print("\n=== 第二阶段训练 ===")
stage2_params = base_params.copy()
stage2_params.update({
    'learning_rate': 0.02,        # 更小的学习率
    'boosting_type': 'dart'       # DART = 随机丢弃之前的树，增加多样性
    # DART的好处：减少过拟合，增强泛化能力
})

lgbmodel = lgb.train(
    stage2_params,
    train_set,
    num_boost_round=300,
    valid_sets=[train_set, val_set],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.log_evaluation(10),
        lgb.early_stopping(50)
    ]
)


# ============================================================
# 步骤5：评估模型 + 提取特征重要性
# ============================================================

# 在验证集上预测
val_pred = lgbmodel.predict(X_val)
print(f"验证集AUC: {roc_auc_score(y_val, val_pred):.4f}")

# 特征重要性排序
# LGBM的feature_importance = 该特征被用来分裂的次数（或增益贡献）
importance_df = pd.DataFrame({
    'feature': lgbmodel.feature_name(),
    'importance': lgbmodel.feature_importance()
}).sort_values(by='importance', ascending=False).reset_index(drop=True)

print("\nTop 10 重要特征：")
print(importance_df.head(10))

# 通常取Top 30-50个特征给下游的逻辑回归使用
top_features = importance_df.head(30)['feature'].tolist()
