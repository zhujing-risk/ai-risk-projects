# 风控建模代码拆解（新手友好版）

## 这套代码做了什么？

从几百个变量中，筛选出最有用的变量，训练模型，并从模型中提取可落地的策略规则。

## 完整流程图

```
原始数据（300+变量）
    │
    ▼
[Part 1] 数据加载（从ODPS读取）
    │
    ▼
[Part 2] IV筛选（删掉没用的变量，300→100）
    │
    ▼
[Part 3] 单调性检验（确保变量逻辑合理，100→60）
    │
    ▼
[Part 4] LGBM建模（精选核心变量，60→30）
    │
    ▼
[Part 5] 规则提取（从树模型中挖掘策略规则）
    │
    ▼
[Part 6] PSI监控（上线后持续监控变量稳定性）
```

## 文件列表

| 文件 | 内容 | 核心知识点 |
|------|------|-----------|
| part1_data_loading.py | ODPS数据读取 | 多进程加载、宽表 |
| part2_iv_calculation.py | IV值计算 | WOE/IV原理、分箱、Laplace平滑 |
| part3_monotonicity_test.py | 单调性检验 | Cochran-Armitage、Spearman、违反比例 |
| part4_lgbm_training.py | LightGBM训练 | GBDT原理、参数调优、两阶段训练、early_stopping |
| part5_rule_extraction.py | 规则提取与评估 | 决策路径提取、覆盖率、提升度、业务逻辑过滤 |
| part6_psi_monitoring.py | PSI稳定性监控 | 分布漂移、Kendall Tau、模型重训触发 |

## 面试速记卡片

见同目录下 `interview_flashcards.md`

## 学习建议

1. 先读 Part 2（IV）和 Part 5（规则提取），这两块面试问得最多
2. Part 4（LGBM）重点理解参数含义，不需要手写
3. Part 6（PSI）理解"为什么需要监控"比会写代码更重要
