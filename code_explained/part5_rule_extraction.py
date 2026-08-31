# ============================================================
# 第5部分：规则提取（从LGBM树中挖掘策略规则）
# ============================================================
# 目的：把LGBM学到的"判断逻辑"翻译成人能读的策略规则
#
# 通俗理解：
#   LGBM训练出了200棵决策树，每棵树都是一串IF-THEN判断
#   比如：IF 信用分<=450 AND 近3月查询>5次 THEN 这人大概率违约
#   我们要从这几万条路径中，找出"又准又覆盖面广"的策略规则
#
# 输出示例：
#   规则: "credit_score≤450 AND query_3m>5"
#   命中样本: 3000人
#   覆盖率: 6%
#   坏账率: 15% (大盘平均5%)
#   提升度: 3.0倍
#   → 结论：这条规则能精准抓住高风险客群
# ============================================================

import pandas as pd
import numpy as np


# ============================================================
# 步骤1：从LGBM模型中提取决策路径
# ============================================================

def extract_decision_paths_fast(model, feature_names, sample_data, max_depth=3):
    """
    从LGBM的每棵树中提取决策路径（规则）

    原理：
        LGBM模型 = 200棵决策树
        每棵树 = 一棵由IF-THEN组成的二叉树
        每个叶子节点 = 一条完整的决策路径

        比如一棵树：
                    [信用分<=500?]
                   /              \\
            [查询次数>3?]       [收入>10000?]
            /        \\           /         \\
          叶子A    叶子B      叶子C       叶子D

        叶子A的路径 = "信用分<=500 AND 查询次数>3"
        叶子B的路径 = "信用分<=500 AND 查询次数<=3"

    参数：
        model: 训练好的LGBM模型
        feature_names: 特征名列表
        sample_data: 要在上面"跑"模型的样本数据
        max_depth: 最多追溯几层（层数越多规则越复杂）
                   3层 = 最多3个条件组合，比如 A AND B AND C

    返回：
        Series: 规则字符串 → 出现频次（频次越高=模型越依赖这条规则）
    """
    # pred_leaf=True → 预测每个样本落在每棵树的哪个叶子上
    leaf_ids = model.predict(sample_data, pred_leaf=True)

    # dump_model() → 把模型的树结构导出为字典格式
    tree_dicts = model.dump_model()['tree_info']

    # ---- 第一步：遍历每棵树，构建"叶子ID → 路径"的映射 ----
    tree_paths = []
    for tree_idx in range(len(tree_dicts)):
        tree_struct = tree_dicts[tree_idx]['tree_structure']
        node_paths = {}  # {叶子ID: [条件1, 条件2, ...]}

        # 用"栈"遍历树（比递归更快）
        # 栈里每个元素 = (当前节点, 到达该节点的路径, 当前深度)
        stack = [(tree_struct, [], 0)]

        while stack:
            node, current_path, depth = stack.pop()

            # 到达叶子或超过最大深度 → 记录路径
            if depth >= max_depth or 'split_feature' not in node:
                if 'leaf_index' in node:
                    node_paths[node['leaf_index']] = current_path
                continue

            # 当前节点的分裂特征和阈值
            feat_idx = node['split_feature']        # 特征编号
            threshold = node['threshold']           # 阈值
            feat_name = feature_names[feat_idx]     # 特征名

            # 构造人可读的条件
            if isinstance(threshold, (int, float)):
                left_cond = f"{feat_name}≤{threshold:.2f}"   # 左子树条件
                right_cond = f"{feat_name}>{threshold:.2f}"  # 右子树条件
            else:
                left_cond = f"{feat_name}≤{threshold}"
                right_cond = f"{feat_name}>{threshold}"

            # 把左右子树压入栈继续遍历
            if 'right_child' in node:
                stack.append((node['right_child'], current_path + [right_cond], depth + 1))
            if 'left_child' in node:
                stack.append((node['left_child'], current_path + [left_cond], depth + 1))

        tree_paths.append(node_paths)

    # ---- 第二步：统计每条规则被多少样本"走过" ----
    rules = []
    for tree_idx in range(leaf_ids.shape[1]):       # 遍历每棵树
        for sample_idx in range(len(sample_data)):  # 遍历每个样本
            leaf_id = leaf_ids[sample_idx, tree_idx]
            path = tree_paths[tree_idx].get(leaf_id, [])
            if path:
                rules.append(' AND '.join(path))    # 拼接成规则字符串

    # 按频次排序：出现越多的规则 = 模型越依赖
    return pd.Series(rules).value_counts()


# ============================================================
# 步骤2：在原始数据上验证规则效果
# ============================================================

def safe_eval_rule(df, rule_str, label_col='dob4_ever10_flg'):
    """
    把规则字符串应用到数据上，返回"哪些样本命中了这条规则"

    比如规则 = "credit_score≤450 AND query_3m>5"
    → 返回一个True/False的Series：True=这个人满足条件

    为什么叫"safe"？
    → 因为从树里提取的规则格式可能有各种奇怪字符
    → 这个函数做了大量的格式清洗和异常处理
    """
    try:
        # 符号标准化：树模型用≤，pandas用<=
        normalized = (
            rule_str.replace('≤', '<=')
                    .replace('≥', '>=')
                    .replace('||', '|')
                    .strip()
        )

        # 拆分AND条件，逐个转换为pandas可执行的表达式
        conditions = []
        for raw_condition in normalized.split(' AND '):
            condition = raw_condition.strip()

            # 处理各种比较运算符
            if '<=' in condition:
                parts = [p.strip() for p in condition.split('<=') if p.strip()]
                if len(parts) == 2:
                    var, val = parts[0], parts[1]
                    conditions.append(f"`{var}`<={val}")

            elif '>=' in condition:
                parts = [p.strip() for p in condition.split('>=') if p.strip()]
                if len(parts) == 2:
                    var, val = parts[0], parts[1]
                    conditions.append(f"`{var}`>={val}")

            elif '>' in condition:
                parts = [p.strip() for p in condition.split('>') if p.strip()]
                if len(parts) == 2:
                    var, val = parts[0], parts[1]
                    conditions.append(f"`{var}`>{val}")

            elif '<' in condition:
                parts = [p.strip() for p in condition.split('<') if p.strip()]
                if len(parts) == 2:
                    var, val = parts[0], parts[1]
                    conditions.append(f"`{var}`<{val}")

        # 用 & 连接所有条件（pandas的AND）
        safe_expr = ' & '.join(conditions)
        return df.eval(safe_expr, engine='python')

    except Exception as e:
        return pd.Series(False, index=df.index)


def evaluate_rules(df, rules, label_col='dob4_ever10_flg',
                   min_coverage=0.05, max_coverage=0.2):
    """
    批量评估规则的效果

    对每条规则计算：
        1. 命中样本数 → 这条规则能抓住多少人
        2. 覆盖率 → 命中人数 / 总人数
        3. 坏账率 → 命中的人中有多少违约
        4. 提升度 → 坏账率 / 大盘平均坏账率（>1说明比平均更危险）

    参数：
        min_coverage: 最低覆盖率（太低=太苛刻，线上没意义）
        max_coverage: 最高覆盖率（太高=太宽泛，没有区分力）

    为什么限制覆盖率？
        覆盖率1% → 只抓住了1%的人，虽然精准但对大盘影响太小
        覆盖率50% → 抓住一半人，说明规则太宽泛，不够精准
        一般5%-20%是比较好的范围
    """
    results = []
    base_bad_rate = df[label_col].mean()  # 大盘平均坏账率

    for rule in rules:
        mask = safe_eval_rule(df, rule, label_col)
        if mask.sum() == 0:
            continue

        coverage = mask.mean()

        # 覆盖率不在合理范围内的规则跳过
        if not (min_coverage <= coverage <= max_coverage):
            continue

        bad_rate = df[label_col][mask].mean()  # 命中人群的坏账率
        bads = df[label_col][mask].sum()       # 命中人群中的坏客户数

        results.append({
            '规则': rule,
            '命中样本': mask.sum(),
            '坏样本数': bads,
            '覆盖率': f"{coverage:.2%}",
            '坏账率': f"{bad_rate:.2%}",
            '提升度': f"{bad_rate/base_bad_rate:.2f}x",
            # 提升度>2意味着"命中的人坏账率是平均的2倍以上"
        })

    return pd.DataFrame(results).sort_values('坏账率', ascending=False)


# ============================================================
# 步骤3：过滤不合理的规则
# ============================================================

def get_default_good_vars(df):
    """
    找出所有"模型评分"类的特征（变量名同时包含model和score）

    这些特征的特点：分数越高 → 风险越低
    所以合理的规则应该是 "score<=某个阈值" （分低=危险）
    而不是 "score>某个阈值" （分高=危险，这违反逻辑）
    """
    good_vars = []
    for col in df.columns:
        col_lower = col.lower()
        if 'model' in col_lower and 'score' in col_lower:
            good_vars.append(col)
    return good_vars


def should_exclude_rule(rule_str, good_vars=None, bad_vars=None):
    """
    判断规则是否"业务不合理"，如果是就排除

    排除逻辑：
        good_vars（分高=好）→ 如果规则说"score>X才危险"→ 违反逻辑 → 排除
        bad_vars（值大=坏）→ 如果规则说"query<X才危险"→ 违反逻辑 → 排除

    举例：
        good_var: credit_score（信用分越高越好）
        规则: "credit_score>700" → 说高分人群危险？不合理！→ 排除

        bad_var: query_count_3m（查询次数越多越危险）
        规则: "query_count_3m<2" → 说查询少的人危险？不合理！→ 排除
    """
    if not good_vars and not bad_vars:
        return False

    normalized = (
        rule_str.replace('≤', '<=')
                .replace('≥', '>=')
                .strip()
    )

    for raw_condition in normalized.split(' AND '):
        condition = raw_condition.strip()

        # good_vars出现">"条件 → 排除（分高不应该危险）
        if good_vars:
            if '>' in condition and '<' not in condition:
                parts = [p.strip() for p in condition.split('>') if p.strip()]
                if len(parts) == 2 and parts[0] in good_vars:
                    return True  # 排除这条规则
            elif '>=' in condition:
                parts = [p.strip() for p in condition.split('>=') if p.strip()]
                if len(parts) == 2 and parts[0] in good_vars:
                    return True

        # bad_vars出现"<"条件 → 排除（查询少不应该危险）
        if bad_vars:
            if '<' in condition and '>' not in condition:
                parts = [p.strip() for p in condition.split('<') if p.strip()]
                if len(parts) == 2 and parts[0] in bad_vars:
                    return True
            elif '<=' in condition:
                parts = [p.strip() for p in condition.split('<=') if p.strip()]
                if len(parts) == 2 and parts[0] in bad_vars:
                    return True

    return False  # 规则合理，保留


# ============================================================
# 实际使用：完整的规则提取流程
# ============================================================

# 第1步：从模型中提取所有规则
top_rules = extract_decision_paths_fast(
    lgbmodel,
    feature_names=train_data_x.columns.tolist(),
    sample_data=train_data_x,
    max_depth=4  # 最多4个条件组合
)
print(f"共提取到 {len(top_rules)} 条不重复规则")

# 第2步：评估规则效果（带覆盖率过滤 + 业务逻辑过滤）
rule_results = evaluate_rules(
    df=dftmp,
    rules=top_rules.index[:500],  # 取频次最高的前500条规则评估
    label_col='dob4_ever10_flg',
    min_coverage=0.05,   # 最少覆盖5%的样本
    max_coverage=0.2     # 最多覆盖20%的样本
)

print(f"\n有效规则数: {len(rule_results)}")
print("\nTop 10 高风险规则：")
print(rule_results.head(10))
