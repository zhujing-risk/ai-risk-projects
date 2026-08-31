"""
项目3：LLM驱动的风控报告自动生成器
================================
技术栈：LLM API + Pandas + Matplotlib + python-docx

输入一个贷款数据CSV，自动完成：
1. 数据EDA（探索性分析）
2. 特征重要性分析
3. 风险指标计算
4. 自然语言报告生成（带图表）

运行方式：python project3_auto_report.py

依赖安装：
pip install pandas numpy matplotlib scikit-learn python-docx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os
from datetime import datetime


# ============================================================
# 1. 数据生成（模拟实际CSV输入）
# ============================================================

def generate_loan_csv(filepath="sample_loans.csv", n=10000):
    """生成模拟贷款数据CSV"""
    np.random.seed(2024)

    data = {
        'loan_id': range(1, n+1),
        'apply_month': np.random.choice(['2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06'], n),
        'channel': np.random.choice(['SMS_A', 'SMS_B', 'SMS_C', 'APP', 'PARTNER_X', 'PARTNER_Y', 'H5'], n,
                                     p=[0.15, 0.12, 0.08, 0.25, 0.15, 0.1, 0.15]),
        'age': np.random.normal(33, 8, n).astype(int).clip(22, 55),
        'gender': np.random.choice(['M', 'F'], n, p=[0.55, 0.45]),
        'education': np.random.choice(['高中及以下', '大专', '本科', '硕士及以上'], n, p=[0.25, 0.3, 0.35, 0.1]),
        'income_monthly': (np.random.lognormal(9.2, 0.5, n)).astype(int).clip(3000, 100000),
        'credit_score': np.random.normal(570, 90, n).astype(int).clip(300, 800),
        'debt_ratio': np.random.beta(2, 5, n).round(3),
        'query_count_3m': np.random.poisson(3, n),
        'loan_count_active': np.random.poisson(2, n),
        'loan_amount': np.random.choice([3000, 5000, 10000, 20000, 30000, 50000], n,
                                         p=[0.1, 0.2, 0.3, 0.2, 0.12, 0.08]),
        'interest_rate': np.random.choice([0.099, 0.12, 0.15, 0.18, 0.24], n,
                                           p=[0.1, 0.2, 0.3, 0.25, 0.15]),
        'term': np.random.choice([6, 12, 24, 36], n, p=[0.2, 0.4, 0.3, 0.1]),
    }

    df = pd.DataFrame(data)

    # 构造违约标签（与特征有因果关系）
    logit = (-3
             + 0.02 * (df['age'] < 26).astype(int)
             - 0.005 * df['credit_score']
             + 2.0 * df['debt_ratio']
             + 0.15 * df['query_count_3m']
             + 0.2 * df['loan_count_active']
             - 0.00005 * df['income_monthly']
             + 0.5 * (df['education'] == '高中及以下').astype(int))
    prob = 1 / (1 + np.exp(-logit))
    df['is_default_m6'] = (np.random.random(n) < prob).astype(int)
    df['dpd_max'] = np.where(df['is_default_m6'] == 1, np.random.choice([31, 45, 60, 90, 120], n), 0)

    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    return filepath


# ============================================================
# 2. 自动EDA模块
# ============================================================

class AutoEDA:
    def __init__(self, df):
        self.df = df
        self.findings = []

    def run(self):
        print("\n📊 ========== 自动EDA分析 ==========")
        self.basic_stats()
        self.target_analysis()
        self.feature_analysis()
        self.channel_analysis()
        return self.findings

    def basic_stats(self):
        n = len(self.df)
        n_default = self.df['is_default_m6'].sum()
        rate = n_default / n * 100

        finding = f"数据集共 {n:,} 条记录，其中违约客户 {n_default:,} 人，整体违约率 {rate:.2f}%"
        self.findings.append(("基础统计", finding))
        print(f"  ✅ {finding}")

    def target_analysis(self):
        monthly = self.df.groupby('apply_month')['is_default_m6'].agg(['count', 'mean'])
        monthly['mean'] = (monthly['mean'] * 100).round(2)

        worst_month = monthly['mean'].idxmax()
        best_month = monthly['mean'].idxmin()

        finding = f"月度违约率范围: {monthly['mean'].min():.2f}% ~ {monthly['mean'].max():.2f}%。最差月份: {worst_month}（{monthly.loc[worst_month, 'mean']:.2f}%），最优月份: {best_month}（{monthly.loc[best_month, 'mean']:.2f}%）"
        self.findings.append(("月度趋势", finding))
        print(f"  ✅ {finding}")

    def feature_analysis(self):
        # 信用分与违约的关系
        bins = [0, 450, 550, 650, 800]
        labels = ['<450', '450-550', '550-650', '>650']
        self.df['score_bin'] = pd.cut(self.df['credit_score'], bins=bins, labels=labels)
        score_default = self.df.groupby('score_bin', observed=True)['is_default_m6'].mean() * 100

        finding = f"信用分区分力明显: <450分违约率{score_default.iloc[0]:.1f}%, >650分违约率{score_default.iloc[-1]:.1f}%, 差异{score_default.iloc[0]-score_default.iloc[-1]:.1f}pp"
        self.findings.append(("信用分分析", finding))
        print(f"  ✅ {finding}")

    def channel_analysis(self):
        ch = self.df.groupby('channel')['is_default_m6'].agg(['count', 'mean'])
        ch['mean'] = (ch['mean'] * 100).round(2)
        worst_ch = ch['mean'].idxmax()
        best_ch = ch['mean'].idxmin()

        finding = f"渠道风险差异显著: 最高 {worst_ch}（{ch.loc[worst_ch, 'mean']:.2f}%），最低 {best_ch}（{ch.loc[best_ch, 'mean']:.2f}%），差异{ch.loc[worst_ch, 'mean']-ch.loc[best_ch, 'mean']:.2f}pp"
        self.findings.append(("渠道分析", finding))
        print(f"  ✅ {finding}")


# ============================================================
# 3. 特征重要性分析
# ============================================================

class FeatureImportanceAnalyzer:
    def __init__(self, df):
        self.df = df

    def run(self):
        print("\n🔍 ========== 特征重要性分析 ==========")

        feature_cols = ['age', 'credit_score', 'debt_ratio', 'query_count_3m',
                       'loan_count_active', 'income_monthly', 'loan_amount', 'interest_rate', 'term']

        X = self.df[feature_cols].fillna(0)
        y = self.df['is_default_m6']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        print(f"  ✅ 模型AUC: {auc:.4f}")

        importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        print(f"\n  📋 Top特征重要性排序：")
        for i, (feat, imp) in enumerate(importance.items()):
            bar = "█" * int(imp * 50)
            print(f"     {i+1}. {feat:20s} {imp:.4f} {bar}")

        return importance, auc


# ============================================================
# 4. 可视化报告生成
# ============================================================

class ReportGenerator:
    def __init__(self, df, findings, importance, auc):
        self.df = df
        self.findings = findings
        self.importance = importance
        self.auc = auc
        self.output_dir = "report_output"
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_all_charts(self):
        print("\n🎨 ========== 生成可视化图表 ==========")

        fig = plt.figure(figsize=(16, 20))

        # 图1: 月度进件量与违约率
        ax1 = fig.add_subplot(4, 2, 1)
        monthly = self.df.groupby('apply_month').agg(
            进件量=('loan_id', 'count'),
            违约率=('is_default_m6', 'mean')
        )
        ax1_twin = ax1.twinx()
        ax1.bar(range(len(monthly)), monthly['进件量'], alpha=0.6, color='#42A5F5', label='进件量')
        ax1_twin.plot(range(len(monthly)), monthly['违约率']*100, 'r-o', linewidth=2, label='违约率')
        ax1.set_xticks(range(len(monthly)))
        ax1.set_xticklabels(monthly.index, rotation=45)
        ax1.set_ylabel('进件量')
        ax1_twin.set_ylabel('违约率(%)', color='red')
        ax1.set_title('月度进件量 & 违约率趋势', fontweight='bold')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')

        # 图2: 渠道违约率对比
        ax2 = fig.add_subplot(4, 2, 2)
        ch_default = self.df.groupby('channel')['is_default_m6'].mean().sort_values() * 100
        colors = ['#4CAF50' if v < ch_default.median() else '#F44336' for v in ch_default.values]
        bars = ax2.barh(ch_default.index, ch_default.values, color=colors)
        ax2.set_xlabel('违约率 (%)')
        ax2.set_title('分渠道违约率（红色=高于中位数）', fontweight='bold')
        for bar, val in zip(bars, ch_default.values):
            ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', va='center')

        # 图3: 信用分分布
        ax3 = fig.add_subplot(4, 2, 3)
        good = self.df[self.df['is_default_m6'] == 0]['credit_score']
        bad = self.df[self.df['is_default_m6'] == 1]['credit_score']
        ax3.hist(good, bins=40, alpha=0.6, density=True, label=f'正常 (n={len(good):,})', color='#2196F3')
        ax3.hist(bad, bins=40, alpha=0.6, density=True, label=f'违约 (n={len(bad):,})', color='#F44336')
        ax3.set_xlabel('信用分')
        ax3.set_ylabel('密度')
        ax3.set_title('信用分分布（好坏客户对比）', fontweight='bold')
        ax3.legend()

        # 图4: 特征重要性
        ax4 = fig.add_subplot(4, 2, 4)
        top_features = self.importance.head(8)
        ax4.barh(top_features.index[::-1], top_features.values[::-1], color='#7B1FA2')
        ax4.set_xlabel('重要性')
        ax4.set_title(f'Top特征重要性 (模型AUC={self.auc:.4f})', fontweight='bold')

        # 图5: 年龄vs违约率
        ax5 = fig.add_subplot(4, 2, 5)
        age_bins = pd.cut(self.df['age'], bins=[20, 25, 30, 35, 40, 45, 55])
        age_data = self.df.groupby(age_bins, observed=True).agg(
            人数=('loan_id', 'count'),
            违约率=('is_default_m6', 'mean')
        )
        ax5.bar(range(len(age_data)), age_data['人数'], alpha=0.5, color='#90CAF9', label='人数')
        ax5_twin = ax5.twinx()
        ax5_twin.plot(range(len(age_data)), age_data['违约率']*100, 'ro-', linewidth=2, label='违约率')
        ax5.set_xticks(range(len(age_data)))
        ax5.set_xticklabels(['22-25', '26-30', '31-35', '36-40', '41-45', '46-55'])
        ax5.set_title('年龄段分布 & 违约率', fontweight='bold')
        ax5.set_ylabel('人数')
        ax5_twin.set_ylabel('违约率(%)', color='red')

        # 图6: 负债率vs违约率
        ax6 = fig.add_subplot(4, 2, 6)
        debt_bins = pd.cut(self.df['debt_ratio'], bins=[0, 0.1, 0.2, 0.3, 0.5, 1.0])
        debt_default = self.df.groupby(debt_bins, observed=True)['is_default_m6'].mean() * 100
        ax6.plot(range(len(debt_default)), debt_default.values, 'g-s', linewidth=2, markersize=8)
        ax6.set_xticks(range(len(debt_default)))
        ax6.set_xticklabels(['0-10%', '10-20%', '20-30%', '30-50%', '50%+'], rotation=45)
        ax6.set_title('负债率 vs 违约率', fontweight='bold')
        ax6.set_ylabel('违约率 (%)')
        ax6.set_xlabel('负债率区间')
        ax6.grid(True, alpha=0.3)

        # 图7: 学历vs违约率
        ax7 = fig.add_subplot(4, 2, 7)
        edu_order = ['高中及以下', '大专', '本科', '硕士及以上']
        edu_data = self.df.groupby('education')['is_default_m6'].agg(['count', 'mean'])
        edu_data = edu_data.reindex(edu_order)
        ax7.bar(edu_order, edu_data['mean']*100, color=['#FFCDD2', '#EF9A9A', '#90CAF9', '#42A5F5'])
        ax7.set_title('学历 vs 违约率', fontweight='bold')
        ax7.set_ylabel('违约率 (%)')

        # 图8: 额度vs违约率
        ax8 = fig.add_subplot(4, 2, 8)
        amt_default = self.df.groupby('loan_amount')['is_default_m6'].mean() * 100
        ax8.plot(amt_default.index/10000, amt_default.values, 'b-o', linewidth=2)
        ax8.set_xlabel('贷款金额 (万元)')
        ax8.set_ylabel('违约率 (%)')
        ax8.set_title('贷款金额 vs 违约率', fontweight='bold')
        ax8.grid(True, alpha=0.3)

        plt.tight_layout(pad=3.0)
        chart_path = os.path.join(self.output_dir, 'full_analysis_charts.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 图表已保存: {chart_path}")
        return chart_path

    def generate_markdown_report(self, chart_path):
        """生成Markdown格式的完整报告"""
        print("\n📝 ========== 生成分析报告 ==========")

        report = f"""# 📊 信贷资产风险分析报告

> **生成时间：** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> **数据来源：** sample_loans.csv
> **样本量：** {len(self.df):,} 条贷款记录
> **分析引擎：** LLM + GradientBoosting + AutoEDA

---

## 一、核心指标摘要

| 指标 | 数值 |
|------|------|
| 总进件量 | {len(self.df):,} |
| 违约客户数 | {self.df['is_default_m6'].sum():,} |
| 整体违约率 | {self.df['is_default_m6'].mean()*100:.2f}% |
| 平均贷款金额 | ¥{self.df['loan_amount'].mean():,.0f} |
| 平均信用分 | {self.df['credit_score'].mean():.0f} |
| 模型AUC | {self.auc:.4f} |

---

## 二、自动化发现（AI Generated Insights）

"""
        for i, (title, finding) in enumerate(self.findings):
            report += f"### {i+1}. {title}\n\n{finding}\n\n"

        report += f"""---

## 三、特征重要性排序

基于GradientBoosting模型的特征重要性分析：

| 排名 | 特征 | 重要性 | 业务含义 |
|------|------|--------|---------|
"""
        feature_meanings = {
            'credit_score': '信用评分，综合反映客户信用状况',
            'debt_ratio': '负债比率，衡量还款压力',
            'query_count_3m': '近3月征信查询次数，反映资金饥渴度',
            'loan_count_active': '在贷笔数，多头借贷风险指标',
            'income_monthly': '月收入，还款能力核心指标',
            'age': '年龄，与风险承受能力相关',
            'loan_amount': '贷款金额，影响还款压力',
            'interest_rate': '利率，高利率客群通常风险更高',
            'term': '期限，长期限客户不确定性更大',
        }

        for i, (feat, imp) in enumerate(self.importance.items()):
            meaning = feature_meanings.get(feat, '-')
            report += f"| {i+1} | {feat} | {imp:.4f} | {meaning} |\n"

        report += f"""
---

## 四、分渠道表现

"""
        ch_stats = self.df.groupby('channel').agg(
            进件量=('loan_id', 'count'),
            平均额度=('loan_amount', 'mean'),
            违约率=('is_default_m6', 'mean'),
            平均信用分=('credit_score', 'mean'),
        ).round(4)

        report += "| 渠道 | 进件量 | 平均额度 | 违约率 | 平均信用分 |\n"
        report += "|------|--------|---------|--------|----------|\n"
        for ch, row in ch_stats.iterrows():
            report += f"| {ch} | {int(row['进件量']):,} | ¥{row['平均额度']:,.0f} | {row['违约率']*100:.2f}% | {row['平均信用分']:.0f} |\n"

        report += f"""

---

## 五、策略建议（AI Generated Recommendations）

基于以上分析，AI Agent给出以下策略建议：

### 5.1 短期优化（立即可执行）
1. **高风险渠道收紧**：对违约率超过均值1.5倍的渠道，收紧准入评分门槛+20分
2. **薄信用客群**：对22-25岁、征信查询>5次的客群，增加学历/社保验证环节
3. **额度管控**：负债率>30%的客户，额度上限降低至当前的60%

### 5.2 中期优化（1-3个月）
4. **模型迭代**：当前AUC={self.auc:.4f}，建议补充行为数据（APP使用、还款习惯）提升至0.75+
5. **差异化定价**：A档客户利率可下调至9.9%以提升竞争力，C/D档维持24%覆盖风险

### 5.3 监控预警
6. **PSI监控**：月度检查评分稳定性，PSI>0.25触发模型重训
7. **Early Warning**：FPD7上升超过基线20%时，自动触发策略收紧预案

---

## 六、可视化附录

详见图表文件：`{chart_path}`

包含：月度趋势、渠道对比、信用分分布、特征重要性、年龄分析、负债率分析、学历分析、额度分析

---

*本报告由 AI 自动生成 | 技术栈: LLM + Pandas + Scikit-learn + Matplotlib*
*模型及结论仅供参考，具体策略执行请结合业务判断*
"""

        report_path = os.path.join(self.output_dir, 'risk_analysis_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"  ✅ 报告已保存: {report_path}")
        return report_path


# ============================================================
# 5. 主程序
# ============================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║        📊 LLM驱动的风控报告自动生成器 v1.0                       ║
║        输入CSV → 自动EDA → 建模 → 生成报告                       ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    # Step 1: 加载/生成数据
    print("📂 Step 1: 准备数据...")
    csv_path = generate_loan_csv()
    df = pd.read_csv(csv_path)
    print(f"   ✅ 已加载 {csv_path}，共 {len(df):,} 条记录")

    # Step 2: 自动EDA
    eda = AutoEDA(df)
    findings = eda.run()

    # Step 3: 特征重要性分析
    analyzer = FeatureImportanceAnalyzer(df)
    importance, auc = analyzer.run()

    # Step 4: 生成可视化和报告
    generator = ReportGenerator(df, findings, importance, auc)
    chart_path = generator.generate_all_charts()
    report_path = generator.generate_markdown_report(chart_path)

    # 完成
    print("\n" + "="*60)
    print("🎉 报告生成完成！输出文件：")
    print("="*60)
    print(f"  📄 {report_path}           — Markdown完整报告")
    print(f"  📊 {chart_path}  — 8张分析图表")
    print(f"  📁 {csv_path}                   — 原始数据文件")
    print()
    print("💡 在实际项目中：")
    print("   - 替换generate_loan_csv()为读取真实数据")
    print("   - EDA findings由LLM根据数据特征自动生成自然语言洞察")
    print("   - 策略建议由LLM根据分析结果+历史策略库综合生成")
    print("   - 可接入Streamlit做成Web界面，上传CSV即出报告")


if __name__ == "__main__":
    main()
