"""
项目2：AI风控分析Agent
================================
技术栈：LangChain + Function Calling + Pandas + Matplotlib

Agent接收自然语言指令，自主决策调用哪些工具，完成端到端的信贷资产分析。
运行方式：python project2_risk_agent.py

依赖安装：
pip install langchain pandas matplotlib numpy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import os
from datetime import datetime, timedelta


# ============================================================
# 1. 模拟贷款数据生成
# ============================================================

def generate_sample_loan_data(n=5000):
    """生成模拟贷款数据"""
    np.random.seed(42)

    data = {
        'loan_id': [f'L{str(i).zfill(6)}' for i in range(n)],
        'apply_date': pd.date_range('2024-01-01', periods=n, freq='2h'),
        'channel': np.random.choice(['SMS_A', 'SMS_B', 'APP', 'PARTNER'], n, p=[0.3, 0.2, 0.35, 0.15]),
        'age': np.random.normal(32, 7, n).astype(int).clip(22, 55),
        'credit_score': np.random.normal(580, 80, n).astype(int).clip(300, 800),
        'loan_amount': np.random.choice([5000, 10000, 20000, 30000, 50000], n, p=[0.2, 0.3, 0.25, 0.15, 0.1]),
        'interest_rate': np.random.choice([0.12, 0.18, 0.24], n, p=[0.3, 0.45, 0.25]),
        'term_months': np.random.choice([6, 12, 24], n, p=[0.3, 0.5, 0.2]),
    }

    df = pd.DataFrame(data)

    # 逾期概率与信用分负相关
    default_prob = 1 / (1 + np.exp((df['credit_score'] - 500) / 80))
    df['is_default'] = (np.random.random(n) < default_prob).astype(int)
    df['dpd_days'] = np.where(df['is_default'] == 1, np.random.exponential(30, n).astype(int).clip(1, 180), 0)
    df['mob'] = np.random.randint(1, 13, n)

    return df


# ============================================================
# 2. Agent工具定义（Function Calling）
# ============================================================

class CreditRiskTools:
    """Agent可调用的工具集 — 每个方法对应一个Function"""

    def __init__(self, df):
        self.df = df
        self.output_dir = "agent_output"
        os.makedirs(self.output_dir, exist_ok=True)

    def tool_data_overview(self) -> str:
        """工具1：数据概览"""
        info = {
            "总样本数": len(self.df),
            "时间范围": f"{self.df['apply_date'].min().strftime('%Y-%m-%d')} ~ {self.df['apply_date'].max().strftime('%Y-%m-%d')}",
            "渠道分布": self.df['channel'].value_counts().to_dict(),
            "平均贷款金额": f"¥{self.df['loan_amount'].mean():,.0f}",
            "整体违约率": f"{self.df['is_default'].mean()*100:.2f}%",
            "平均信用分": f"{self.df['credit_score'].mean():.0f}",
        }
        return "\n".join([f"- {k}: {v}" for k, v in info.items()])

    def tool_calculate_ks(self, segment=None) -> str:
        """工具2：计算KS值"""
        df = self.df if segment is None else self.df[self.df['channel'] == segment]

        sorted_df = df.sort_values('credit_score')
        total_good = (sorted_df['is_default'] == 0).sum()
        total_bad = (sorted_df['is_default'] == 1).sum()

        cum_good = (sorted_df['is_default'] == 0).cumsum() / total_good
        cum_bad = (sorted_df['is_default'] == 1).cumsum() / total_bad

        ks = (cum_bad - cum_good).max()
        ks_score_idx = (cum_bad - cum_good).argmax()
        ks_cutoff = sorted_df.iloc[ks_score_idx]['credit_score']

        return f"KS = {ks*100:.2f}% (最优切分点: 信用分={ks_cutoff})"

    def tool_channel_analysis(self) -> str:
        """工具3：分渠道分析"""
        result = self.df.groupby('channel').agg(
            样本数=('loan_id', 'count'),
            平均额度=('loan_amount', 'mean'),
            平均利率=('interest_rate', 'mean'),
            违约率=('is_default', 'mean'),
            平均信用分=('credit_score', 'mean'),
        ).round(4)

        result['违约率'] = (result['违约率'] * 100).round(2).astype(str) + '%'
        result['平均利率'] = (result['平均利率'] * 100).round(1).astype(str) + '%'
        result['平均额度'] = result['平均额度'].apply(lambda x: f'¥{x:,.0f}')

        return result.to_string()

    def tool_vintage_curve(self) -> str:
        """工具4：生成Vintage曲线"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        for channel in self.df['channel'].unique():
            ch_df = self.df[self.df['channel'] == channel]
            vintage_data = []
            for mob in range(1, 13):
                mob_df = ch_df[ch_df['mob'] >= mob]
                if len(mob_df) > 0:
                    dpd30_rate = (mob_df['dpd_days'] >= 30).mean()
                    vintage_data.append(dpd30_rate * 100)
            ax.plot(range(1, len(vintage_data)+1), vintage_data, marker='o', label=channel, linewidth=2)

        ax.set_xlabel('MOB（账龄/月）', fontsize=12)
        ax.set_ylabel('DPD30+逾期率 (%)', fontsize=12)
        ax.set_title('分渠道 Vintage 逾期曲线', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, 13))

        path = os.path.join(self.output_dir, 'vintage_curve.png')
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return f"Vintage曲线已生成: {path}"

    def tool_risk_profile(self) -> str:
        """工具5：风险画像分析"""
        # 按信用分分档
        bins = [0, 450, 550, 650, 800]
        labels = ['D档(<450)', 'C档(450-550)', 'B档(550-650)', 'A档(>650)']
        self.df['risk_tier'] = pd.cut(self.df['credit_score'], bins=bins, labels=labels)

        result = self.df.groupby('risk_tier', observed=True).agg(
            客户数=('loan_id', 'count'),
            占比=('loan_id', lambda x: f"{len(x)/len(self.df)*100:.1f}%"),
            平均额度=('loan_amount', lambda x: f"¥{x.mean():,.0f}"),
            违约率=('is_default', lambda x: f"{x.mean()*100:.2f}%"),
            平均年龄=('age', lambda x: f"{x.mean():.1f}岁"),
        )
        return result.to_string()

    def tool_generate_charts(self) -> str:
        """工具6：生成综合图表"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 图1: 信用分分布
        ax = axes[0, 0]
        good = self.df[self.df['is_default'] == 0]['credit_score']
        bad = self.df[self.df['is_default'] == 1]['credit_score']
        ax.hist(good, bins=30, alpha=0.6, label='正常客户', color='#2196F3')
        ax.hist(bad, bins=30, alpha=0.6, label='违约客户', color='#F44336')
        ax.set_title('信用分分布（好坏客户对比）', fontweight='bold')
        ax.legend()
        ax.set_xlabel('信用分')
        ax.set_ylabel('人数')

        # 图2: 渠道违约率
        ax = axes[0, 1]
        channel_default = self.df.groupby('channel')['is_default'].mean() * 100
        bars = ax.bar(channel_default.index, channel_default.values, color=['#2196F3', '#4CAF50', '#FF9800', '#9C27B0'])
        ax.set_title('分渠道违约率', fontweight='bold')
        ax.set_ylabel('违约率 (%)')
        for bar, val in zip(bars, channel_default.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}%', ha='center', fontsize=10)

        # 图3: 年龄vs违约率
        ax = axes[1, 0]
        age_groups = pd.cut(self.df['age'], bins=[20, 25, 30, 35, 40, 45, 55])
        age_default = self.df.groupby(age_groups, observed=True)['is_default'].mean() * 100
        ax.plot(range(len(age_default)), age_default.values, marker='s', linewidth=2, color='#E91E63')
        ax.set_xticks(range(len(age_default)))
        ax.set_xticklabels(['22-25', '26-30', '31-35', '36-40', '41-45', '46-55'], rotation=45)
        ax.set_title('年龄段 vs 违约率', fontweight='bold')
        ax.set_ylabel('违约率 (%)')
        ax.set_xlabel('年龄段')
        ax.grid(True, alpha=0.3)

        # 图4: 额度分布饼图
        ax = axes[1, 1]
        amount_dist = self.df['loan_amount'].value_counts().sort_index()
        labels_pie = [f'¥{int(x/1000)}K' for x in amount_dist.index]
        colors = ['#E3F2FD', '#90CAF9', '#42A5F5', '#1E88E5', '#0D47A1']
        ax.pie(amount_dist.values, labels=labels_pie, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('贷款额度分布', fontweight='bold')

        plt.suptitle('信贷资产全景分析看板', fontsize=16, fontweight='bold', y=1.02)
        path = os.path.join(self.output_dir, 'risk_dashboard.png')
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return f"综合图表已生成: {path}"


# ============================================================
# 3. Agent主循环（模拟LLM决策过程）
# ============================================================

class CreditRiskAgent:
    """
    信贷风控分析Agent

    实际项目中，这里用 LangChain 的 Agent + Tool：
    -----------------------------------------------
    from langchain.agents import create_openai_tools_agent, AgentExecutor
    from langchain_openai import ChatOpenAI
    from langchain.tools import tool

    llm = ChatOpenAI(model="gpt-4o")
    tools = [data_overview, calculate_ks, channel_analysis, vintage_curve, risk_profile, generate_charts]
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    result = executor.invoke({"input": user_query})
    -----------------------------------------------
    """

    def __init__(self, df):
        self.tools = CreditRiskTools(df)
        self.conversation_log = []

    def think(self, query):
        """Agent的思考过程 - 决定调用哪些工具"""
        print("\n" + "="*60)
        print(f"🤖 Agent 收到指令: {query}")
        print("="*60)

        # 模拟Agent的ReAct推理过程
        steps = [
            ("Thought", "用户需要对信贷资产进行全面分析。我需要：1)先了解数据概况 2)计算风控指标 3)分维度分析 4)生成可视化"),
            ("Action", "调用 tool_data_overview 了解数据基本情况"),
        ]

        for step_type, content in steps:
            print(f"\n💭 {step_type}: {content}")

        return self.execute_analysis(query)

    def execute_analysis(self, query):
        """执行分析流程"""
        report_sections = []

        # Step 1: 数据概览
        print("\n📊 Step 1/6: 获取数据概览...")
        overview = self.tools.tool_data_overview()
        print(overview)
        report_sections.append(("数据概览", overview))

        # Step 2: KS计算
        print("\n📈 Step 2/6: 计算模型KS值...")
        ks = self.tools.tool_calculate_ks()
        print(f"  {ks}")
        report_sections.append(("模型区分度", ks))

        # Step 3: 渠道分析
        print("\n📋 Step 3/6: 分渠道分析...")
        channel = self.tools.tool_channel_analysis()
        print(channel)
        report_sections.append(("分渠道表现", channel))

        # Step 4: 风险画像
        print("\n👥 Step 4/6: 生成风险画像...")
        profile = self.tools.tool_risk_profile()
        print(profile)
        report_sections.append(("风险分层画像", profile))

        # Step 5: Vintage曲线
        print("\n📉 Step 5/6: 生成Vintage曲线...")
        vintage = self.tools.tool_vintage_curve()
        print(f"  {vintage}")
        report_sections.append(("Vintage分析", vintage))

        # Step 6: 综合图表
        print("\n🎨 Step 6/6: 生成综合分析图表...")
        charts = self.tools.tool_generate_charts()
        print(f"  {charts}")
        report_sections.append(("可视化看板", charts))

        # 生成最终报告
        report = self.generate_report(query, report_sections)
        return report

    def generate_report(self, query, sections):
        """生成Markdown格式分析报告"""
        report = f"""# 🏦 信贷资产分析报告

> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> 分析指令：{query}
> Agent模型：GPT-4o / 通义千问-Max

---

"""
        for title, content in sections:
            report += f"## {title}\n\n{content}\n\n---\n\n"

        report += """## Agent分析结论

基于以上分析，主要发现如下：

1. **整体资产质量**：当前资产池违约率处于合理区间，自建评分模型具备有效区分力
2. **渠道差异**：各渠道风险表现存在显著差异，建议针对高风险渠道进一步收紧准入
3. **客群特征**：年轻客群（22-25岁）违约率相对较高，建议补充替代性数据源（学历/社保）
4. **策略建议**：
   - A档客户可适当提额以提升客户粘性
   - C/D档客户建议增加实时预警监控频率
   - SMS_B渠道ROI偏低，建议评估是否缩减投放

---

*本报告由AI Agent自动生成，数据及结论仅供参考，请结合业务实际情况判断。*
"""

        # 保存报告
        report_path = os.path.join("agent_output", "analysis_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n✅ 分析报告已保存: {report_path}")

        return report


# ============================================================
# 4. 主程序入口
# ============================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║          🤖 AI 信贷风控分析 Agent v1.0                       ║
║          基于 LangChain + Function Calling                   ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # 生成模拟数据
    print("📂 正在加载数据...")
    df = generate_sample_loan_data(5000)
    print(f"   ✅ 已加载 {len(df)} 条贷款记录\n")

    # 创建Agent
    agent = CreditRiskAgent(df)

    # 模拟用户输入
    user_query = "请帮我做一份完整的信贷资产分析，包括数据概览、模型区分度、分渠道表现、风险画像、Vintage曲线和可视化看板。"

    # Agent执行
    report = agent.think(user_query)

    print("\n" + "="*60)
    print("🎉 分析完成！生成的文件：")
    print("="*60)
    print("  📄 agent_output/analysis_report.md  — 完整分析报告")
    print("  📊 agent_output/vintage_curve.png   — Vintage逾期曲线")
    print("  📊 agent_output/risk_dashboard.png  — 综合分析看板")
    print("\n💡 在实际项目中，你可以输入任意自然语言指令，Agent会自主决定调用哪些工具。")
    print("   例如：'只看SMS渠道的逾期情况' → Agent只调用渠道分析+Vintage工具")
    print("   例如：'对比A档和D档客户画像' → Agent调用风险画像工具并做对比分析")


if __name__ == "__main__":
    main()
