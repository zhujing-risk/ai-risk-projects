"""
项目1：信贷策略知识库问答系统
================================
技术栈：LangChain + ChromaDB + 通义千问/OpenAI + Streamlit

这是一个完整可运行的Demo，展示最终项目的架构和效果。
运行方式：streamlit run project1_rag_qa_system.py

依赖安装：
pip install streamlit langchain langchain-community chromadb dashscope tiktoken
"""

import streamlit as st
import json
import os
from datetime import datetime

# ============================================================
# 模拟数据 - 实际项目中这些来自真实的策略文档PDF
# ============================================================
SAMPLE_STRATEGY_DOCS = [
    {
        "title": "短信渠道准入策略V3.2",
        "content": """
        短信获客渠道准入策略规则：
        1. 基础准入条件：年龄22-55岁，有稳定收入来源，征信查询次数近3个月≤6次
        2. 风险分层：使用自建LightGBM模型评分（KS=20%），按分数分为A/B/C/D四档
        3. A档客户：评分>650，直接通过，额度上限5万
        4. B档客户：评分550-650，通过率约70%，需补充验证，额度上限3万
        5. C档客户：评分450-550，通过率约30%，强验证+降额，额度上限1万
        6. D档客户：评分<450，拒绝
        7. 定价策略：A档年化12%，B档年化18%，C档年化24%
        8. 更新时间：2025年6月，目标终态损失率4.2%
        """,
        "metadata": {"channel": "SMS", "version": "3.2", "update_date": "2025-06"}
    },
    {
        "title": "定价准入策略优化方案",
        "content": """
        定价准入策略优化：
        1. 背景：原策略24个月定价通过率仅4.5%，大量低风险客户被误拒
        2. 优化方案：按渠道差异化定价准入
           - 自有APP渠道：放宽至评分>500即可进入定价环节
           - 短信渠道：维持评分>550的准入门槛
           - 联合贷渠道：评分>600，因资金方要求更严格
        3. 效果：通过率从4.5%提升至9.0%（+5pp）
        4. 风险验证：捞回客群MOB5逾期仅为大盘平均的0.7倍（"放量不放险"）
        5. 关键指标监控：每周观察FPD7、MOB2、MOB5逾期率变化
        """,
        "metadata": {"type": "pricing", "version": "2.0", "update_date": "2025-07"}
    },
    {
        "title": "自建评分模型开发文档",
        "content": """
        自建信用评分模型技术文档：
        1. 模型架构：LightGBM + Logistic Regression融合
        2. 特征工程：500+特征变量
           - 征信特征：查询次数、负债比、历史逾期
           - 行为特征：APP使用频率、页面停留时间
           - 多头借贷：近期申请平台数、已有贷款笔数
           - 时序特征：收入波动趋势、消费变化率
        3. 模型性能：KS=20%, AUC=60%, IV top特征>0.1
        4. 稳定性监控：月度PSI<0.1视为稳定，>0.25触发模型重训
        5. 上线替代：替代29个外部模型评分，数据成本节省30%
        6. 样本：好坏样本定义为MOB6 DPD30+，样本窗口2024.01-2024.06
        """,
        "metadata": {"type": "model", "version": "1.0", "update_date": "2025-05"}
    },
    {
        "title": "催收策略与失联修复",
        "content": """
        催收策略分层：
        1. M1阶段（逾期1-30天）：
           - 智能外呼为主，人工催收为辅
           - 按还款意愿评分分配催收强度
           - 目标回收率：85%+
        2. M2阶段（逾期31-60天）：
           - 人工催收为主，每日至少触达1次
           - 启动失联修复流程：大数据查找备用联系方式
           - 目标回收率：60%+
        3. M3+阶段（逾期60天以上）：
           - 委外催收或法催评估
           - 核销标准：M6且回收可能性<5%
        4. 催收回收率归因：迁徙风险贡献约40%，结构性风险贡献约60%
        """,
        "metadata": {"type": "collection", "version": "1.5", "update_date": "2025-04"}
    },
    {
        "title": "年轻客群放量策略",
        "content": """
        年轻客群（22-28岁）策略优化：
        1. 痛点：年轻客群征信数据薄，传统评分模型区分度低
        2. 解决方案：
           - 接入学历直连验证（学信网API）
           - 重设客户分层：本科及以上学历客户单独分群
           - 退退低区分度规则（如"工作年限<2年拒绝"）
        3. 效果：
           - 通过率提升7pp
           - FPD7风险下降21%
           - 实现放量与控险并举
        4. 经验总结：替代性数据（学历、社保）对薄信用人群有显著区分力
        """,
        "metadata": {"type": "segment", "version": "1.0", "update_date": "2025-01"}
    },
]

# ============================================================
# RAG核心逻辑（简化版展示）
# 实际项目中用 LangChain + ChromaDB 实现
# ============================================================

def simple_keyword_search(query, docs, top_k=3):
    """简化的检索逻辑 - 实际项目中用向量相似度检索"""
    scores = []
    for doc in docs:
        score = 0
        query_terms = query.lower().replace("？", "").replace("?", "")
        for term in query_terms:
            if term in doc["content"].lower():
                score += 1
            if term in doc["title"].lower():
                score += 2
        scores.append((score, doc))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scores[:top_k]]


def generate_answer(query, context_docs):
    """
    模拟LLM回答 - 实际项目中调用通义千问/OpenAI API

    实际代码：
    from langchain.chains import RetrievalQA
    from langchain_community.llms import Tongyi

    llm = Tongyi(model_name="qwen-max", dashscope_api_key="your-key")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    answer = qa_chain.run(query)
    """
    context_text = "\n---\n".join([doc["content"] for doc in context_docs])
    sources = [doc["title"] for doc in context_docs]

    # 模拟回答（实际由LLM生成）
    answers_map = {
        "准入": f"""根据策略文档，当前短信渠道的准入条件如下：

**基础条件：**
- 年龄：22-55岁
- 收入：有稳定收入来源
- 征信查询：近3个月≤6次

**风险分层（自建LightGBM模型，KS=20%）：**
| 档位 | 评分范围 | 通过率 | 额度上限 | 定价（年化） |
|------|---------|--------|---------|------------|
| A档 | >650 | ~100% | 5万 | 12% |
| B档 | 550-650 | ~70% | 3万 | 18% |
| C档 | 450-550 | ~30% | 1万 | 24% |
| D档 | <450 | 拒绝 | - | - |

**目标：** 终态损失率控制在4.2%以内。

📄 来源：{', '.join(sources)}""",

        "定价": f"""关于定价准入策略的优化情况：

**优化前：** 24个月定价通过率仅4.5%，存在大量低风险客户被误拒的问题。

**优化方案（分渠道差异化）：**
- 自有APP渠道：评分>500即可进入定价环节
- 短信渠道：维持评分>550
- 联合贷渠道：评分>600（资金方要求更严格）

**优化效果：**
- ✅ 通过率：4.5% → 9.0%（+5pp）
- ✅ 风险验证：捞回客群MOB5逾期 = 大盘平均 × 0.7
- ✅ 实现"放量不放险"

**监控指标：** 每周观察FPD7、MOB2、MOB5逾期率变化

📄 来源：{', '.join(sources)}""",

        "模型": f"""关于自建评分模型的技术详情：

**模型架构：** LightGBM + Logistic Regression 融合模型

**特征工程（500+变量）：**
- 征信特征：查询次数、负债比、历史逾期
- 行为特征：APP使用频率、页面停留时间
- 多头借贷：近期申请平台数、已有贷款笔数
- 时序特征：收入波动趋势、消费变化率

**模型性能：**
| 指标 | 数值 |
|------|------|
| KS | 20% |
| AUC | 60% |
| Top特征IV | >0.1 |

**稳定性监控：**
- PSI < 0.1 → 稳定
- PSI > 0.25 → 触发模型重训

**业务成果：** 替代29个外部模型评分，数据成本节省30%

📄 来源：{', '.join(sources)}""",

        "催收": f"""关于催收策略的分层体系：

**M1阶段（逾期1-30天）：**
- 策略：智能外呼为主 + 人工催收为辅
- 分配：按还款意愿评分决定催收强度
- 目标回收率：85%+

**M2阶段（逾期31-60天）：**
- 策略：人工催收为主，每日至少触达1次
- 补充：启动失联修复（大数据查找备用联系方式）
- 目标回收率：60%+

**M3+阶段（逾期60天+）：**
- 策略：委外催收或法催评估
- 核销标准：M6且回收可能性<5%

**归因分析：**
- 迁徙风险贡献：~40%
- 结构性风险贡献：~60%

📄 来源：{', '.join(sources)}""",

        "年轻": f"""关于年轻客群（22-28岁）的策略优化：

**核心痛点：** 征信数据薄，传统评分模型区分度低

**解决方案：**
1. 接入学历直连验证（学信网API）
2. 重设客户分层：本科及以上学历客户单独分群
3. 退退低区分度规则（如"工作年限<2年即拒绝"）

**实施效果：**
| 指标 | 变化 |
|------|------|
| 通过率 | +7pp ⬆️ |
| FPD7风险 | -21% ⬇️ |

**关键洞察：** 替代性数据（学历、社保）对薄信用人群有显著区分力，可以弥补征信不足的问题。

📄 来源：{', '.join(sources)}""",
    }

    for key, answer in answers_map.items():
        if key in query:
            return answer

    return f"""根据检索到的相关文档，以下是我的回答：

{context_docs[0]['content'].strip()}

📄 来源：{', '.join(sources)}

---
*注：以上回答基于知识库中的策略文档自动生成。如需更详细信息，请指定具体策略名称。*"""


# ============================================================
# Streamlit 前端界面
# ============================================================

def main():
    st.set_page_config(
        page_title="信贷策略知识库 QA System",
        page_icon="🏦",
        layout="wide"
    )

    # 侧边栏
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=80)
        st.title("📚 策略知识库")
        st.markdown("---")
        st.markdown("**已录入文档：**")
        for doc in SAMPLE_STRATEGY_DOCS:
            st.markdown(f"- 📄 {doc['title']}")
        st.markdown("---")
        st.markdown("**系统信息：**")
        st.markdown(f"- 文档数量：{len(SAMPLE_STRATEGY_DOCS)}")
        st.markdown("- 向量模型：text-embedding-v2")
        st.markdown("- LLM：通义千问-Max")
        st.markdown("- 向量数据库：ChromaDB")
        st.markdown("---")
        st.markdown("**技术架构：**")
        st.code("""
PDF文档 → 分块(Chunk)
    → Embedding向量化
    → 存入ChromaDB

用户提问 → Query Embedding
    → 向量相似度检索(Top-3)
    → 拼接Context + Prompt
    → LLM生成回答
        """, language="text")

    # 主界面
    st.title("🤖 信贷策略智能问答系统")
    st.markdown("*基于RAG（检索增强生成）的策略文档问答 — 输入自然语言问题，AI从知识库检索并生成答案*")
    st.markdown("---")

    # 示例问题
    st.markdown("**💡 试试这些问题：**")
    col1, col2, col3 = st.columns(3)

    example_questions = [
        "短信渠道的准入条件是什么？",
        "定价通过率是怎么从4.5%提升到9%的？",
        "自建模型的KS和AUC是多少？",
        "催收策略怎么分层的？",
        "年轻客群怎么做到放量又控险？",
    ]

    selected_q = None
    with col1:
        if st.button("📋 短信渠道准入条件"):
            selected_q = example_questions[0]
        if st.button("💰 定价策略优化效果"):
            selected_q = example_questions[1]
    with col2:
        if st.button("📊 自建模型性能指标"):
            selected_q = example_questions[2]
        if st.button("📞 催收分层策略"):
            selected_q = example_questions[3]
    with col3:
        if st.button("👤 年轻客群放量方案"):
            selected_q = example_questions[4]

    st.markdown("---")

    # 输入框
    query = st.text_input(
        "🔍 输入你的问题：",
        value=selected_q if selected_q else "",
        placeholder="例如：短信渠道的准入条件是什么？各档位的额度和定价如何？"
    )

    if query:
        with st.spinner("🔍 正在从知识库检索相关文档..."):
            import time
            time.sleep(0.5)  # 模拟检索延迟
            retrieved_docs = simple_keyword_search(query, SAMPLE_STRATEGY_DOCS)

        # 显示检索结果
        with st.expander("📑 检索到的相关文档（点击展开）", expanded=False):
            for i, doc in enumerate(retrieved_docs):
                st.markdown(f"**{i+1}. {doc['title']}** (相关度: {'⭐' * (3-i)})")
                st.text(doc["content"][:200] + "...")
                st.markdown("---")

        # 生成回答
        with st.spinner("🤖 AI正在生成回答..."):
            time.sleep(0.8)  # 模拟LLM生成延迟
            answer = generate_answer(query, retrieved_docs)

        st.markdown("### 💬 AI 回答：")
        st.markdown(answer)

        # 反馈
        st.markdown("---")
        col_a, col_b, _ = st.columns([1, 1, 4])
        with col_a:
            st.button("👍 有帮助")
        with col_b:
            st.button("👎 不准确")

    # 底部技术说明
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 12px;'>
    Built with LangChain + ChromaDB + Qwen-Max | RAG Architecture | Demo Version
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
