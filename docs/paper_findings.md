# 阶段3 论文调研: 提升 query 质量 / verifier / grounding 评估

基于 5 模型评估发现的三个瓶颈,检索 2024-2026 论文找候选方法。

## 发现的瓶颈(来自 docs/model_eval_compare.md)
1. **answerability**: query 常因"视觉证据不足"被判掉(af3_vl/timechat 主因)。
2. **emotion_relevance**: query 写成"行为/语气/说话方式"而非目标情绪(qwen3_omni/qwen_audio_vl/avocado 主因,avocado 达 207 条)。
3. **无 grounding 评估**: 当前无 IoU/mIoU/R@n,timestamp 质量测不出。

## 候选方法(按瓶颈映射)

### A. 针对 answerability(最直接可用)
- **YTCommentQA: Video Question Answerability in Instructional Videos** (arXiv 2401.17343) — 专做"视频问题是否可答"的判定与数据集。可借鉴其 answerability 判据来强化我们的 verifier answerability 维度。
- **Can Video LLMs Refuse to Answer? Alignment for Answerability in Video LLMs** (arXiv 2507.04976) — 给 video LLM 做"answerability 对齐",让模型在证据不足时拒答。**直接对应我们 answerability verifier 的核心问题**——可把其对齐思路用于我们的 verify 阶段。
- **Detecting (Un)answerability in LLMs with Linear Directions** (arXiv 2509.22449) — 用激活空间线性方向检测可答性,轻量。

### B. 针对 query 质量评估(引入客观指标)
- **EVQAScore: A Fine-grained Metric for VideoQA Data Quality** (arXiv 2411.06908) — 细粒度 VideoQA 数据质量指标。**可作为接受率之外的客观 query 质量分**,补上我们"无质量金标准"的缺口。
- **FingER: Content-Aware Fine-grained Evaluation with Reasoning** (arXiv 2504.10358) — 把评估拆成实体级 Q&A 打分。可借鉴其"分解式评分"做 per-query 质量分。

### C. 针对 query 生成自精炼(减少行为化/低质 query)
- **Self-Evolving Visual Questioner** (arXiv 2606.13929) — 视觉提问器自演化,自生成-筛选-精炼 query。**契合我们 verify⇄rewrite 循环**,可升级 rewrite 策略。
- **Verifier-guided self-refinement**(通用,多篇)— 生成多候选 + verifier 反馈打分选优,报告有 +6pt 提升。

### D. 针对 grounding 评估缺口(引入 IoU/mIoU)
- **TRACE: Temporal Grounding Video LLM via Causal Event Modeling** (ICLR 2025) — 用因果事件建模做时间定位的 video LLM,给出 grounding + 指标框架。
- **Enrich and Detect: Video Temporal Grounding with Multimodal LLMs** (arXiv 2510.17023) — MLLM 做 temporal grounding。**可借其评估协议(IoU/mIoU/R@n)给我们的 query 补 grounding 质量评估**。

## 给 leader 的建议(3 个候选实验方向,待用户拍板)
1. **[低成本/高价值] 引入 grounding 评估**: 参考 TRACE/Enrich-and-Detect 的 IoU/mIoU/R@n 协议,给已生成的 query+time_range 加一层 grounding 质量度量。当前完全缺失,补上后 timestamp 质量才可测——这是"更准 timestamp"目标的前提。
2. **[中成本] 强化 verifier answerability**: 借鉴 "Can Video LLMs Refuse to Answer" 的 answerability 对齐 / YTCommentQA 判据,改进 answerability 维度的 prompt 或加判据,降低"证据不足"误判。
3. **[中成本] 升级 rewrite 为自精炼**: 参考 Self-Evolving Visual Questioner,把当前"revise 直接改写"升级为"多候选+verifier打分选优",专治 emotion_relevance 的行为化 query。

_论文调研: 2026-07-09。来源见各 arXiv ID。_
