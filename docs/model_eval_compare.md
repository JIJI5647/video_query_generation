# Caption 模型 → query 生成+评估 横向对比

实验: `output/eval_pilot_p7_rolecot/`。固定下游(Gemini emotion-event+query 生成 → Qwen3-Omni-Instruct p7_rolecot 按维度验证),只换 caption 模型。目标: 找出哪种 caption 最能喂出高质量 query。

## 结果表(真实文件计算)

| 模型 | 状态 | query 数 | 接受 | 接受率 | emotion_state | evidence_cue | explicit_event |
|---|---|---|---|---|---|---|---|
| af3_vl | ✅ | 153 | 21 | **13.7%** | 8 | 132 | 13 |
| timechat | ✅ | 238 | 76 | **31.9%** | 82 | 144 | 12 |
| avocado | 待重跑 | - | - | - | - | - | - |
| qwen3_omni | 跑中 | - | - | - | - | - | - |
| qwen_audio_vl | 未开始 | - | - | - | - | - | - |

## 关键差异 (af3_vl vs timechat)

- **timechat 接受率是 af3_vl 的 2.3 倍**(31.9% vs 13.7%),且生成的 query 更多(238 vs 153)。
- **query 类型结构不同**: timechat 产出大量 `emotion_state` query(82 vs af3_vl 仅 8)。af3_vl 几乎全是 `evidence_cue`(132/153)。
  - 解读: timechat 的 caption 带时间戳且描述面部/肢体表情("contorted face""wide eyes"),让下游更容易写出可 grounding 的情绪状态 query;af3_vl 情绪信号偏音频韵律、视觉中性,下游只能写"证据线索"型 query,且更多被 answerability 判掉。

## 案例

**af3_vl** — accepted: "When does the child cling to the adult with wide eyes and an open mouth..."
discarded: "When does the young boy appear sad while holding a knife?" ← *answerability: 画面无 sad 的视听证据*(af3_vl 典型失败: 情绪在音频/推断,视觉中性)

**timechat** — accepted: "When does the woman appear disappointed while looking down with a weary expression?"
discarded: "When does the woman appear surprised while gasping with her eyes closed?" ← *answerability: 证据不足*

两模型**共同瓶颈**都是 **answerability**(证据不够)和 **emotion_relevance**(query 写成了宽泛心理状态/物理动作,非 8 类目标情绪)。

## 待补

- avocado(重跑中,带 runaway 护栏)、qwen3_omni、qwen_audio_vl 完成后补入。
- 三维度单独通过率(verification_rounds.jsonl 字段结构待确认,当前先用 final decision + failure_reason 归因)。
- ⚠️ 无独立 grounding 评估(IoU/mIoU)——timestamp 质量当前测不出,是阶段3查论文重点。

_更新: 2026-07-08, af3_vl + timechat 完成后_

## 更新: qwen3_omni 完成 (3/5)

| 模型 | query 数 | 接受 | 接受率 | emotion_state | evidence_cue | explicit_event | 主要丢弃维度 |
|---|---|---|---|---|---|---|---|
| af3_vl | 153 | 21 | 13.7% | 8 | 132 | 13 | answerability |
| timechat | 238 | 76 | **31.9%** | 82 | 144 | 12 | answerability+relevance |
| qwen3_omni | **387** | 111 | 28.7% | **187** | 171 | 29 | **emotion_relevance(124)** |

**qwen3_omni 观察**:
- **产出 query 最多**(387,是 af3_vl 的 2.5 倍)——结构化 captioner 提取的 event 最多(236),下游写出的 query 也最多。
- 接受率 28.7%,接近 timechat,远高于 af3_vl。
- **主要丢弃原因是 emotion_relevance(124条)**,和另两个模型不同(它们是 answerability 为主)。原因:qwen3_omni 的结构化 `visual_expression` caption 描述大量动作/手势,下游容易写成"行为型" query(如 "gesture with her hand""have her mouth open"),被判"描述行为非情绪"。
- accepted 例:"When does the woman with blonde curly hair appear happy while speaking..."；discarded 例:"...gesture with her hand" ← emotion_relevance: 聚焦行为非情绪。

**三模型小结**:timechat 接受率最高(32%),qwen3_omni 产量最高(387)但被"行为化 query"拖累接受率,af3_vl 最弱(14%,视觉情绪信号不足)。瓶颈维度因模型而异:af3_vl/timechat 卡 answerability,qwen3_omni 卡 emotion_relevance。

_更新: 2026-07-08, qwen3_omni 完成后_

## 更新: qwen_audio_vl 完成 (4/5)

| 模型 | query 数 | 接受 | 接受率 | emotion_state | evidence_cue | explicit_event | 主要丢弃维度 |
|---|---|---|---|---|---|---|---|
| af3_vl | 153 | 21 | 13.7% | 8 | 132 | 13 | answerability |
| timechat | 238 | 76 | **31.9%** | 82 | 144 | 12 | answerability+relevance |
| qwen3_omni | **387** | 111 | 28.7% | 187 | 171 | 29 | emotion_relevance |
| qwen_audio_vl | 170 | 50 | 29.4% | 58 | 79 | **33** | emotion_relevance(66) |

**qwen_audio_vl 观察**:
- 接受率 29.4%(第二梯队,与 qwen3_omni 相当)。产出 explicit_event 最多(33)。
- accepted 例明显靠**音频韵律**映射情绪:"seem frustrated, using a dry sarcastic tone""appear angry, forceful confrontational"——Qwen3-Omni-Captioner 音频半边给的情绪韵律信号强,能对上目标情绪。
- 主要丢弃 emotion_relevance(66):query 写成"describes speech style"(说话方式)而非明确目标情绪;其次 answerability(27)。

**四模型阶段小结**:接受率 timechat(32%) ≈ qwen_audio_vl(29%) ≈ qwen3_omni(29%) >> af3_vl(14%)。共性:视觉/音频情绪信号强的 caption(timechat 表情+时间戳、qwen_audio_vl 音频韵律、qwen3_omni 结构化)都能到 ~30%;af3_vl 视觉中性最弱。瓶颈维度:af3_vl/timechat 卡 answerability(证据不足),qwen3_omni/qwen_audio_vl 卡 emotion_relevance(query 写成行为/说话方式非情绪)。

_更新: 2026-07-08, qwen_audio_vl 完成后。剩 avocado 重跑。_

## ★ 最终: 全 5 模型完成

| 模型 | query数 | 接受 | 接受率 | emotion_state | evidence_cue | explicit_event | events | 主丢弃维度 |
|---|---|---|---|---|---|---|---|---|
| af3_vl | 153 | 21 | 13.7% | 8 | 132 | 13 | 130 | emotion_relevance |
| **timechat** | 238 | 76 | **31.9%** | 82 | 144 | 12 | 227 | relevance+answerability |
| qwen3_omni | 387 | 111 | 28.7% | 187 | 171 | 29 | 236 | relevance+answerability |
| qwen_audio_vl | 170 | 50 | 29.4% | 58 | 79 | 33 | 88 | emotion_relevance |
| avocado | **445** | 85 | 19.1% | 107 | 303 | 35 | **342** | emotion_relevance(207) |

**排名**:
- 接受率: timechat(32%) > qwen_audio_vl(29%) ≈ qwen3_omni(29%) > avocado(19%) > af3_vl(14%)
- query 产量: avocado(445) > qwen3_omni(387) > timechat(238) > qwen_audio_vl(170) > af3_vl(153)

**avocado 观察**:
- 产量最高(445 query, 342 event)——冗长 caption 提取的 event 最多。护栏成功防住 runaway(最大 89 event/视频,<上限192,首次跑的180-event失控未再现)。
- 但接受率仅 19.1%,被 emotion_relevance 大量判掉(207条):query 写成"说话方式/语气"(如 "speak quickly and dismissively""direct challenging tone"),非目标情绪。accepted 例是明确情绪的:"appear to be in extreme fear""screaming in..."。

**总结论**:
1. **timechat 综合最优**(接受率 32% 最高,产量适中 238)——带时间戳+面部表情的 caption 最契合下游 query 生成。
2. **产量≠质量**:avocado/qwen3_omni 产量最高但接受率被"行为化/语气化 query"拖累。
3. **瓶颈是 verifier 的两个维度**:emotion_relevance(query 描述行为而非情绪)和 answerability(视觉证据不足)。所有模型都卡在这两个上。
4. **caption 情绪信号强弱决定上限**:视觉表情(timechat)、音频韵律(qwen_audio_vl)、结构化(qwen3_omni)都能到~30%;af3_vl 视觉中性最弱(14%)。
5. ⚠️ **无 grounding 评估**(IoU/mIoU)——timestamp 质量测不出,是下一步重点。

_全 5 模型完成: 2026-07-09_

## ★★ 修正: 用真实 pass 率(去掉 max_accepted=8 cap 的 confound)

之前的"接受率"被每视频 max_accepted=8 上限系统性压低(高产视频的好 query 被砍)。用真实验证通过率(decision=pass,忽略 cap)重排:

| 模型 | query数 | 旧capped接受率 | **真实pass率** | 排名 |
|---|---|---|---|---|
| qwen_audio_vl | 170 | 29% | **45%** | **1** |
| qwen3_omni | 387 | 28% | 43% | 2 |
| timechat | 238 | 32% | 41% | 3 |
| avocado | 445 | 19% | 37% | 4 |
| af3_vl | 153 | 13% | 13% | 5 |

**结论变化**:
- **timechat 不再是第一**(旧 capped 排名把它排第1是假象);**qwen_audio_vl 真实 pass 率最高(45%)**。
- avocado 受 cap 影响最大(19%→37%,几乎翻倍)——它产量最高(445),被砍最多。
- af3_vl 的 pass率=接受率(13%),说明它从没有视频产出>8个好query,印证其caption情绪信号最弱。
- **写论文应用 pass 率而非 capped 接受率**,否则排名会错。

_修正: 2026-07-09, 用真实 pass 率_
