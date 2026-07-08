# ERRATA — t7_rag_gate_probe.json（2026-07-08，append-only 注记）

本文件是 `_repro/t7_rag_gate_probe.json` 的勘误 sibling。按 append-only 纪律，原结果 JSON
保持原样不改；本注记纠正其中一个**误导性的边界声明**。

## 勘误内容

原 JSON 的 `boundary` 字段声称 **"answers never injected"（答案从未被注入）**。该表述在
字段层面为真（gold answer *字段*确实没有直接注入），但在信息层面为假：

- T7 的 KB 以 heysquad 每条目自己的 `context` 段落为 value，而该段落**内含本条目的 gold
  answer 字符串**。泄露量化（`scripts/t7_leakage_audit.py`）：
  `answer_in_own_KB_passage ≈ 1.0`，`answer_in_injected_topk ≈ inject_k 准确率`。
- 因此 T7 的 headline 增益（base 0.283 → inject_k 0.767；H0 oracle−base = +0.517）
  是 **answer-lookup（答案抄读），不是知识利用**。
- 另一处边界违规：检索 query 用了 gold **question 文本**（部署侧不具备；应为音频或自产 ASR）。

## 取代关系

T8（`t8_clean_rag_rerun.json`，boundary-clean：query=自产 ASR、注入段落 gold 清洗、
残留 1.7%）重跑后 **clean_H0 = −0.066，CI [−0.167, 0.033]，null** —— T7 的"RAG 大幅有效"
结论不成立，其数值仅可作为"泄露上界/answer-lookup 幅度"引用（raw−scrub = +0.516）。

**引用规则**：任何后续文档引用 T7 数字时必须连带本 errata；T7 的 `boundary` 字段不可
作为边界干净的依据。

记录：wiki `2026-07-08-three-anchors-critical-audit.md` §3-刺1、§4-#1。
