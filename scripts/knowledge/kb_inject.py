"""kb_inject — standardized ways to DELIVER retrieved knowledge to the frozen omni.

Delivery FORM is the one clean, training-free lever the knowledge track found: on SQuAD-zh,
single-turn injection 0.175 -> two-turn tool-style delivery 0.35 (+0.175, ~doubles adoption)
(``t10_proto_agentic_2turn.py``). By contrast, raw single-turn RAG gives no reasoning gain once
answer-scrubbed (T8 clean_H0 = -0.066). So the injection API foregrounds delivery form as a first
-class, selectable parameter — the object a Stage-2 training-free RL policy would optimize over.

Templates are the ones actually used in the runs (``t7_rag_gate_probe.REF`` / ``ANS_INSTR``,
``t10`` two-turn), kept verbatim so the persisted KB stays comparable to prior results.
"""
from __future__ import annotations

REF = "参考资料(检索得到,可能相关也可能无关):\n{docs}\n\n"
ANS_INSTR = "请只用一个简短答案回答音频里的问题(只给答案,不要解释)。"
DELIVERY_FORMS = ("none", "single_turn", "two_turn_tool")


def format_docs(passages: list[str]) -> str:
    return "\n---\n".join(passages)


def single_turn_prompt(passages: list[str], instruction: str = ANS_INSTR) -> str:
    """Inject all passages inline before the instruction (t7 inject_k form)."""
    if not passages:
        return instruction
    return REF.format(docs=format_docs(passages)) + instruction


def two_turn_messages(passages: list[str], instruction: str = ANS_INSTR) -> list[dict]:
    """Agentic tool-style delivery (t10) — knowledge arrives as a 'tool result' turn, then the ask.

    Returns a messages list (text parts only; the caller adds the audio part to the user turn), which
    ~doubled adoption vs single-turn in T10. This is the recommended delivery form.
    """
    tool_result = REF.format(docs=format_docs(passages)) if passages else ""
    return [
        {"role": "user", "content": "我需要回答音频里的问题,先检索相关资料。"},
        {"role": "assistant", "content": f"[tool: retrieve] 检索到以下资料:\n{tool_result}"},
        {"role": "user", "content": instruction},
    ]


def deliver(passages: list[str], instruction: str = ANS_INSTR, form: str = "two_turn_tool"):
    """Dispatch on delivery FORM (the training-free lever). Returns str (single) or messages (two_turn)."""
    if form == "none":
        return instruction
    if form == "single_turn":
        return single_turn_prompt(passages, instruction)
    if form == "two_turn_tool":
        return two_turn_messages(passages, instruction)
    raise ValueError(f"unknown delivery form {form!r}; expected one of {DELIVERY_FORMS}")
