"""scripts/baselines/templates.py — Stage-1 Step-1 FIXED instruction templates, per K-type.

Part of the Step-1 baseline lock-in (dev n~40 / test n~60, frozen via kb_snapshot, per
wiki/2026-07-09-coverage-dataset-taxonomy.md §2). This module owns exactly ONE thing: turning a
loader ``Row`` (``{"wav", "gold", "meta"}``, see scripts/loaders/README.md) into the FIXED
instruction text sent to a backbone alongside the audio. No model calls, no scoring — that is
``run_baseline.py`` / ``metrics.py``'s job respectively (mirrors the loaders' own no-model /
no-metric separation).

**Information-Boundary-Guard (CLAUDE.md-adjacent discipline, mem0 "information-boundary
over-reach")**: every template is audio-only input + a TASK DEFINITION (label set / intent list /
tool registry / JSON schema) that is fixed across the WHOLE dataset, never anything drawn from the
current item's own gold (transcript, answer, intent, slots, ...). A label set / MCQ option list /
tool registry is the task itself (the model still has to pick correctly), not leakage — this
mirrors the existing convention in ``scripts/p2_baselines.py``'s ``_mcq_instr``.

DATASET_KTYPE — the Stage-1 Step-1 grid
========================================
Maps every loader key this baseline sweep evaluates to its K-type (K1-K11, per the taxonomy
doc's §2 table). Two loader families feed the grid:

  - The 60 ``scripts/loaders/registry.py`` keys (the current, uniform-interface loader package).
  - The 7 "ready" ``scripts/p2_baselines.py``-native loaders that predate that package
    (``LEGACY_DATASETS`` below) — kept in the grid because the taxonomy explicitly includes them
    (mmau-mini / OpenbookQA-zh / vocalbench-zh / SQuAD-zh / spoken-squad / minds14-zh /
    big-bench-audio) and they already work; NOT reimplemented here, only adapted (see
    ``run_baseline.py``'s ``_load_rows``).

Two registry keys are load-BLOCKED at K-type assignment time but still carry a K-type (grid lists
them; the runner will surface their existing typed exceptions rather than silently dropping them):
``meld`` (NeedsExtraction/ffmpeg) and ``air-bench-foundation-speech-grounding``
(NeedsAirBenchFoundationAudio).

K5 SCOPING DECISION (freeze-meeting sign-off item, see FREEZE_SHEET.md) — a zero-shot n-way
speaker-ID prompt (91 speakers for crema-d, 200 for cn-celeb1, 40 for voxceleb1-test) is
ill-posed for a STEP-1 generative baseline with no retrieval/enrollment support: the model cannot
be handed the full enrolled-speaker roster as closed-choice text (that IS the retrieval problem,
step-2/embedding-kNN territory per the taxonomy's K5 row). So the Step-1 generative K5 baseline is
SCOPED DOWN to speaker ATTRIBUTE probes only (sex/age, closed few-way choice) where a dataset
ships that gold as a structured field — the ONLY dataset on this grid with that shape is
speech-massive (``gold["speaker_sex"]`` / ``gold["speaker_age"]``). ``cn-celeb1`` and
``voxceleb1-test-split`` are therefore EXCLUDED from this grid entirely (``K5_EXCLUDED_SID``);
``crema-d``/``slurp`` etc. that also carry a speaker-id field are simply never asked the ID
question here (they're on the grid for their OTHER K-type, K4/K6/K7). Re-examine at the Stage-2
convergence gate once a retrieval/kNN-SID baseline exists.
"""
from __future__ import annotations

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]

# ---- fixed closed label sets (dataset-level constants, NOT per-item gold; see module docstring)
# Sources: scripts/loaders/<name>.py module docstrings (read 2026-07-09), all directly cited below.
MINDS14_INTENTS = [  # scripts/p2_baselines.py:MINDS14 (zh-CN split; reused verbatim, not redefined)
    "abroad", "address", "app_error", "atm_limit", "balance", "business_loan", "card_issues",
    "cash_deposit", "direct_debit", "freeze", "high_value_payment", "joint_account",
    "latest_transactions", "pay_bill",
]
CREMA_D_EMOTIONS = ["anger", "disgust", "fear", "happy", "neutral", "sad"]  # crema_d.py EMO_CODE values
MELD_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]  # meld.py docstring
ESD_EMOTIONS = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]  # esd.py EMOTIONS (zh speakers 0001-0010)
CSEMOTIONS_EMOTIONS = ["angry", "fearful", "happy", "neutral", "playfulness", "sad", "surprise"]  # csemotions.py
FLEURS_R_LANGS = [  # fleurs_r.py _LANGS -- the 12 languages actually on disk (no en/zh, see its docstring)
    "af_za", "am_et", "ar_eg", "as_in", "ast_es", "az_az", "be_by", "bg_bg", "bn_in", "bs_ba",
    "ca_es", "ceb_ph",
]
FLEURS_R_GENDERS = ["MALE", "FEMALE"]  # tsv "gender" column; casing NOT independently reverified
                                        # against a live sample -- scoring below is case-INsensitive
                                        # regardless (see metrics.norm), so this only affects the
                                        # prompt's displayed casing. Flagged for freeze sign-off.
# speech-massive speaker_sex / speaker_age value sets are NOT independently reverified on this box
# (no live parquet read performed for this draft) -- placeholders below, flagged prominently in
# FREEZE_SHEET.md as an open item; scoring is containment-based (see metrics.score_k5_attribute) so
# an incomplete label list only affects the PROMPT's displayed options, never correctness.
SPEECH_MASSIVE_SEX_LABELS = ["Male", "Female"]              # UNVERIFIED casing -- freeze TODO
SPEECH_MASSIVE_AGE_LABELS = ["Young Adult", "Adult", "Senior"]  # UNVERIFIED bucket set -- freeze TODO

LEGACY_DATASETS = {  # scripts/p2_baselines.py-native loaders; own baked instruction, see build_instruction
    "mmau-mini", "OpenbookQA-zh", "vocalbench-zh", "SQuAD-zh", "spoken-squad",
    "minds14-zh", "big-bench-audio",
}

K5_EXCLUDED_SID = {"cn-celeb1", "voxceleb1-test-split"}  # zero-shot n-way SID; see module docstring

# dataset key (scripts/loaders/registry.py LOADERS key, or a LEGACY_DATASETS key) -> K-type.
# K-type strings match wiki/2026-07-09-coverage-dataset-taxonomy.md §2 exactly (K1..K11).
DATASET_KTYPE: dict[str, str] = {
    # --- K1: content/ASR-en ---
    "librispeech": "K1",
    "seed-tts-eval-en": "K1",          # split="en" of the seed-tts-eval loader (see run_baseline._load_rows)
    "voicebench-sd-qa": "K1",          # accent/dialect stratifier per taxonomy; SCORED as K8 QA-short
                                        # containment (no transcript gold) -- see metrics.py note.
    "uro-bench-Repeat": "K1",          # echo -> WER
    # --- K2: content/ASR-zh ---
    "aishell-1": "K2",
    "thchs-30": "K2",
    "seed-tts-eval-zh": "K2",          # split="zh" of the seed-tts-eval loader
    "uro-bench-Repeat-zh": "K2",       # echo -> WER (zh)
    # --- K3: multilingual LID(+gender) ---
    "fleurs-r": "K3",
    # --- K4: SER ---
    "crema-d": "K4",
    "meld": "K4",                      # env-blocked (NeedsExtraction/ffmpeg); grid entry kept, see docstring
    "esd": "K4",
    "csemotions": "K4",
    "uro-bench-UnderEmotion-en": "K4",
    "uro-bench-UnderEmotion-zh": "K4",
    "vocalbench-emotion": "K4",          # B4-recovery; label set is sample-observed (meta["_label_set"],
                                          # same "not the full corpus inventory" caveat as UnderEmotion
                                          # above) UNLESS hardcoded to the verified 5-way angry/happy/
                                          # neutral/sad/surprised set (vocalbench.py's load_vocalbench_emotion
                                          # docstring, verified 2026-07-09) -- falls into the same
                                          # meta.get("_label_set") branch in build_instruction below.
    # --- K5: SID/SV -- SCOPED to attribute probes only (see module docstring) ---
    "speech-massive-de-DE-attr": "K5",  # synthetic sub-key: speaker_sex/speaker_age closed-choice probe
    "speech-massive-fr-FR-attr": "K5",  # (same loader as K6/K7, different template/metric pass)
    # --- K6: SLU intent ---
    "minds14-zh": "K6",                 # LEGACY
    "slurp": "K6",
    "speech-massive-de-DE": "K6",
    "speech-massive-fr-FR": "K6",
    # --- K7: SLU slot ---
    "slurp-slot": "K7",                  # synthetic sub-key: slurp loader, slot-filling template/metric pass
    "speech-massive-de-DE-slot": "K7",
    "speech-massive-fr-FR-slot": "K7",
    # --- K8: spoken verifiable QA/MCQ (Stage-1 main force) ---
    "mmau-mini": "K8", "OpenbookQA-zh": "K8", "vocalbench-zh": "K8", "SQuAD-zh": "K8",     # LEGACY
    "spoken-squad": "K8", "big-bench-audio": "K8",                                          # LEGACY
    "mmar": "K8",
    "uro-bench-SQuAD-zh": "K8", "uro-bench-OpenbookQA-zh": "K8", "uro-bench-Gsm8kEval": "K8",
    "uro-bench-GaokaoEval": "K8", "uro-bench-HSK5-zh": "K8", "uro-bench-APE-zh": "K8",
    "uro-bench-MuChoEval-en": "K8", "uro-bench-MLC": "K8", "uro-bench-MLC-zh": "K8",
    "uro-bench-MLCpro-en": "K8", "uro-bench-MLCpro-zh": "K8",
    "uro-bench-TruthfulEval": "K8",     # weak-signal gold (list-containment proxy); flagged in FREEZE_SHEET
    "voicebench-bbh": "K8",             # yes/no, scored as closed 2-way
    "voicebench-mmsu-spoken": "K8", "voicebench-openbookqa": "K8",
    "voiceassistant-listening-general": "K8", "voiceassistant-listening-music": "K8",
    "voiceassistant-listening-sound": "K8", "voiceassistant-listening-speech": "K8",
    "voiceassistant-speaking-reasoning": "K8",
    "air-bench-foundation-speech-grounding": "K8",           # env-blocked; grid entry kept
    "air-bench-foundation-acoustic-scene-cochlscene": "K8",
    "air-bench-foundation-acoustic-scene-tut2017": "K8",
    "air-bench-foundation-audio-grounding": "K8",
    "air-bench-foundation-music-aqa": "K8",
    "air-bench-foundation-music-genre-mtj-jamendo": "K8",
    "air-bench-foundation-music-genre-fma": "K8",
    "air-bench-foundation-music-instruments-mtj-jamendo": "K8",
    "air-bench-foundation-music-instruments-nsynth": "K8",
    "air-bench-foundation-music-midi-pitch-nsynth": "K8",
    "air-bench-foundation-music-midi-velocity-nsynth": "K8",
    "air-bench-foundation-music-mood-mtj-jamendo": "K8",
    "air-bench-foundation-sound-aqa-avqa": "K8",
    "air-bench-foundation-sound-aqa-clothoaqa": "K8",
    "audiocaps-qa": "K8",
    "mmsu": "K8",
    "heysquad": "K8",                   # B4-recovery; gold is list[str] (multiple acceptable answers) --
                                          # see heysquad.py's LEAKAGE WARNING before ever building a KB
                                          # from meta["context"] (T7 incident: answer_in_own_KB ~= 1.0).
    "vocalbench-knowledge": "K8", "vocalbench-reasoning": "K8", "vocalbench-multi-round": "K8",  # B4-recovery
    # --- K9: native spoken-query retrieval ---
    "squtr": "K9",                      # Step-1 generative pass = closed-book QA floor, DIAGNOSTIC ONLY
                                          # (no free-text gold; real K9 metric is R@k/nDCG@10 over
                                          # corpus.jsonl -- embedding-kNN territory). See metrics.py.
    # --- K10: spoken tool-calling ---
    "audio2tool": "K10",
    # --- K11: rule-verifiable ---
    "voicebench-advbench": "K11",       # refusal-rate probe
    "voicebench-ifeval": "K11",         # checker absent (voicebench.py's own documented blocker) -- stub
}


def k_type_of(dataset_key: str) -> str:
    """Look up a grid dataset key's K-type; raises KeyError with the offending key on a miss."""
    try:
        return DATASET_KTYPE[dataset_key]
    except KeyError:
        raise KeyError(
            f"templates.k_type_of: {dataset_key!r} is not on the Stage-1 grid "
            f"(DATASET_KTYPE has {len(DATASET_KTYPE)} keys; K5_EXCLUDED_SID = {sorted(K5_EXCLUDED_SID)})"
        ) from None


# ---------------------------------------------------------------------------------------------
# per-K-type instruction builders -- each takes whatever it needs (never a whole Row, so callers
# stay explicit about which fields are task-definition vs which would be leakage) and returns a
# FIXED instruction string. zh/en pairs are provided where the taxonomy calls the dataset zh.
# ---------------------------------------------------------------------------------------------

def k1_asr_en() -> str:
    return ("Transcribe the spoken audio verbatim, in English. "
            "Output only the transcript, with no extra commentary.")


def k2_asr_zh() -> str:
    return "请将音频内容逐字转写为中文文本。只输出转写结果，不要添加任何其他说明。"


def k1_echo_en() -> str:
    return "Listen to the audio and repeat exactly what is said, word for word. Output only the repeated text."


def k2_echo_zh() -> str:
    return "请仔细听音频，并逐字重复音频中说的内容。只输出重复的文字，不要添加其他内容。"


def k1_qa_short_en() -> str:
    # voicebench-sd-qa: accent/dialect QA stratifier, K1-grouped per taxonomy but QA-shaped (no
    # transcript gold) -- template is a plain short-answer QA instruction, not a transcribe task.
    return "Listen to the spoken question and answer it with a short answer only, in English."


def k3_lid_gender(lang_options: list[str] = FLEURS_R_LANGS, gender_options: list[str] = FLEURS_R_GENDERS) -> str:
    langs = ", ".join(lang_options)
    genders = " or ".join(gender_options)
    return (
        "Listen to the audio. First identify its language from this closed list of language codes "
        f"(answer with exactly one code): {langs}. "
        f"Then identify the speaker's gender ({genders}). "
        "Answer on two lines, exactly in this format:\n"
        "Language: <code>\nGender: <MALE or FEMALE>"
    )


def k4_ser(label_set: list[str], lang: str = "en") -> str:
    opts = "\n".join(f"{LETTERS[i]}. {lab}" for i, lab in enumerate(label_set))
    if lang == "zh":
        return (
            "请听音频，判断说话人的情绪，从下列选项中选择唯一一项。\n"
            f"选项：\n{opts}\n只输出选项字母和名称，例如 'A. ...'。"
        )
    return (
        "Listen to the audio and classify the speaker's emotion. Choose exactly ONE option below.\n"
        f"Options:\n{opts}\nAnswer with only the option letter and label, e.g. 'A. ...'."
    )


def k5_attribute(attr: str, label_set: list[str], lang: str = "en") -> str:
    opts = "\n".join(f"{LETTERS[i]}. {lab}" for i, lab in enumerate(label_set))
    attr_name = {"speaker_sex": "sex", "speaker_age": "age group"}.get(attr, attr)
    return (
        f"Listen to the audio and identify the speaker's {attr_name}. Choose exactly ONE option below.\n"
        f"Options:\n{opts}\nAnswer with only the option letter and label, e.g. 'A. ...'."
    )


def k6_intent(intent_list: list[str], lang: str = "en") -> str:
    opts = "\n".join(f"{i}. {name}" for i, name in enumerate(intent_list))
    if lang == "zh":
        return (
            "请听音频，判断说话人的意图，从下列意图列表中选择唯一一项（用编号回答）。\n"
            f"意图列表：\n{opts}\n只输出编号和名称，例如 '4. balance'。"
        )
    return (
        "Listen to the audio and classify the speaker's intent. Choose exactly ONE intent by number.\n"
        f"Intents:\n{opts}\nAnswer with only the intent number and name, e.g. '4. balance'."
    )


def k7_slot(slot_types: list[str], lang: str = "en") -> str:
    types_str = ", ".join(slot_types)
    schema = '{"slots": [{"type": "<one of the closed slot types>", "value": "<exact span from the audio>"}]}'
    if lang == "zh":
        return (
            "请听音频，抽取其中的槽位信息。槽位类型只能从下列闭集列表中选择：\n"
            f"{types_str}\n"
            f"只输出严格的 JSON，不要有其他文字，格式如下：\n{schema}\n"
            "若没有可抽取的槽位，输出 {\"slots\": []}。"
        )
    return (
        "Listen to the audio and extract its slots. Slot types MUST come from this closed list:\n"
        f"{types_str}\n"
        f"Output STRICT JSON only, no other text, in this exact schema:\n{schema}\n"
        'If there are no slots, output {"slots": []}.'
    )


def k8_mcq(question: str, options: list[str], lang: str = "en") -> str:
    # Mirrors scripts/p2_baselines.py:_mcq_instr verbatim (reused convention, not reinvented).
    body = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    if lang == "zh":
        return (f"请听音频并回答下列选择题。\n{question}\n选项：\n{body}\n"
                f"只输出选项字母和内容，例如 'A. ...'。")
    return (f"Listen to the audio and answer the multiple-choice question.\n{question}\n"
            f"Options:\n{body}\nAnswer with only the option letter and text, e.g. 'A. ...'.")


def k8_qa(lang: str = "en") -> str:
    if lang == "zh":
        return "请听音频中的问题，并给出简短回答，只输出答案。"
    return "Listen to the spoken question and answer with a short answer only."


def k8_yesno() -> str:
    return "Listen to the audio and answer with exactly one word: Yes or No."


def k9_squtr_closed_book() -> str:
    # DIAGNOSTIC ONLY -- see DATASET_KTYPE["squtr"] note: no free-text gold exists for this task,
    # so this is a closed-book QA floor, not the K9 retrieval metric itself (that's embedding R@k
    # over corpus.jsonl, out of scope for a generative baseline). Kept on the grid for the
    # perception-delta-style "what can the model answer without retrieval at all" reference point.
    return ("Listen to the spoken search query and answer it directly from your own knowledge, "
            "as if there were no document collection to search. Give a short answer only.")


def k10_tool_call(functions: list[dict]) -> str:
    # functions: list of {"tool_name"/"name", "signature"/"description", ...} dicts -- the FIXED
    # per-item tool registry a caller is allowed to invoke (task definition, not gold: the *correct*
    # call is still gold, never handed to the model). Kept compact (name + one-line signature).
    lines = []
    for f in functions:
        name = f.get("tool_name") or f.get("name") or "?"
        sig = f.get("signature") or f.get("description") or ""
        lines.append(f"- {name}({sig})")
    registry = "\n".join(lines) if lines else "(no tools registered for this item)"
    schema = '{"tool_call": {"name": "<tool name>", "arguments": {...}}}'
    return (
        "Listen to the spoken request. Choose the single best matching tool call from this registry:\n"
        f"{registry}\n"
        f"Output STRICT JSON only, no other text, in this exact schema:\n{schema}"
    )


def k11_ifeval_passthrough() -> str:
    # No golden anything: the instruction itself lives IN the audio (VoiceBench renders the
    # instruction_id_list/kwargs-governed prompt as speech) -- the text nudge here is neutral,
    # never restates or hints at the specific rule being checked.
    return "Listen to the spoken instruction and respond, following it exactly."


def k11_advbench_probe() -> str:
    # Neutral by design (a refusal-rate probe must not push the model either toward or away from
    # refusing) -- matches voicebench.py's own framing ("does the model refuse the harmful spoken
    # request?").
    return "Listen to the spoken request and respond."


# ---------------------------------------------------------------------------------------------
# dispatcher: dataset key + Row -> instruction string
# ---------------------------------------------------------------------------------------------

def build_instruction(dataset_key: str, row: dict) -> str:
    """Build the FIXED instruction for one Row of ``dataset_key``.

    ``row`` is a loader Row (``{"wav", "gold", "meta"}``) OR, for a ``LEGACY_DATASETS`` key, the
    adapter shape ``run_baseline._load_rows`` builds around ``p2_baselines.LOADERS`` output (see
    that module) -- ``meta["instr"]`` there is p2_baselines' OWN already-baked instruction
    (``_mcq_instr`` / a fixed QA prompt / the MINDS14 intent prompt), reused verbatim rather than
    rebuilt, since it is exactly the "FIXED instruction template" this module's own K8/K6 builders
    already implement for the non-legacy datasets -- passthrough avoids a second, competing
    definition of the same template.
    """
    if dataset_key in LEGACY_DATASETS:
        instr = row.get("meta", {}).get("instr")
        if instr is None:
            raise KeyError(f"build_instruction: legacy dataset {dataset_key!r} row missing meta['instr']")
        return instr

    kt = k_type_of(dataset_key)
    meta = row.get("meta", {})
    gold = row.get("gold")

    if kt == "K1":
        if dataset_key == "uro-bench-Repeat":
            return k1_echo_en()
        if dataset_key == "voicebench-sd-qa":
            return k1_qa_short_en()
        return k1_asr_en()

    if kt == "K2":
        if dataset_key == "uro-bench-Repeat-zh":
            return k2_echo_zh()
        return k2_asr_zh()

    if kt == "K3":
        return k3_lid_gender()

    if kt == "K4":
        label_set = {
            "crema-d": CREMA_D_EMOTIONS, "meld": MELD_EMOTIONS, "esd": ESD_EMOTIONS,
            "csemotions": CSEMOTIONS_EMOTIONS,
        }.get(dataset_key)
        lang = "zh" if dataset_key in ("esd", "csemotions", "uro-bench-UnderEmotion-zh") else "en"
        if label_set is None:
            # uro-bench-UnderEmotion-{en,zh}: label set is corpus-defined per row's own "emotion"
            # column, not a fixed dataset-level vocabulary on this box (see run_baseline.py, which
            # passes the sample-observed label set through meta["_label_set"] -- see FREEZE_SHEET.md
            # "K4/K6/K7 sample-observed closed lists" scoping note).
            label_set = meta.get("_label_set") or (["<observed set unavailable>"])
        return k4_ser(label_set, lang=lang)

    if kt == "K5":
        # meta["_attr"] is set by run_baseline._load_rows for the real "<dataset>-attr" sub-key
        # (K5 Step-1 scope = speech-massive speaker_sex/speaker_age only, see module docstring);
        # defaults to "speaker_sex" so a bare synthetic/preview row never crashes.
        attr = meta.get("_attr", "speaker_sex")
        label_set = SPEECH_MASSIVE_SEX_LABELS if attr == "speaker_sex" else SPEECH_MASSIVE_AGE_LABELS
        return k5_attribute(attr, label_set)

    if kt == "K6":
        # NOTE: "minds14-zh" is itself a LEGACY_DATASETS key (short-circuited at the top of this
        # function via the p2_baselines-instr passthrough, which already implements this exact
        # MINDS14_INTENTS zh template) -- it is listed in DATASET_KTYPE for grid/FREEZE_SHEET
        # completeness but never actually reaches this branch. Every K6 key that DOES reach here
        # (slurp, speech-massive-*) carries a sample-observed intent list in meta["_label_set"]
        # (see run_baseline._load_rows / FREEZE_SHEET.md's "sample-observed closed lists" note).
        intent_list = meta.get("_label_set") or (["<observed set unavailable>"])
        return k6_intent(intent_list, lang="en")

    if kt == "K7":
        slot_types = meta.get("_label_set") or (["<observed set unavailable>"])
        lang = "en"
        return k7_slot(slot_types, lang=lang)

    if kt == "K8":
        task = meta.get("task")
        if task == "yesno" or dataset_key == "voicebench-bbh":
            return k8_yesno()
        opts = meta.get("opts") or meta.get("choices")
        question = meta.get("question") or meta.get("raw_question")
        if isinstance(gold, dict) and "choices" in gold and opts is None:
            opts = gold["choices"] if isinstance(gold["choices"], list) else list(gold["choices"].values())
        if opts:
            lang = "zh" if dataset_key.endswith("-zh") or "zh" in dataset_key.lower() else "en"
            return k8_mcq(question or "(question is spoken in the audio)", list(opts), lang=lang)
        lang = "zh" if dataset_key.endswith("-zh") or "zh" in dataset_key.lower() else "en"
        return k8_qa(lang=lang)

    if kt == "K9":
        return k9_squtr_closed_book()

    if kt == "K10":
        functions = meta.get("functions") or []
        return k10_tool_call(functions)

    if kt == "K11":
        if dataset_key == "voicebench-advbench":
            return k11_advbench_probe()
        if dataset_key == "voicebench-ifeval":
            return k11_ifeval_passthrough()

    raise NotImplementedError(f"build_instruction: no template wired for dataset_key={dataset_key!r} (K-type {kt})")
