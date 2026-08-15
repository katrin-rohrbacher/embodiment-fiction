#!/usr/bin/env python
"""Run the LLM benchmark over the annotated set.

Prompts each model under the crossed conditions (few-shot x context) and writes
one CSV per (model, condition) to cfg.BENCHMARK_DIR. Checkpointed every 100
sentences and resumed on restart, so it is safe to run detached.

    python scripts/run_llm_benchmark.py qwen3-32b-local
    python scripts/run_llm_benchmark.py gpt-5-mini --conditions zeroshot,fewshot

Hosted models read their key from the environment (or ~/.api_keys.env); local
models expect an Ollama server on localhost:11434. One results directory per
prompt version: resume treats any row already present as done, so pointing a
changed prompt at an old directory would mix versions in one file.
"""
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

import sys

print("=== runner start:", time.strftime("%Y-%m-%d %H:%M:%S"), "===", flush=True)

# Hosted models need their key in the environment. A detached `nohup` shell has no
# exports, so load ~/.api_keys.env the same way the notebook's key cell does.
KEYFILE = Path.home() / ".api_keys.env"
if KEYFILE.exists():
    for _line in KEYFILE.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
    print(f"loaded keys from {KEYFILE}", flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from embodiment import config as cfg  # noqa: E402

CONTEXT_CSV = cfg.CONTEXT_CSV
RESULTS_DIR = cfg.BENCHMARK_DIR
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED        = cfg.SEED

MODELS = {
    "llama3.3-70b-local": dict(model="llama3.3:70b",
                               base_url="http://localhost:11434/v1", key_env=None),
    "gpt-5-mini":         dict(model="gpt-5-mini", base_url=None,
                               key_env="OPENAI_API_KEY"),
    "gpt-5.1":            dict(model="gpt-5.1", base_url=None,
                               key_env="OPENAI_API_KEY"),
    "qwen3-32b-local":    dict(model="qwen3:32b",
                               base_url="http://localhost:11434/v1", key_env=None),
}
# usage:  python run_llama_benchmark.py [model ...] [--conditions a,b]
#   default model      : the local llama
#   default conditions : the two context conditions (see RUN_CONDITIONS below)
_argv = sys.argv[1:]
_conditions_arg = None
if "--conditions" in _argv:
    _i = _argv.index("--conditions")
    _conditions_arg = _argv[_i + 1].split(",")
    _argv = _argv[:_i] + _argv[_i + 2:]

ENABLED = _argv or ["llama3.3-70b-local"]
for _m in ENABLED:
    if _m not in MODELS:
        raise SystemExit(f"unknown model {_m!r}; known: {list(MODELS)}")

# Ollama serialises a 70B on one GPU, so extra concurrency there only adds queueing;
# a hosted endpoint is genuinely parallel.
MAX_WORKERS = 4 if any(MODELS[m]["base_url"] and "localhost" in MODELS[m]["base_url"]
                       for m in ENABLED) else 8

CONDITION_CONFIGS = {
    "zeroshot": {
        "few_shot": False,
        "use_context": False,
    },
    "fewshot": {
        "few_shot": True,
        "use_context": False,
    },
    "zeroshot_context": {
        "few_shot": False,
        "use_context": True,
    },
    "fewshot_context": {
        "few_shot": True,
        "use_context": True,
    },
}
RUN_CONDITIONS = _conditions_arg or [
    "zeroshot_context",
    "fewshot_context",
]
for _c in RUN_CONDITIONS:
    if _c not in CONDITION_CONFIGS:
        raise SystemExit(f"unknown condition {_c!r}; known: {list(CONDITION_CONFIGS)}")
# "zeroshot_context" uses SYSTEM_PROMPT without appending FEW_SHOT_BLOCK, but presents
# the target sentence inside its ~200-word passage rather than in isolation.
# "fewshot_context" uses the same contextual input format and appends FEW_SHOT_BLOCK
# to SYSTEM_PROMPT.
# Both context-sensitive conditions run only on sentences with verified,
# recoverable context.

ID2LABEL = cfg.ID2LABEL
LABELS = cfg.LABELS

# Full annotated set with context already attached (gold + pretext/posttext/ctx_source)
eval_df = pd.read_csv(CONTEXT_CSV)
assert eval_df["gold"].isin(LABELS).all(), "unexpected gold labels in context CSV"
print(f"eval set: {len(eval_df)} sentences, {eval_df['pretext'].notna().sum()} with context", flush=True)


# --- prompt (from notebook) ---
SYSTEM_PROMPT = """You are an expert annotator for a computational literary studies project on embodiment in narrative fiction. You classify individual sentences from novels into exactly one category.

# Task
Assign exactly one label to the target sentence. The label captures the DOMINANT mode of embodiment in the sentence.
If several modes are present, apply the decision procedure below, which encodes the precedence rules. Do not invent labels.
In addition to the label, provide a short, human-readable explanation for your decision.

# Labels
haptic_human_animal  Physical contact with a person or animal
haptic_material      Physical contact with objects, surfaces, or environment (incl. passive contact such as leaning or resting against)
character_move       Character changes position or location, no emphasis on physical contact
sight_human          Foregrounded act of looking at people or animals
sight_material       Foregrounded act of looking at objects or environment
character_feltness   A present character is affected by sensory or affective qualities of the environment (sound, smell, taste, temperature, atmosphere; NOT sight)
facial_expr          Explicit description of a character's facial expression
gesture              Communicative or expressive bodily movements (nodding, waving, pointing, recoiling, raising a hand)
no_rel               None of the above; also embodiment that is only recalled, anticipated, hypothetical, or conditional; explicit mental processes; plain dialogue; abstract statements
book_paratext        Non-narrative elements (titles, headings, front/back matter)
segmentation_error   Wrongly segmented sentence: a fragment cut out of a longer sentence (e.g. cut off at "Mrs."/"Ms."/"Dr.", at a colon, or otherwise incomplete). A sentence split at a semicolon is NOT a segmentation error.

# Decision procedure (apply in order, stop at the first match)
0. SEGMENTATION ERROR: FIRST decide whether the target sentence is COMPLETE, before interpreting what it describes. Judge this from how the sentence ENDS.
   Label segmentation_error if any of the following holds:
   - it ends in an abbreviation or title that normally continues: "Mr.", "Mrs.", "Ms.", "Dr.", "Rev.", "Hon.", "St.", "Messrs.", "Capt.", "No.";
   - it ends in a colon, or in a dangling function word ("and", "but", "of", "with", "that", "as");
   - it ends without terminal punctuation, or consists of stray characters, brackets, or markup;
   - it begins mid-clause, or is a fragment with no finite verb (if it is a heading, use book_paratext instead).
   A sentence split at a SEMICOLON is NOT a segmentation error.
   CRITICAL: a truncated sentence usually still reads as sensible narrative and appears to describe an action ("Gerald went down-stairs and found Mrs."). Decide by the ending, not by whether the sentence makes sense. If it is truncated, label segmentation_error and do NOT label the embodiment it appears to contain.
1. PARATEXT: if the sentence is a non-narrative element (title, heading, table of contents, notes) -> book_paratext
2. TEMPORALITY FILTER: embodiment only counts if it occurs in the actual narrated situation. If the embodied event is recalled from the more distant past, anticipated, planned, hypothetical, or conditional -> no_rel.
   - "When I was a child, my father used to hit me." -> no_rel (NOT haptic_human_animal)
   - "When we get to the next stop, he will come back." -> no_rel
   - "If you came here, you would understand." -> no_rel
3. HAPTICS: direct bodily contact, active (grasping, pressing, pulling) or passive (leaning, resting) -> haptic_human_animal or haptic_material. Haptics beats movement and sight ("She saw the painting and touched it." -> haptic_material). Tool use with implied bodily contact counts ("pulled a pair of oars" -> haptic_material).
4. MOVEMENT: change of position or location without emphasis on contact -> character_move. Static position without change of location ("She stood at the door.") is NOT movement; continue (usually no_rel).
5. SIGHT: annotate ONLY if the act of seeing is foregrounded from a character's standpoint ("she saw", "he looked at", "his glance roved") -> sight_human or sight_material.
   Do NOT annotate sight if: (a) seeing is part of movement or touch, (b) sight directly leads to an action (then label the action), (c) the sentence merely describes what happens and a character only implicitly observes it. Implicit observation is not embodiment.
6. FELTNESS: sensory or affective qualities of the environment (sound, smell, taste, temperature, texture, atmosphere, mood; sight excluded). Annotate character_feltness ONLY if a character is present and affected by the surroundings. Environmental description without a perceiving character ("The walls were rough and uneven.") -> no_rel.
7. EXPRESSION / GESTURE: explicit facial expression -> facial_expr. Communicative or expressive bodily movement -> gesture.
   NOTE: explicit mental processes (thinking, knowing, remembering, deciding) are NOT annotated as a separate category in this schema version -> no_rel.
8. DEFAULT: -> no_rel (includes plain dialogue without embodied content and abstract statements).

# Explanation
The explanation should provide a short, human-readable justification for the selected label.
Base your justification on the evidence used for the classification. Refer to specific phrases or lexical cues in the target sentence whenever possible. Do not merely restate the category definition.
If contextual information is required to interpret the target sentence correctly, you may refer to the surrounding passage in your explanation.
Write the explanation in English in one or two sentences.

# Output format
Respond with a single JSON object, no other text.
Return exactly this JSON format:

{"label": "<one label from the list>",
"explanation": "<short English justification>"}

"""

FEW_SHOT_BLOCK = """# Examples
"She grasped his hands." -> {"label": "haptic_human_animal", "explanation": "The sentence describes direct physical contact ('grasped his hands') between two people."}
"He pressed his palm against the wall." -> {"label": "haptic_material", "explanation": "The sentence describes physical contact between a person and an object."}
"Within half an hour they were setting this matter right by going up the river." -> {"label": "character_move", "explanation": "The sentence describes the character moving through space by 'going up the river'."}
"The man glanced up at Stubbs and then turned his attention to Wilson." -> {"label": "sight_human", "explanation": "The sentence describes a character ('the man') looking ('glanced') at another person (first 'Stubbs' and then 'Wilson')."}
"She looked at the painting." -> {"label": "sight_material", "explanation": "The sentence describes a character ('She') looking at an object ('the painting')."}
"But pacing along under the stars, with only the sound of his own footsteps for company, placed a very different light on the matter." -> {"label": "character_feltness", "explanation": "The sentence describes a character's internal experience or feeling being affected by the atmosphere created by the stars and the sound of his footsteps."}
"Mirabelle Marie gaped speechlessly at him." -> {"label": "facial_expr", "explanation": "The sentence describes a character's facial expression ('gaped speechlessly') in response to another character."}
"Poirot lifted his hand." -> {"label": "gesture", "explanation": "The sentence describes a character ('Poirot') performing a communicative or expressive bodily movement ('lifted his hand')."}
"Mr. Crisparkle could not determine this in his mind." -> {"label": "no_rel", "explanation": "The sentence describes a character's mental process ('could not determine this in his mind'), which is not considered as one of the defined categories."}
"Do not let me hear of your offending again." -> {"label": "no_rel", "explanation": "The sentence is a directive or statement that does not involve any embodiment or physical action."}
"When I was a child, my father used to hit me." -> {"label": "no_rel", "explanation": "The sentence describes a character's past experience with physical harm, which fails the temporality filter in the decision procedure."}
"She stood at the door." -> {"label": "no_rel", "explanation": "The sentence describes a character's static position or location, which does not involve movement. No emphasis on physical contact with the door."}
"The walls were rough and uneven." -> {"label": "no_rel", "explanation": "The sentence describes the characteristics of an object (the walls), withouth emphasis on physical interaction, the explicit act of looking at the object, or a sensory or affective effect on the character."}
"She saw the painting and touched it." -> {"label": "haptic_material", "explanation": "The sentence describes a character ('She') interacting with an object ('the painting') through touch. Both the act of seeing and touching are present, but the haptic interaction takes precedence over the visual observation according to the decision procedure."}
"CHAPTER XII." -> {"label": "book_paratext", "explanation": "The sentence is a non-narrative element (a chapter heading)."}
"Mrs." -> {"label": "segmentation_error", "explanation": "The sentence is wrongly segmented sentence, as it is incomplete and does not form a full sentence. It is likely cut off at the abbreviation 'Mrs.'."}
"The carriage stopped before the house of Mr." -> {"label": "segmentation_error", "explanation": "The sentence ends at the title 'Mr.', which normally continues with a name, so it has been cut off mid-sentence even though the preceding clause reads as complete."}
"She hurried along the passage and knocked at the door of Dr." -> {"label": "segmentation_error", "explanation": "Although the sentence appears to describe movement and contact, it terminates at the abbreviation 'Dr.' and is an incomplete fragment; completeness is decided before the embodied content."}"""

# MANUALLY UPDATE after changes to SYSTEM_PROMPT or FEW_SHOT_BLOCK!
PROMPT_METADATA = {
    "system_prompt_version": "1.2",
    "few_shot_block_version": "1.2",
    "date": "2026-08-09",
    "notes": "v1.1 added explanation output. v1.2 rewrites the segmentation_error "
             "definition and decision step 0 to test sentence completeness from the "
             "ending first, and adds two truncated-sentence few-shot examples; "
             "v1.1 scored 0.00 recall on segmentation_error (144 gold, 3 predicted).",
}


# Hashes allow exact prompt versions to be identified across runs, independently of the
# manually assigned version numbers.
def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def build_messages(sentence, few_shot, pretext=None, posttext=None):
    system = SYSTEM_PROMPT + ("\n\n" + FEW_SHOT_BLOCK if few_shot else "")
    if pretext is not None:
        # Mirrors the highlighted target sentence annotators saw in the app's context
        # window (pretext + posttext, ~200-word window) instead of the bare sentence.
        passage = f"{pretext} >>> {sentence.strip()} <<< {posttext}".strip()
        user = ('Passage, with the target sentence marked between >>> and <<<:\n"' + passage + '"\n\n'
                "Classify ONLY the marked target sentence, using the surrounding passage as context. "
                "Return the label and explanation in the required JSON format only.")
    else:
        user = ('Target sentence:\n"' + sentence.strip() + '"\n\n'
                "Classify the target sentence. Return the label and explanation in the required JSON format only.")
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


# The few-shot examples are shown to the model as gold answers; if any also appear in the
# eval set they would leak. Remove them so every condition scores the same clean set.
import re as _re_fs
FEWSHOT_TEXTS = {m.group(1).strip() for m in _re_fs.finditer(r'^"(.*)" ->', FEW_SHOT_BLOCK, _re_fs.M)}
_before = len(eval_df)
eval_df = eval_df[~eval_df['text'].str.strip().isin(FEWSHOT_TEXTS)].reset_index(drop=True)
if _before != len(eval_df):
    print(f'removed {_before - len(eval_df)} few-shot example sentence(s) from the eval set')


# --- run log (from notebook) ---
RUN_TIMESTAMP = datetime.now().astimezone()
RUN_ID = RUN_TIMESTAMP.strftime("%Y%m%d-%H%M%S")

LOG_DIR = RESULTS_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

RUN_LOG_PATH = LOG_DIR / f"run_log_{RUN_ID}.json"

RUN_LOG = {
    "run_id": RUN_ID,
    "runner": "run_llama_benchmark.py",   # distinguishes headless runs from notebook runs
    "started_at": RUN_TIMESTAMP.isoformat(),
    "status": "running",
    "prompt": {
        **PROMPT_METADATA,
        "system_prompt_hash": text_hash(SYSTEM_PROMPT),
        "few_shot_block_hash": text_hash(FEW_SHOT_BLOCK),
    },
    "conditions": RUN_CONDITIONS,
    "models": {},
    "usage": {},
    "errors": [],
}

for name in ENABLED:
    cfg = MODELS[name]
    RUN_LOG["models"][name] = {
        "requested_model": cfg["model"],
        "response_model": None,  # filled in after first successful API call
        "base_url": cfg["base_url"],
        "api_key_env": cfg["key_env"],
    }


def save_run_log():
    with RUN_LOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(RUN_LOG, f, ensure_ascii=False, indent=2)


save_run_log()
print(f"run log -> {RUN_LOG_PATH}", flush=True)


# --- parse_label / parse_explanation (from notebook) ---
ALIASES = {
    # codebook names -> canonical training names
    "haptic_human":        "haptic_human_animal",
    "facial_expressions":  "facial_expr",
    "facial_expression":   "facial_expr",
    "gestures":            "gesture",
    "sents_seg_err":       "segmentation_error",
    "sents_seg_error":     "segmentation_error",
    # labels folded/excluded upstream of training
    "human_cognition":     "no_rel",
    "character_cognition": "no_rel",
    "character_static":    "no_rel",
    "description_feltness": "no_rel",
}


def parse_label(raw):
    if not raw or not isinstance(raw, str):
        return None
    cand = None
    m = re.search(r"\{[^{}]*\}", raw, re.S)
    if m:
        try:
            cand = json.loads(m.group(0)).get("label")
        except json.JSONDecodeError:
            pass
    if cand is None:  # fall back: longest known label mentioned anywhere
        low = raw.lower()
        hits = [l for l in list(ALIASES) + LABELS if l in low]
        cand = max(hits, key=len) if hits else None
    if cand is None:
        return None
    cand = str(cand).strip().lower()
    return ALIASES.get(cand, cand if cand in LABELS else None)


def parse_explanation(raw):
    if not raw or not isinstance(raw, str):
        return None
    match = re.search(r"\{[^{}]*\}", raw, re.S)
    if not match:
        return None
    try:
        explanation = json.loads(
            match.group(0)
        ).get("explanation")
    except json.JSONDecodeError:
        return None
    if explanation is None:
        return None
    explanation = str(explanation).strip()
    return explanation or None


# sanity checks
assert parse_label('{"label": "haptic_human"}') == "haptic_human_animal"
assert parse_label('The answer is {"label": "gestures"}.') == "gesture"
assert parse_label('{"label": "human_cognition"}') == "no_rel"
assert parse_label("nonsense") is None
assert parse_explanation('{"explanation": "This is a test explanation."}') == "This is a test explanation."
assert parse_explanation("nonsense") is None

# --- runner (from notebook) ---
import openai
from openai import OpenAI

_clients = {}


def get_client(cfg):
    ident = (cfg["base_url"], cfg["key_env"])
    if ident not in _clients:
        api_key = os.environ.get(cfg["key_env"], "") if cfg["key_env"] else "local"
        kw = {"api_key": api_key}
        if cfg["base_url"]:
            kw["base_url"] = cfg["base_url"]
        _clients[ident] = OpenAI(**kw)
    return _clients[ident]


PARAM_SETS = [
    dict(temperature=0, response_format={"type": "json_object"}),
    dict(temperature=0),
    dict(),
]


def classify(client, model_id, sentence, few_shot, pretext=None, posttext=None, max_retries=6):
    messages = build_messages(sentence, few_shot, pretext, posttext)
    p, last_err = 0, None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_id, messages=messages, **PARAM_SETS[p])
            content = resp.choices[0].message.content
            usage = None
            if getattr(resp, "usage", None) is not None:
                usage = {
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                    "total_tokens": getattr(resp.usage, "total_tokens", None),
                }
            response_model = getattr(resp, "model", None)
            return {"raw": content, "usage": usage, "response_model": response_model}  # message content + log data
        except openai.BadRequestError as e:
            last_err = e
            if p < len(PARAM_SETS) - 1:
                p += 1          # endpoint rejects a parameter -> drop it and retry
            else:
                raise
        except (openai.RateLimitError, openai.APIConnectionError,
                openai.APIStatusError) as e:
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"max retries exceeded ({last_err})")


RESULT_COLS = ["filename", "sentence_id", "text", "gold", "raw", "pred", "explanation"]


def run_benchmark(name, cfg, data, condition):
    condition_cfg = CONDITION_CONFIGS[condition]
    few_shot = condition_cfg["few_shot"]
    use_context = condition_cfg["use_context"]

    if use_context:
        data = data[data["pretext"].notna()]   # only rows with verified, recoverable context
    out_path = RESULTS_DIR / f"{name}__{condition}.csv"
    done = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(columns=RESULT_COLS)
    done_keys = set(zip(done["filename"], done["sentence_id"]))
    todo = data[[(f, s) not in done_keys
                 for f, s in zip(data["filename"], data["sentence_id"])]]
    if todo.empty:
        print(f"{name} / {condition}: all {len(done)} sentences cached")
        return done
    print(f"{name} / {condition}: {len(todo)} to classify ({len(done)} cached)")

    client = get_client(cfg)
    rows = list(done.to_dict("records"))

    def work(r):
        pre  = r.pretext if use_context else None
        post = r.posttext if use_context else None
        response = classify(client, cfg["model"], r.text, few_shot, pre, post)
        raw = response["raw"]
        return {
            "result": dict(
                filename=r.filename,
                sentence_id=r.sentence_id,
                text=r.text,
                gold=r.gold,
                raw=raw,
                pred=parse_label(raw),
                explanation=parse_explanation(raw),
            ),
            "usage": response["usage"],
            "response_model": response["response_model"],
        }

    usage_key = f"{name}__{condition}"
    if usage_key not in RUN_LOG["usage"]:
        RUN_LOG["usage"][usage_key] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "requests_with_usage": 0,
        }

    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futures = {
            ex.submit(work, r): r
            for r in todo.itertuples()
        }
        for i, fut in enumerate(tqdm(as_completed(futures), total=len(futures),
                                     desc=f"{name}/{condition}")):
            r = futures[fut]
            try:
                completed = fut.result()
                rows.append(completed["result"])
                usage = completed["usage"]
                response_model = completed["response_model"]
                if response_model is not None:
                    RUN_LOG["models"][name]["response_model"] = response_model
                usage_log = RUN_LOG["usage"][usage_key]
                if usage is not None:
                    for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                        value = usage.get(key)
                        if value is not None:
                            usage_log[key] += value
                    usage_log["requests_with_usage"] += 1
            except Exception as e:
                RUN_LOG["errors"].append({
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "model": name,
                    "condition": condition,
                    "filename": r.filename,
                    "sentence_id": r.sentence_id,
                    "error_type": type(e).__name__,
                    "message": str(e),
                })
                print("  FAILED:", str(e)[:200])
                save_run_log()
            if (i + 1) % 100 == 0:
                pd.DataFrame(rows)[RESULT_COLS].to_csv(out_path, index=False)
                save_run_log()

    result = pd.DataFrame(rows)[RESULT_COLS]

    # checking for explanations that are missing or empty
    missing_explanations = result["explanation"].isna().mean()

    result.to_csv(out_path, index=False)
    save_run_log()
    print(
        f"  saved {len(result)} rows -> {out_path.name} "
        f"(missing explanations: {missing_explanations:.1%})"
    )
    return result


# --- run loop ---
for name in ENABLED:
    cfg = MODELS[name]
    if cfg["key_env"] and not os.environ.get(cfg["key_env"]):
        print(f"-- skipping {name}: ${cfg['key_env']} not set", flush=True)
        continue
    for condition in RUN_CONDITIONS:
        print(f"\n>>> {name} / {condition}", flush=True)
        try:
            run_benchmark(name, cfg, eval_df, condition)
        except Exception as e:
            RUN_LOG["errors"].append({
                "timestamp": datetime.now().astimezone().isoformat(),
                "model": name,
                "condition": condition,
                "error_type": type(e).__name__,
                "message": str(e),
            })
            print(f"!!! {name}/{condition} FAILED: {type(e).__name__}: {e}", flush=True)

RUN_LOG["status"] = "completed"
RUN_LOG["finished_at"] = datetime.now().astimezone().isoformat()
save_run_log()

print("=== runner done:", time.strftime("%Y-%m-%d %H:%M:%S"), "===", flush=True)
