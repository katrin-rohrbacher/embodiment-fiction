# Modelling Embodiment in Narrative Fiction

Code and gold-standard data for the paper. Sentence-level annotation scheme for embodiment
in narrative fiction, a fine-tuned RoBERTa classifier, an LLM benchmark, and the corpus
analysis of touch and sight.

## Layout

```
embodiment/config.py   paths, label set, plotting defaults
notebooks/             the analysis, in run order
scripts/               long-running jobs (corpus prediction, LLM benchmark)
codebook/              annotation codebook (Appendix A) and LLM prompt (Appendix B)
data/                  gold annotations, splits, book-level prediction counts
figures/  tables/      generated output
```

## Notebooks

| Notebook | Produces |
|---|---|
| `01_prepare_data.ipynb` | `data/gold_annotations.csv`, train/dev/test splits |
| `02_agreement.ipynb` | inter-annotator agreement (Table 2) |
| `03_train_classifier.ipynb` | fine-tuned model, held-out and CV scores (Table 3) |
| `04_llm_benchmark.ipynb` | LLM scores (Tables 4 and 5) |
| `05_corpus_descriptives.ipynb` | corpus size by decade (Figure 1) |
| `06_touch_sight.ipynb` | touch/sight results (Figures 2–4, Section 5) |
| `07_precedence_check.ipynb` | touch-over-sight precedence bound (Section 5) |
| `08_touch_sight_referents.ipynb` | person vs. animal referents of touch and sight (Section 5) |

Notebooks 02, 04, 05 and 06 run from the committed data alone. Notebook 03 needs a GPU.
Notebook 01 needs the raw annotation exports, which are not committed.

## Data

Committed (`data/`):

| File | Rows | What |
|---|---|---|
| `gold_annotations.csv` | 3,861 | adjudicated gold standard, one label per sentence |
| `train.csv` / `dev.csv` / `test.csv` | 2,702 / 579 / 580 | stratified 70/15/15 split |
| `label_corrections.csv` | 387 | append-only adjudication log, applied at load time |
| `cv_predictions.csv` | 3,861 | 5-fold out-of-fold predictions |
| `double_coded.csv` | 2,003 | the double-coded sample, for agreement |
| `book_label_counts.csv` | 4,246 | per-book sentence and label counts over the corpus |
| `book_token_counts.csv` | 4,246 | per-book token counts |
| `english_originals.txt` | 4,077 | filenames of works originally written in English |


## Model

The fine-tuned checkpoint is released on the Hugging Face Hub: https://huggingface.co/katrohrb/embodiment-classifier. 

```python
from transformers import pipeline

clf = pipeline("text-classification", model="/path/to/checkpoint")
clf("She laid her hands on her husband's shoulders.")
# [{'label': 'haptic_human_animal', 'score': 0.99}]
```

## Setup

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

`torch` is needed only for notebook 03 and `scripts/predict_corpus.py`.

## Scripts

```
python scripts/predict_corpus.py                      # classify the corpus (GPU, hours)
python scripts/run_llm_benchmark.py qwen3-32b-local   # one model, both context conditions
python scripts/score_benchmark.py                     # rebuild the benchmark table
```

`run_llm_benchmark.py` checkpoints every 100 sentences and resumes on restart. Hosted
models read their key from `OPENAI_API_KEY`; local models expect an Ollama server on
`localhost:11434`.

## Corpus

4,077 works of English-language prose fiction published 1780–1930, from Project Gutenberg,
with metadata from the Gender Novel Corpus. Sentence-segmented with Stanza; the sentence is
the unit of annotation, classification and every rate reported.
