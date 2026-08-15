"""Paths, label set and plotting defaults shared by the notebooks and scripts."""
import os
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
FIGURES = REPO / "figures"
TABLES = REPO / "tables"

# Inputs too large to commit. Override with environment variables.
CORPUS_DIR = Path(os.environ.get(
    "EMBODIMENT_CORPUS", "/data/TL_00/gutenberg_prose_en"))
CORPUS_META = CORPUS_DIR / "metadata" / "gutenberg_en.csv"
MODEL_DIR = Path(os.environ.get(
    "EMBODIMENT_MODELS", "/data/TL_08_KR_JR/embodiment/Models"))
# Set once the checkpoint is released on the Hub; until then the notebooks and
# scripts read the local copy under MODEL_DIR.
HF_MODEL = None
PREDICTIONS_DIR = Path(os.environ.get(
    "EMBODIMENT_PREDICTIONS",
    "/data/TL_08_KR_JR/annotate/Data/results/predict-roberta-gutenberg-en-20260719-1415"))
BENCHMARK_DIR = Path(os.environ.get(
    "EMBODIMENT_BENCHMARK",
    "/data/TL_08_KR_JR/annotate/Data/results/llm_benchmark_v12"))
# Raw annotation exports, used only by notebooks/01_prepare_data.ipynb.
ANNOTATION_DIR = Path(os.environ.get(
    "EMBODIMENT_ANNOTATIONS", "/data/TL_08_KR_JR/annotate"))
# The gold set with each sentence's surrounding passage attached, built by the
# annotation pipeline and read by scripts/run_llm_benchmark.py.
CONTEXT_CSV = ANNOTATION_DIR / "Data" / "embodiment_full_with_context.csv"

# Committed data
GOLD = DATA / "gold_annotations.csv"
TRAIN = DATA / "train.csv"
DEV = DATA / "dev.csv"
TEST = DATA / "test.csv"
CORRECTIONS = DATA / "label_corrections.csv"
CV_PREDICTIONS = DATA / "cv_predictions.csv"
DOUBLE_CODED = DATA / "double_coded.csv"
ENGLISH_ORIGINALS = DATA / "english_originals.txt"
BOOK_COUNTS = DATA / "book_label_counts.csv"
BOOK_TOKENS = DATA / "book_token_counts.csv"

LABEL2ID = {
    "no_rel": 0,
    "character_move": 1,
    "haptic_material": 2,
    "character_feltness": 3,
    "facial_expr": 4,
    "sight_human": 5,
    "book_paratext": 6,
    "haptic_human_animal": 7,
    "sight_material": 8,
    "segmentation_error": 9,
    "gesture": 10,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
LABELS = list(LABEL2ID)

# The four categories that share the person/material contrast (Section 5).
TOUCH_SIGHT = ["haptic_human_animal", "haptic_material",
               "sight_human", "sight_material"]

# Labels from earlier drafts of the scheme, folded into no_rel or dropped.
RETIRED = {"character_cognition", "description_feltness", "character_static"}

SEED = 42

BLUE, AQUA = "#2a78d6", "#1baf7a"     # person-directed, material-directed
INK, SEC_INK, MUTED, GRID, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb")


def set_style():
    """Figure defaults; fonttype 42 embeds TrueType rather than Type 3."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": SEC_INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 11,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save_fig(fig, name, dpi=200):
    """Write a figure to figures/ as both PDF (vector) and PNG."""
    FIGURES.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=dpi, bbox_inches="tight")
    print(f"saved -> figures/{name}.{{pdf,png}}")
