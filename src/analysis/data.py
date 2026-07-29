"""Load judged generation rows and name the paper's models.

Judged files live at rq1_runs/classified_gen_[adv_|placebo_]<slug>_<dataset>_<probe>.jsonl.
All loaders keep the canonical prompt set (src.analysis.filters); load_judged,
load_pooled, and paired also drop rows whose classifier_label is not YES or NO.
"""
import json

from core.config import OUT_DIR, PAPER_MODELS, ROOT_DIR
from src.analysis.filters import is_canonical

DATASETS = ["harmbench", "jailbreakbench", "strongreject", "socialharmbench"]
# Bootstrap draws index into the pooled list, so pool order moves SEs at the
# second decimal. decompose_delta, analyze_placebo, and plot_gap_collapse pool
# in this order.
POOL_ORDER = ["harmbench", "socialharmbench", "jailbreakbench", "strongreject"]
RESULTS_DIR = ROOT_DIR / "results"
FIG_DIR = ROOT_DIR / "paper" / "figures"

_SLUG = {name: slug for slug, name in PAPER_MODELS}


def models(*names):
    """Return (slug, display name) pairs for the named paper models, in the given order."""
    return [(_SLUG[name], name) for name in names]


def judged_path(slug, ds, kind="static", probe="f1"):
    """Return the judged file for one model, dataset, prefill kind, and probe.

    kind is "static" (control and static prefills), "adv", or "placebo".
    """
    prefix = {"static": "", "adv": "adv_", "placebo": "placebo_"}[kind]
    return OUT_DIR / f"classified_gen_{prefix}{slug}_{ds}_{probe}.jsonl"


def canonical_rows(path, ds):
    """Yield the canonical rows of one jsonl file, or nothing if it is absent."""
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if is_canonical(ds, r["prompt_id"]):
                yield r


def is_judged(row):
    """Return True if the row carries a YES or NO classifier verdict."""
    return row.get("classifier_label") in ("YES", "NO")


def claim(row):
    """Return 1 if the probe reply claims the response, else 0."""
    return int(row["classifier_label"] == "YES")


def load_judged(slug, ds, kind, condition, probe="f1"):
    """Return {prompt_id: row} over judged canonical rows of one condition.

    A later row for the same prompt replaces an earlier one. An absent file
    gives an empty dict.
    """
    return {r["prompt_id"]: r for r in canonical_rows(judged_path(slug, ds, kind, probe), ds)
            if r.get("condition") == condition and is_judged(r)}


def load_pooled(slug, kind, condition, probe="f1"):
    """Return {(dataset, prompt_id): row} pooled over DATASETS, as in load_judged."""
    return {(ds, pid): r for ds in DATASETS
            for pid, r in load_judged(slug, ds, kind, condition, probe).items()}


def paired(slug, probe="f1", datasets=DATASETS):
    """Return (dataset, prompt_id, control_row, adv_row) for every prompt judged in both conditions."""
    out = []
    for ds in datasets:
        ctrl = load_judged(slug, ds, "static", "control", probe)
        adv = load_judged(slug, ds, "adv", "adv_prefill", probe)
        for pid in set(ctrl) & set(adv):
            out.append((ds, pid, ctrl[pid], adv[pid]))
    return out
