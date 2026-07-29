import re
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from core.config import HARM_BENCH_CSV, ROOT_DIR

# Cached, deterministic filtered JBB-Behaviors prompt set: the 55 Source==Original
# rows minus any that fuzzy-match HarmBench-standard or AdvBench at token-Jaccard
# >= JBB_OVERLAP_THRESHOLD. We persist it so re-runs use the exact same prompt
# subset across the long generation/judge/guard pipeline.
JBB_FILTERED_CSV = ROOT_DIR / "data" / "jailbreakbench_filtered.csv"
JBB_OVERLAP_THRESHOLD = 0.70

# StrongREJECT (Souly et al., NeurIPS 2024). 313 prompts; we drop rows tagged
# source=AdvBench (25) plus any that fuzzy-match HarmBench-standard / AdvBench
# / SocialHarmBench / JBB at the same Jaccard threshold used for JBB. The
# survivors are cached to a frozen CSV so the pipeline sees a stable subset.
SR_FILTERED_CSV = ROOT_DIR / "data" / "strongreject_filtered.csv"
SR_OVERLAP_THRESHOLD = 0.70

# SocialHarmBench is normally pulled from the HF Hub, but on offline/sandboxed
# hosts (e.g. Modal) a frozen local CSV is preferred when present.
SHB_FILTERED_CSV = ROOT_DIR / "data" / "socialharmbench_filtered.csv"


class DatasetLoader:
    def __init__(self):
        self.harmbench_csv_path = HARM_BENCH_CSV

    def load(self, dataset_name: str) -> pd.DataFrame:
        dataset_name = dataset_name.lower().strip()

        if dataset_name == "socialharmbench":
            return self._load_socialharmbench()
        elif dataset_name == "harmbench":
            return self._load_harmbench()
        elif dataset_name == "jailbreakbench":
            return self._load_jailbreakbench()
        elif dataset_name == "strongreject":
            return self._load_strongreject()
        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")

    def _load_socialharmbench(self) -> pd.DataFrame:
        if SHB_FILTERED_CSV.exists():
            df = pd.read_csv(SHB_FILTERED_CSV)
            return df.dropna(subset=["prompt_text"]).reset_index(drop=True)
        ds = load_dataset("psyonp/SocialHarmBench", split="train")
        df = ds.to_pandas()
        
        for col in ["prompt_id", "category", "sub_topic", "type"]:
            if col not in df.columns:
                df[col] = None
                
        front_cols = ["prompt_id", "category", "sub_topic", "type", "prompt_text"]
        rest_cols = [c for c in df.columns if c not in front_cols]
        df = df[front_cols + rest_cols]
        
        return df.dropna(subset=["prompt_text"]).reset_index(drop=True)

    def _load_harmbench(self) -> pd.DataFrame:
        if not self.harmbench_csv_path.exists():
            raise FileNotFoundError(f"HarmBench CSV not found at {self.harmbench_csv_path}")
            
        df = pd.read_csv(self.harmbench_csv_path)
        df.columns = [c.strip().rstrip(".") for c in df.columns]
        
        df["prompt_text"] = df["Behavior"].astype(str).str.strip()
        df["prompt_id"] = df["BehaviorID"]
        df["category"]  = df["SemanticCategory"]
        df["sub_topic"] = df["SemanticCategory"]
        df["type"]      = df["FunctionalCategory"]
        df = df[["prompt_id", "category", "sub_topic", "type", "prompt_text"]]
        # filter df to include only rows where FunctionalCategory is standard
        df = df[df["type"] == "standard"].reset_index(drop=True)
        return df.dropna(subset=["prompt_text"]).reset_index(drop=True)

    def _load_jailbreakbench(self) -> pd.DataFrame:
        """JBB-Behaviors harmful split, filtered to prompts that don't overlap
        HarmBench-standard or AdvBench. The filter runs once and caches its
        output to data/jailbreakbench_filtered.csv so the pipeline always sees
        the same prompt set across stages."""
        if JBB_FILTERED_CSV.exists():
            df = pd.read_csv(JBB_FILTERED_CSV)
            return df.dropna(subset=["prompt_text"]).reset_index(drop=True)

        jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful").to_pandas()

        # Drop the 45 rows JBB explicitly attributes to AdvBench / TDC-HarmBench.
        keep = jbb[jbb["Source"] == "Original"].copy()

        # Empirical near-duplicate filter against the actual prompt strings used
        # elsewhere in this project; token-Jaccard catches edits like
        # "Give detailed instructions" vs "give step by step instructions".
        hb = pd.read_csv(self.harmbench_csv_path)
        hb_std = hb[hb["FunctionalCategory"] == "standard"]["Behavior"].astype(str).tolist()
        adv = list(load_dataset("walledai/AdvBench", split="train")["prompt"])

        ref_norm = [_norm_for_overlap(s) for s in hb_std + adv]
        ref_tokens = [set(s.split()) for s in ref_norm]

        def has_overlap(text: str) -> bool:
            t = set(_norm_for_overlap(text).split())
            for r in ref_tokens:
                if not t or not r:
                    continue
                j = len(t & r) / len(t | r)
                if j >= JBB_OVERLAP_THRESHOLD:
                    return True
            return False

        keep["_overlap"] = keep["Goal"].apply(has_overlap)
        keep = keep[~keep["_overlap"]].drop(columns=["_overlap"])

        df = pd.DataFrame({
            "prompt_id":   keep["Index"].apply(lambda i: f"jbb_{int(i):03d}"),
            "category":    keep["Category"],
            "sub_topic":   keep["Behavior"],
            "type":        "standard",
            "prompt_text": keep["Goal"].astype(str).str.strip(),
        }).reset_index(drop=True)

        JBB_FILTERED_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(JBB_FILTERED_CSV, index=False)
        return df

    def _load_strongreject(self) -> pd.DataFrame:
        """StrongREJECT (Souly et al., 2024), 313 forbidden prompts. We drop
        rows StrongREJECT itself attributes to AdvBench, then run the same
        token-Jaccard filter against HarmBench-standard / AdvBench / JBB-Originals
        / SocialHarmBench so the surviving subset is disjoint from anything
        else in this project. Cached to data/strongreject_filtered.csv."""
        if SR_FILTERED_CSV.exists():
            df = pd.read_csv(SR_FILTERED_CSV)
            return df.dropna(subset=["prompt_text"]).reset_index(drop=True)

        sr = load_dataset("walledai/StrongREJECT", split="train").to_pandas()
        keep = sr[sr["source"] != "AdvBench"].copy()

        # Build the reference set we want to be disjoint from. JBB and
        # SocialHarmBench come from their own loaders so the filtering stays
        # consistent with what the rest of the pipeline considers "ours".
        hb = pd.read_csv(self.harmbench_csv_path)
        hb_std = hb[hb["FunctionalCategory"] == "standard"]["Behavior"].astype(str).tolist()
        adv = list(load_dataset("walledai/AdvBench", split="train")["prompt"])
        jbb_df = self._load_jailbreakbench()
        shb_df = self._load_socialharmbench()
        reference = (
            hb_std
            + adv
            + jbb_df["prompt_text"].astype(str).tolist()
            + shb_df["prompt_text"].astype(str).tolist()
        )
        ref_tokens = [set(_norm_for_overlap(s).split()) for s in reference]

        def has_overlap(text: str) -> bool:
            t = set(_norm_for_overlap(text).split())
            if not t:
                return False
            for r in ref_tokens:
                if not r:
                    continue
                if len(t & r) / len(t | r) >= SR_OVERLAP_THRESHOLD:
                    return True
            return False

        keep["_overlap"] = keep["prompt"].apply(has_overlap)
        keep = keep[~keep["_overlap"]].drop(columns=["_overlap"]).reset_index(drop=True)

        df = pd.DataFrame({
            "prompt_id":   [f"sr_{i:03d}" for i in range(len(keep))],
            "category":    keep["category"],
            "sub_topic":   keep["category"],  # StrongREJECT has no separate sub-topic
            "type":        "standard",
            "prompt_text": keep["prompt"].astype(str).str.strip(),
        }).reset_index(drop=True)

        SR_FILTERED_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(SR_FILTERED_CSV, index=False)
        return df


def _norm_for_overlap(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s