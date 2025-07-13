from pathlib import Path
from typing import List, Dict, Optional, Union

import pandas as pd
from transformers import AutoTokenizer
import torch

__all__ = [
    "MaterialInferenceDataset",
]


def _default_prompt_formatter(row: pd.Series) -> str:
    """Return a natural‑language prompt from a dataframe *row*.

    The default assumes the row has columns *P1*, *P2*, and *C*, building the
    classic two‑premise‑and‑conclusion miniature argument. Override by passing a
    different *prompt_formatter* to :class:`MaterialInferenceDataset` if you
    need a custom surface form for a specific CSV variant.
    """
    return f"{row['P1']}. {row['P2']}. Therefore, {row['C']}."


class MaterialInferenceDataset:
    """Lightweight dataset wrapper for the *material‑inference* CSV files.

    The class mirrors the public surface of *SyllogismDataset* so you can swap
    it into existing training pipelines with minimal friction.
    """

    def __init__(
        self,
        csv_dir: Union[str, Path],
        *,
        files: Optional[List[str]] = None,
        tokenizer_name: str = "gpt2",
        device: str = "cpu",
        prompt_formatter=_default_prompt_formatter,
    ) -> None:
        """Read the CSV files and build *input* / *label* pairs.

        Parameters
        ----------
        csv_dir
            Folder containing the material‑inference CSVs.
        files
            Explicit list of CSV filenames to load (e.g. ["if_then_100.csv" ]).
            If *None*, every ``*.csv`` file in *csv_dir* is consumed.
        tokenizer_name
            🤗 model name or local path for the tokenizer.
        device
            Torch device string ("cpu", "cuda", "mps" …).
        prompt_formatter
            Callable that converts one *row* (``pd.Series``) into a string
            prompt. Must access at least the column used as the *label*.
        """
        self.csv_dir = Path(csv_dir)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.prompt_formatter = prompt_formatter

        self.data: List[Dict[str, str]] = []

        # ── gather files ────────────────────────────────────────────────────
        paths: List[Path]
        if files is None:
            paths = sorted(self.csv_dir.glob("*.csv"))
        else:
            paths = [self.csv_dir / fname for fname in files]

        # ── load each CSV and build prompt/label dicts ──────────────────────
        for csv_path in paths:
            df = pd.read_csv(csv_path)

            # Heuristic: the rightmost column is typically the conclusion / label
            label_col = df.columns[-1]

            for _, row in df.iterrows():
                prompt = prompt_formatter(row)
                label = row[label_col]

                self.data.append(
                    {
                        "input": prompt,
                        "label": str(label),
                        "file": csv_path.name,
                    }
                )

        # ── public convenience lists mirroring *SyllogismDataset* ───────────
        self.sentences: List[str] = [d["input"] for d in self.data]
        self.labels: List[str] = [d["label"] for d in self.data]

    # =========================================================================
    # Optional helpers – nice for quick inspection & model input preparation
    # =========================================================================

    def tokenise(self, max_length: int = 128, prepend_bos: bool = False):
        """Return *input_ids* and *attention_mask* tensors, ready for a model."""
        enc = self.tokenizer(
            self.sentences,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if prepend_bos:
            bos = torch.full((enc.input_ids.size(0), 1), self.tokenizer.bos_token_id)
            enc["input_ids"] = torch.cat([bos, enc.input_ids], dim=1)
            bos_mask = torch.ones_like(bos)
            enc["attention_mask"] = torch.cat([bos_mask, enc.attention_mask], dim=1)
        return {k: v.to(self.device) for k, v in enc.items()}

    # ---------------------------------------------------------------------
    # Regex template discovery (optional but often handy)
    # ---------------------------------------------------------------------

    @staticmethod
    def derive_regex_template(strings: List[str]) -> str:
        """Return a minimal regex capturing *only* the variable parts.

        Uses the token‑by‑token algorithm developed in earlier prototyping.
        """
        import re
        from itertools import groupby

        tokenised = [re.findall(r"\w+|[^\w\s]", s) for s in strings]
        base = tokenised[0]
        tokens = []
        for i, tok in enumerate(base):
            constant = all(i < len(ts) and ts[i] == tok for ts in tokenised[1:])
            tokens.append(re.escape(tok) if constant else r"(.+?)")
        tokens = [t if t != r"(.+?)" else r"(.+?)" for t, _ in groupby(tokens)]
        return r"\s*".join(tokens)

    def regex_by_column(self, column: str) -> str:
        """Compute the shared skeleton for an individual *column* across *all* rows."""
        col_strings = [row[column] for row in self.iter_rows()]
        return self.derive_regex_template(col_strings)

    # ---------------------------------------------------------------------
    # Convenience: iterable/len interface to behave like a normal dataset
    # ---------------------------------------------------------------------

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def iter_rows(self):
        """Yield each *pandas Series* representing the original CSV rows."""
        for item in self.data:
            yield item
