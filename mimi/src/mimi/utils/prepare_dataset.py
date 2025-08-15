from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Type, Optional

import torch as t
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
import random

# your existing import
from mimi.utils.global_variables import IMAGE_DIR, DATASET_DIR


class AMRType(str, Enum):
    ARG_SUB = "argument_substitution"
    PRED_SUB = "predicate_substitution"
    FRAME_SUB = "frame_substitution"
    COND_FRAME = "conditional_frame_insertion_substitution"
    ARG_INS = "argument_insertion"
    FRAME_CONJ = "frame_conjunction"
    ARG_PRED_GEN = "argument_predicate_generalisation"
    ARG_SUB_PROP = "property_inheritance"
    EXAMPLE = "example"
    IFT = "if_then"
    UNK = "unknown"


class Corruption(Enum):
    NO = "no"
    MID = "middle"
    ALL = "all"


def get_filtered_sample(iterable, excluded: List[str] = []):
    """
    Sample a single value from an iterable, excluding any values in `excluded`.
    Returns the first whitespace-separated token to guarantee 'one word'.
    """
    sample_list = pd.Series([x for x in iterable if x not in set(excluded)])
    # Guard: if everything was excluded (pathological), fall back to original iterable
    if sample_list.empty:
        sample_list = pd.Series(list(iterable))
    sample = sample_list.sample().iloc[0]
    return str(sample).split()[0]


# ===== AMR builder base & concrete classes =====

class BaseAMRBuilder:
    """
    Base interface for AMR-type-specific prompt/label generation & corruption.
    Subclasses may override any of these to customize text templates or label selection.
    """

    def __init__(self, df: pd.DataFrame, N: int, seed: int, tokenizer):
        self.df = df
        self.N = N
        self.seed = seed
        self.tokenizer = tokenizer
        self.samples = self.df.sample(n=self.N, random_state=self.seed)
        # Columns commonly used for corruptions (A,B,C refer to the classic syllogistic slots)
        self.A_col = df["Premise1_Subject"]
        self.B_col = df["Premise2_Subject"]   # middle term in classic ARG_SUB datasets
        self.C_col = df["Premise2_Object"]

    # ---------- core generation ----------

    def gen_prompt_label_pairs(self) -> List[Dict]:
        prompts = []
        for _, row in self.samples.iterrows():
            prompts.append(self.get_prompt_label_pair_from_row(row))
        return prompts

    def get_prompt_label_pair_from_row(self, row: pd.Series, b2: Optional[str] = None) -> Dict:
        """
        Default mapping (ARG_SUB-friendly): 
        A = Premise1_Subject
        B = Premise1_Object  (middle term)
        C = Premise2_Object
        """
        a = row["Premise1_Subject"]
        b = row["Premise1_Object"]       # middle term
        c = row["Premise2_Object"]       # NOTE: fixed bug where 'c' got overwritten
        return self.get_prompt_label_pair_from_row_and_abc(row, a, b, c, b2=b2)

    def get_prompt_label_pair_from_row_and_abc(
        self, row: pd.Series, a: str, b: str, c: str, b2: Optional[str] = None
    ) -> Dict:
        prompt = {}
        if not b2:
            b2 = b
        premise_1 = f"{a} {row['Premise1_Verb']} {b}"
        premise_2 = f"{b2} {row['Premise2_Verb']} {c}"
        conclusion_set_up = f"{a} {row['Conclusion_Verb']}"

        prompt["input"] = f"Since {premise_1} and {premise_2}, we can deduce that {conclusion_set_up}"
        prompt["a"] = a
        prompt["b"] = b
        prompt["b2"] = b2
        prompt["v1"] = row["Premise1_Verb"]
        prompt["v2"] = row["Premise2_Verb"]
        prompt["v3"] = row["Conclusion_Verb"]
        prompt["label"] = row["Conclusion_Object"]
        return prompt

    # ---------- corruptions ----------

    def corrupt_middle_term(self) -> List[Dict]:
        """
        Replace the middle term subject in premise 2 (b2) with a random alternative.
        """
        prompts = []
        for _, row in self.samples.iterrows():
            b2 = get_filtered_sample(self.B_col, [row["Premise2_Subject"]])
            prompts.append(self.get_prompt_label_pair_from_row(row, b2=b2))
        return prompts

    def corrupt_all_terms(self) -> List[Dict]:
        """
        Independently replace A, B, and C with random alternatives (one-word form).
        """
        prompts = []
        for _, row in self.samples.iterrows():
            a = get_filtered_sample(self.A_col, [row["Premise1_Subject"]])
            b = get_filtered_sample(self.B_col, [row["Premise2_Subject"]])
            c = get_filtered_sample(self.C_col, [row["Premise2_Object"]])
            prompts.append(self.get_prompt_label_pair_from_row_and_abc(row, a, b, c))
        return prompts


# You can specialize behaviors per type by overriding any of the above methods.
# Below, all classes inherit the safe default. Customize as needed later.

class ArgSubAMRBuilder(BaseAMRBuilder):
    """Argument substitution: uses the default BaseAMRBuilder behavior."""
    def __init__(self, df: pd.DataFrame, N: int, seed: int, tokenizer):
        super().__init__(df, N, seed, tokenizer)
        # Columns commonly used for corruptions (A,B,C refer to the classic syllogistic slots)
        self.A_col = df["Premise1_Subject"]
        self.B_col = df["Premise2_Subject"]   # middle term in classic ARG_SUB datasets
        self.C_col = df["Premise2_Object"]
        self.labels = [
            "BEGIN",
            "a_1",
            "a_1 => b",
            "b",
            "b -> c",
            "c",
            "c ∈ a_2",
            "a_2",
            "a_2 -> b",
            "b",
            "END"
        ]


    def get_prompt_label_pair_from_row(self, row: pd.Series, corruption: Optional[str] = None) -> Dict:
        """
        Default mapping (ARG_SUB-friendly): 
        A = Premise1_Subject
        B = Premise1_Object  (middle term)
        C = Premise2_Object
        """
        a = row["Premise1_Subject"]
        b = row["Premise1_Object"]       # middle term
        c = row["Premise2_Subject"]       # NOTE: fixed bug where 'c' got overwritten
        return self.get_prompt_label_pair_from_row_and_abc(row, a, b, c, corruption=corruption)
    
    def get_prompt_label_pair_from_row_and_abc(
        self, row: pd.Series, a: str, b: str, c: str, corruption: Optional[str] = None
    ) -> Dict:
        prompt = {}
        premise_1 = f"{a} {row['Premise1_Verb']} {b}"
        if corruption:
            premise_2 = f"{c} {row['Premise2_Verb']} {corruption}"
        else:
            premise_2 = f"{c} {row['Premise2_Verb']} {a}"
        conclusion_set_up = f"{b} {row['Conclusion_Verb']}"

        prompt["input"] = f"Since {premise_1} and {premise_2}, therefore {conclusion_set_up}"
        prompt["a"] = a
        prompt["b"] = b
        prompt["c"] = c
        prompt["v1"] = row["Premise1_Verb"]
        prompt["v2"] = row["Premise2_Verb"]
        prompt["v3"] = row["Conclusion_Verb"]
        prompt["corruption"] = corruption
        prompt["labels"] = (c, a)
        return prompt
    
    def corrupt_middle_term(self) -> List[Dict]:
        """
        Replace the middle term subject in premise 2 (a2) with a random alternative.
        """
        prompts = []
        for _, row in self.samples.iterrows():
            corruption = get_filtered_sample(self.A_col, [row["Premise1_Subject"]])
            prompts.append(self.get_prompt_label_pair_from_row(row, corruption=corruption))
        return prompts

    def corrupt_all_terms(self) -> List[Dict]:
        """
        Independently replace A, B, and C with random alternatives (one-word form).
        """
        prompts = []
        for _, row in self.samples.iterrows():
            a = get_filtered_sample(self.A_col, [row["Premise1_Subject"]])
            b = get_filtered_sample(self.B_col, [row["Premise1_Object"]])
            c = get_filtered_sample(self.C_col, [row["Premise2_Subject"]])
            prompts.append(self.get_prompt_label_pair_from_row_and_abc(row, a, b, c))
        return prompts

    def tlen(self,text):
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def get_label_token_lengths(self, prompts):
        """
        Compute the maximum token length for each symbolic label across the dataset,
        accounting for variable tokenization of a, b, and c (and verbs).
        Uses the new label set in self.labels.
        """
        max_len = {lab: 0 for lab in self.labels}


        # Fixed tokens for connectors / operators
        max_len["BEGIN"]      = self.tlen("Since ")
        max_len["b -> c"]     = self.tlen(" and ")
        max_len["a_2 -> b"]   = self.tlen(", therefore ")

        prompt_lens = []

        for prompt in prompts:
            prompt_len = {lab: 0 for lab in self.labels}

            # BEGIN
            prompt_len["BEGIN"] = self.tlen("Since ")

            # a_1
            prompt_len["a_1"] = self.tlen(prompt["a"] + " ")
            max_len["a_1"] = max(max_len["a_1"], prompt_len["a_1"])

            # a_1 => b
            prompt_len["a_1 => b"] = self.tlen(prompt["v1"] + " ")
            max_len["a_1 => b"] = max(max_len["a_1 => b"], prompt_len["a_1 => b"])

            # b
            prompt_len["b"] = self.tlen(prompt["b"] + " ")
            max_len["b"] = max(max_len["b"], prompt_len["b"])

            # b -> c
            prompt_len["b -> c"] = self.tlen(" and ")

            # c
            prompt_len["c"] = self.tlen(prompt["c"] + " ")
            max_len["c"] = max(max_len["c"], prompt_len["c"])

            # c ∈ a_2
            prompt_len["c ∈ a_2"] = self.tlen(prompt["v2"] + " ")
            max_len["c ∈ a_2"] = max(max_len["c ∈ a_2"], prompt_len["c ∈ a_2"])

            # a_2
            if prompt["corruption"]:
                prompt_len["a_2"] = self.tlen(prompt["corruption"] + " ")
            else:
                prompt_len["a_2"] = self.tlen(prompt["a"] + " ")

            max_len["a_2"] = max(max_len["a_2"], prompt_len["a_2"])

            # a_2 -> b
            prompt_len["a_2 -> b"] = self.tlen(", therefore ")

            # second b (conclusion b)
            prompt_len["b"] = max(prompt_len["b"], self.tlen(prompt["b"]))
            max_len["b"] = max(max_len["b"], prompt_len["b"])

            # END
            prompt_len["END"] = self.tlen(prompt["v3"])
            max_len["END"] = max(max_len["END"], prompt_len["END"])

            prompt_lens.append(prompt_len)

        return prompt_lens, max_len

    
    def get_adjusted_token_sequences(self, max_len, prompts) -> t.Tensor:
        tokenised_sentences = []

        # Fixed token sequences
        begin_tokens     = self.tokenizer("Since ", add_special_tokens=False)["input_ids"]
        and_tokens       = self.tokenizer(" and ", add_special_tokens=False)["input_ids"]
        deduction_tokens = self.tokenizer(", we can deduce that ", add_special_tokens=False)["input_ids"]

        for prompt in prompts:
            tokens = []

            # BEGIN
            tokens += begin_tokens

            # a_1
            tokens += self.tokenizer(prompt["a"], add_special_tokens=False,
                                padding="max_length", max_length=max_len["a_1"], truncation=True)["input_ids"]

            # a_1 => b
            tokens += self.tokenizer(prompt["v1"], add_special_tokens=False,
                                padding="max_length", max_length=max_len["a_1 => b"], truncation=True)["input_ids"]

            # b
            tokens += self.tokenizer(prompt["b"], add_special_tokens=False,
                                padding="max_length", max_length=max_len["b"], truncation=True)["input_ids"]

            # b -> c
            tokens += and_tokens

            # c
            tokens += self.tokenizer(prompt["c"], add_special_tokens=False,
                                padding="max_length", max_length=max_len["c"], truncation=True)["input_ids"]

            # c ∈ a_2
            tokens += self.tokenizer(prompt["v2"], add_special_tokens=False,
                                padding="max_length", max_length=max_len["c ∈ a_2"], truncation=True)["input_ids"]

            # a_2
            if prompt["corruption"]:
                tokens += self.tokenizer(prompt["corruption"], add_special_tokens=False,
                                padding="max_length", max_length=max_len["a_2"], truncation=True)["input_ids"]
            else:
                tokens += self.tokenizer(prompt["a"], add_special_tokens=False,
                                padding="max_length", max_length=max_len["a_2"], truncation=True)["input_ids"]

            # a_2 -> b
            tokens += deduction_tokens

            # b (conclusion)
            tokens += self.tokenizer(prompt["b"], add_special_tokens=False,
                                padding="max_length", max_length=max_len["b"], truncation=True)["input_ids"]

            # END
            tokens += self.tokenizer(prompt["v3"], add_special_tokens=False,
                                padding="max_length", max_length=max_len["END"], truncation=True)["input_ids"]

            tokenised_sentences.append(tokens)

        return t.tensor(tokenised_sentences, dtype=t.long)



class PredSubAMRBuilder(BaseAMRBuilder):
    """
    Predicate substitution: same base template for now.
    Optionally, you could override get_prompt_label_pair_from_row_and_abc to
    vary the phrasing or how labels are chosen for predicate-oriented tasks.
    """
    pass


class FrameSubAMRBuilder(BaseAMRBuilder):
    """Frame substitution: defaults to base behavior unless customized."""
    pass


class ConditionalFrameInsertionSubstitutionAMRBuilder(BaseAMRBuilder):
    """
    Conditional frame insertion/substitution.
    For a different surface form, you could override like:
      return f"If {premise_1} and {premise_2}, then {conclusion_set_up}"
    """
    # Example of a custom phrasing (commented out; enable if you want):
    # def get_prompt_label_pair_from_row_and_abc(self, row, a, b, c, b2=None):
    #     prompt = super().get_prompt_label_pair_from_row_and_abc(row, a, b, c, b2)
    #     # Rebuild input string using a conditional template
    #     premise_1 = f"{a} {row['Premise1_Verb']} {prompt['b']}"
    #     premise_2 = f"{prompt['b2']} {row['Premise2_Verb']} {row['Conclusion_Object']}"
    #     conclusion_set_up = f"{a} {row['Conclusion_Verb']}"
    #     prompt["input"] = f"If {premise_1} and {premise_2}, then {conclusion_set_up}"
    #     return prompt
    pass


class ArgInsAMRBuilder(BaseAMRBuilder):
    """Argument insertion: defaults to base behavior unless customized."""
    pass


class FrameConjAMRBuilder(BaseAMRBuilder):
    """Frame conjunction: defaults to base behavior unless customized."""
    pass


class ArgPredGenAMRBuilder(BaseAMRBuilder):
    """Argument/predicate generalisation: defaults to base behavior unless customized."""
    pass


class ArgSubPropAMRBuilder(BaseAMRBuilder):
    """Property inheritance: defaults to base behavior unless customized."""
    pass


class ExampleAMRBuilder(BaseAMRBuilder):
    """Example type: defaults to base behavior unless customized."""
    pass


class IfThenAMRBuilder(BaseAMRBuilder):
    """If-then phrasing (you can override template if you want a stronger 'if-then' surface)."""
    # Example (commented out):
    # def get_prompt_label_pair_from_row_and_abc(self, row, a, b, c, b2=None):
    #     prompt = super().get_prompt_label_pair_from_row_and_abc(row, a, b, c, b2)
    #     premise_1 = f"{a} {row['Premise1_Verb']} {prompt['b']}"
    #     premise_2 = f"{prompt['b2']} {row['Premise2_Verb']} {row['Conclusion_Object']}"
    #     conclusion_set_up = f"{a} {row['Conclusion_Verb']}"
    #     prompt["input"] = f"If {premise_1} and {premise_2}, then {conclusion_set_up}"
    #     return prompt
    pass


class UnknownAMRBuilder(BaseAMRBuilder):
    """Unknown type: keep generic behavior to avoid surprises."""
    pass


# Registry mapping AMRType -> builder class
AMR_BUILDERS: Dict[AMRType, Type[BaseAMRBuilder]] = {
    AMRType.ARG_SUB: ArgSubAMRBuilder,
    AMRType.PRED_SUB: PredSubAMRBuilder,
    AMRType.FRAME_SUB: FrameSubAMRBuilder,
    AMRType.COND_FRAME: ConditionalFrameInsertionSubstitutionAMRBuilder,
    AMRType.ARG_INS: ArgInsAMRBuilder,
    AMRType.FRAME_CONJ: FrameConjAMRBuilder,
    AMRType.ARG_PRED_GEN: ArgPredGenAMRBuilder,
    AMRType.ARG_SUB_PROP: ArgSubPropAMRBuilder,
    AMRType.EXAMPLE: ExampleAMRBuilder,
    AMRType.IFT: IfThenAMRBuilder,
    AMRType.UNK: UnknownAMRBuilder,
}


class MaterialInferenceDataset:
    def __init__(
        self,
        seed: int = 42,
        N: int = 100,
        type: AMRType = AMRType.ARG_SUB,
        corruption: Corruption = Corruption.NO,
        tokenizer=None,
    ):
        self.N = N
        self.seed = seed

        # Tokenizer setup
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("gpt2")
        space_id = self.tokenizer.encode(" ", add_special_tokens=False)[0]
        self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(space_id)   # set by string
        self.tokenizer.pad_token_id = space_id  
        #self.tokenizer.pad_token = self.tokenizer.eos_token
        #self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"
        self.prepend_bos = False

        # Dataframe (one CSV per type)
        self.df = pd.read_csv(f"{DATASET_DIR}/examples_100/{type.value}.csv")

        # Seeds
        random.seed(self.seed)
        np.random.seed(self.seed)


        # AMR strategy object based on `type`
        BuilderCls = AMR_BUILDERS.get(type, UnknownAMRBuilder)
        self.amr_object: BaseAMRBuilder = BuilderCls(self.df, self.N, self.seed, self.tokenizer)

        # Build prompts according to corruption mode
        if corruption == Corruption.NO:
            self.prompts = self.amr_object.gen_prompt_label_pairs()
        elif corruption == Corruption.MID:
            self.prompts = self.amr_object.corrupt_middle_term()
        elif corruption == Corruption.ALL:
            self.prompts = self.amr_object.corrupt_all_terms()
        else:
            # Fallback to NO if an unknown corruption mode somehow appears
            self.prompts = self.amr_object.gen_prompt_label_pairs()

        # Expose convenience attributes for downstream code
        self.sentences = [p["input"] for p in self.prompts]
        self.labels    = [p["labels"] for p in self.prompts]
        self.A         = [p["a"] for p in self.prompts]
        self.B         = [p["b"] for p in self.prompts]
        self.C        = [p["c"] for p in self.prompts]
        self.V1        = [p["v1"] for p in self.prompts]
        self.V2        = [p["v2"] for p in self.prompts]
        self.V3        = [p["v3"] for p in self.prompts]
