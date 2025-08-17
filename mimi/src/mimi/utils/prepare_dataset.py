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
        

    # ---------- core generation ----------

    def gen_prompt_label_pairs(self) -> List[Dict]:
        prompts = []
        for _, row in self.samples.iterrows():
            prompts.append(self.get_prompt_label_pair_from_row(row))
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

        self.element_list = list(self.A_col) + list(self.B_col) + list(self.C_col)

        self.begin_str = "because "
        self.and_str = "and because "
        self.deduction_str = "thus "
        self.are_str = "are"
        self.belongs_str = "belongs to "

        self.labels = [
            "BEGIN",
            "a",
            "∈_1",
            "b1",
            "∧",
            "b2",
            "∈_2",
            "c",
            "=>",
            "a",
            "->"
        ]


    def get_prompt_label_pair_from_row(self, row: pd.Series, corruption: Optional[str] = None) -> Dict:
        """
        Default mapping (ARG_SUB-friendly): 
        A = Premise1_Subject
        B = Premise1_Object  (middle term)
        C = Premise2_Object
        """
        a = " " + row["Premise1_Subject"]
        b = " " + row["Premise1_Object"]       
        c = " " + row["Premise2_Object"]      
        return self.get_prompt_label_pair_from_row_and_components(row, a, b, c, corruption=corruption)
    
    def get_prompt_label_pair_from_row_and_components(
        self, row: pd.Series, a: str, b: str, c: str, corruption: Optional[str] = None
    ) -> Dict:
        prompt = {}
        premise_1 = f"{a} {self.belongs_str} {b}"
        if corruption:
            premise_2 = f"{corruption} {self.belongs_str} {c}"
        else:
            premise_2 = f"{b} {self.belongs_str} {c}"
        
        conclusion_set_up = f"{a} {self.belongs_str}"

        prompt["input"] = f"{self.begin_str}{premise_1} {self.and_str}{premise_2} {self.deduction_str}{conclusion_set_up} "
        prompt["a"] = a
        prompt["b"] = b
        prompt["c"] = c
        prompt["v1"] = row["Premise1_Verb"]
        prompt["v2"] = row["Premise2_Verb"]
        prompt["v3"] = row["Conclusion_Verb"]
        prompt["corruption"] = corruption
        prompt["labels"] = (c, b)
        return prompt
    
    def corrupt_middle_term(self) -> List[Dict]:
        """
        Replace the middle term subject in premise 2 (a2) with a random alternative.
        """
        prompts = []
        for _, row in self.samples.iterrows():
            corruption = get_filtered_sample(self.B_col, [row["Premise2_Subject"]])
            prompts.append(self.get_prompt_label_pair_from_row(row, corruption=corruption))
        return prompts

    def corrupt_all_terms(self) -> List[Dict]:
        """
        Independently replace A, B, and C with random alternatives (one-word form).
        """
        prompts = []
        for _, row in self.samples.iterrows():
            a = " " + get_filtered_sample(self.A_col, [row["Premise1_Subject"]])
            b = " " + get_filtered_sample(self.B_col, [row["Premise2_Subject"]])
            c = " " + get_filtered_sample(self.C_col, [row["Premise2_Object"]])
            prompts.append(self.get_prompt_label_pair_from_row_and_components(row, a, b, c))
        return prompts

    def tlen(self,text):
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def get_label_token_lengths(self, prompts):
        # init with all labels present
        max_len = {lab: 0 for lab in self.labels}

        # fixed operators (unchanged text)
        max_len["BEGIN"] = self.tlen(self.begin_str)
        max_len["∧"]     = self.tlen(self.and_str)
        max_len["=>"]    = self.tlen(self.deduction_str)

        # fixed relation used for both ∈_1, ∈_2 and the final ->
        rel_len = self.tlen(self.belongs_str)
        max_len["∈_1"] = rel_len
        max_len["∈_2"] = rel_len
        max_len["->"]  = rel_len

        prompt_lens = []
        for prompt in prompts:
            pl = {lab: 0 for lab in self.labels}

            # fixed pieces (constant per prompt)
            pl["BEGIN"] = max_len["BEGIN"]
            pl["∧"]     = max_len["∧"]
            pl["=>"]    = max_len["=>"]
            pl["∈_1"]   = max_len["∈_1"]
            pl["∈_2"]   = max_len["∈_2"]
            pl["->"]    = max_len["->"]

            # variable parts (note: prompt["a"], ["b"], ["c"] already include leading spaces upstream)
            a_len  = self.tlen(prompt["a"] + " ")
            b2_len = self.tlen(prompt["b"] + " ")
            c_len  = self.tlen(prompt["c"] + " ")

            if prompt.get("corruption"):
                b1_len = self.tlen(prompt["corruption"] + " ")
            else:
                b1_len = self.tlen(prompt["b"] + " ")

            # fill per-prompt lens
            pl["a"]  = a_len
            pl["b1"] = b1_len
            pl["b2"] = b2_len
            pl["c"]  = c_len

            # update global maxima
            max_len["a"]  = max(max_len.get("a", 0),  a_len)
            max_len["b1"] = max(max_len.get("b1", 0), b1_len)
            max_len["b2"] = max(max_len.get("b2", 0), b2_len)
            max_len["c"]  = max(max_len.get("c", 0),  c_len)

            prompt_lens.append(pl)

        return prompt_lens, max_len


    def get_adjusted_token_sequences(self, max_len, prompts) -> t.Tensor:
        """
        Sequence (labels shown; text unchanged):
        BEGIN  a  ∈_1  b1  ∧  b2  ∈_2  c  =>  a  ->
        """
        tokenised = []

        BEGIN = self.tokenizer(self.begin_str, add_special_tokens=False)["input_ids"]
        AND   = self.tokenizer(self.and_str, add_special_tokens=False)["input_ids"]
        DED   = self.tokenizer(self.deduction_str, add_special_tokens=False)["input_ids"]

        # Pre-tokenize the fixed relation for each label slot (kept separate for clarity/padding)
        REL1 = self.tokenizer(
            self.belongs_str, add_special_tokens=False,
            padding="max_length", max_length=max_len["∈_1"], truncation=True
        )["input_ids"]
        REL2 = self.tokenizer(
            self.belongs_str, add_special_tokens=False,
            padding="max_length", max_length=max_len["∈_2"], truncation=True
        )["input_ids"]
        REL3 = self.tokenizer(
            self.belongs_str, add_special_tokens=False,
            padding="max_length", max_length=max_len["->"], truncation=True
        )["input_ids"]

        for prompt in prompts:
            seq = []

            # BEGIN
            seq += BEGIN

            # a
            seq += self.tokenizer(
                prompt["a"], add_special_tokens=False,
                padding="max_length", max_length=max_len["a"], truncation=True
            )["input_ids"]

            # ∈_1 (fixed relation)
            seq += REL1

            # b1 (premise 1 object / possibly corrupted)
            b1_text = prompt["corruption"] if prompt.get("corruption") else prompt["b"]
            seq += self.tokenizer(
                b1_text, add_special_tokens=False,
                padding="max_length", max_length=max_len["b1"], truncation=True
            )["input_ids"]

            # ∧
            seq += AND

            # b2 (premise 2 subject)
            seq += self.tokenizer(
                prompt["b"], add_special_tokens=False,
                padding="max_length", max_length=max_len["b2"], truncation=True
            )["input_ids"]

            # ∈_2 (fixed relation)
            seq += REL2

            # c
            seq += self.tokenizer(
                prompt["c"], add_special_tokens=False,
                padding="max_length", max_length=max_len["c"], truncation=True
            )["input_ids"]

            # =>
            seq += DED

            # a (repeated in conclusion)
            seq += self.tokenizer(
                prompt["a"], add_special_tokens=False,
                padding="max_length", max_length=max_len["a"], truncation=True
            )["input_ids"]

            # -> (fixed relation used in the conclusion setup)
            seq += REL3

            tokenised.append(seq)

        return t.tensor(tokenised, dtype=t.long)




class PredSubAMRBuilder(BaseAMRBuilder):
    """
    Predicate substitution: same base template for now.
    Optionally, you could override get_prompt_label_pair_from_row_and_components to
    vary the phrasing or how labels are chosen for predicate-oriented tasks.
    """
    def __init__(self, df: pd.DataFrame, N: int, seed: int, tokenizer):
        super().__init__(df, N, seed, tokenizer)
        # Columns commonly used for corruptions (A,B,C refer to the classic syllogistic slots)
        self.A_col = df["Premise1_Subject"]
        self.B_col = df["Premise1_Object"]   # middle term in classic ARG_SUB datasets
        self.V1_col = df["Premise2_Subject"]
        self.V2_col = df["Premise2_Object"]

        self.element_list = list(self.V1_col) + list(self.V2_col)

        self.begin_str = "because"
        self.and_str = "and because"
        self.means_str = "means"
        self.deduction_str = "thus what"
        self.generic_verb = "do on"
        self.end_str = "is"

        self.labels = [
            "BEGIN",
            "a",
            "v1",
            "b",
            "∧",
            "v1_2",
            "->",
            "v2",
            "=>",
            "a_2",
            "-->",
            "b_2",
            "END"
        ]


    def get_prompt_label_pair_from_row(self, row: pd.Series, corruption: Optional[str] = None) -> Dict:
        
        a = " " + row["Premise1_Subject"]
        b = " " + row["Premise1_Object"]
        v1 = " " + row["Premise2_Subject"]         
        v2 = " " + row["Premise2_Object"]       
        return self.get_prompt_label_pair_from_row_and_components(row, a, b, v1, v2, corruption=corruption)
    
    def get_prompt_label_pair_from_row_and_components(
        self, row: pd.Series, a: str, b: str, v1: str, v2: str, corruption: Optional[str] = None
    ) -> Dict:
        prompt = {}
        premise_1 = f"{a}{v1}{b}"
        if corruption:
            premise_2 = f"{corruption} {self.means_str}{v2}"
        else:
            premise_2 = f"{v1} {self.means_str}{v2}"
        
        conclusion_set_up = f"{a} {self.generic_verb}{b} {self.end_str}"

        prompt["input"] = f"{self.begin_str}{premise_1} {self.and_str}{premise_2} {self.deduction_str}{conclusion_set_up}"
        prompt["a"] = a
        prompt["b"] = b
        prompt["v1"] = v1
        prompt["v2"] = v2
        prompt["corruption"] = corruption
        prompt["labels"] = (v2, v1)
        return prompt
    
    def corrupt_middle_term(self) -> List[Dict]:
        """
        Replace the middle term subject in premise 2 (a2) with a random alternative.
        """
        prompts = []
        for _, row in self.samples.iterrows():
            corruption = " " + get_filtered_sample(self.V1_col, [row["Premise2_Subject"]])
            prompts.append(self.get_prompt_label_pair_from_row(row, corruption=corruption))
        return prompts

    def corrupt_all_terms(self) -> List[Dict]:
        """
        Independently replace A, B, and C with random alternatives (one-word form).
        """
        prompts = []
        for _, row in self.samples.iterrows():
            a = " " + get_filtered_sample(self.A_col, [row["Premise1_Subject"]])
            b = " " + get_filtered_sample(self.B_col, [row["Premise1_Object"]])
            v1 = " " + get_filtered_sample(self.V2_col, [row["Premise2_Subject"]])
            v2 = " " + get_filtered_sample(self.V2_col, [row["Premise2_Object"]])
            prompts.append(self.get_prompt_label_pair_from_row_and_components(row, a, b, v1, v2))
        return prompts

    def tlen(self,text):
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def get_label_token_lengths(self, prompts):
        # init
        max_len = {lab: 0 for lab in self.labels}

        # fixed operators (unchanged text)
        max_len["BEGIN"] = self.tlen(self.begin_str)
        max_len["∧"]     = self.tlen(self.and_str)
        max_len["->"]    = self.tlen(self.means_str)
        max_len["=>"]    = self.tlen(self.deduction_str)
        max_len["-->"]   = self.tlen(self.generic_verb)
        max_len["END"]   = self.tlen(self.end_str)

        prompt_lens = []
        for prompt in prompts:
            pl = {lab: 0 for lab in self.labels}

            # fixed parts (constant per prompt)
            pl["BEGIN"] = max_len["BEGIN"]
            pl["∧"]     = max_len["∧"]
            pl["=>"]    = max_len["=>"]
            pl["->"]    = max_len["->"]
            pl["-->"]   = max_len["-->"]
            pl["END"]    = max_len["END"]

            # variable parts
            a_len   = self.tlen(prompt["a"] + " ")
            v1_len  = self.tlen(prompt["v1"] + " ")
            b_len  = self.tlen(prompt["b"] + " ")
            v2_len  = self.tlen(prompt["v2"] + " ")


            if prompt.get("corruption"):
                v1_2_len = self.tlen(prompt["corruption"] + " ")
            else:
                v1_2_len = self.tlen(prompt["v1"] + " ")


            # fill per-prompt
            pl["a"]  = a_len
            pl["a_2"]  = a_len
            pl["v1"]  = v1_len
            pl["b"] = b_len
            pl["b_2"] = b_len
            pl["v2"] = v2_len
            pl["v1_2"]  = v1_2_len

            # update global maxima
            max_len["a"]  = max(max_len["a"],  a_len)
            max_len["a_2"]  = max_len["a"]
            max_len["v1"]  = max(max_len["v1"],  v1_len)
            max_len["b"] = max(max_len["b"], b_len)
            max_len["b_2"] = max_len["b"]
            max_len["v1_2"] = max(max_len["v1_2"], v1_2_len)
            max_len["v2"] = max(max_len["v2"], v2_len)

            prompt_lens.append(pl)

        return prompt_lens, max_len


    
    def get_adjusted_token_sequences(self, max_len, prompts) -> t.Tensor:
        """
        Same assembly as before; only label names changed.
        Sequence (labels shown; text unchanged):
        BEGIN  a  ∈  b1  ∧  b2  ->  c(first=a_2)  =>  c(second=conclusion b)  <-
        """
        tokenised = []

        BEGIN = self.tokenizer(self.begin_str,               add_special_tokens=False)["input_ids"]
        AND   = self.tokenizer(self.and_str,                add_special_tokens=False)["input_ids"]
        DED   = self.tokenizer(self.deduction_str,add_special_tokens=False)["input_ids"]
        MEANS = self.tokenizer(self.means_str,               add_special_tokens=False)["input_ids"]
        GEN   = self.tokenizer(self.generic_verb,                add_special_tokens=False)["input_ids"]
        END   = self.tokenizer(self.end_str,add_special_tokens=False)["input_ids"]

        for prompt in prompts:
            seq = []

            # BEGIN
            seq += BEGIN

            # a
            seq += self.tokenizer(prompt["a"], add_special_tokens=False,
                    padding="max_length", max_length=max_len["a"], truncation=True)["input_ids"]

            # ∈ (v1)
            seq += self.tokenizer(prompt["v1"], add_special_tokens=False,
                    padding="max_length", max_length=max_len["v1"], truncation=True)["input_ids"]

            # b1 (premise b)
            seq += self.tokenizer(prompt["b"], add_special_tokens=False,
                    padding="max_length", max_length=max_len["b"], truncation=True)["input_ids"]

            # ∧
            seq += AND

            # b2 (premise c)
            v1_2_text = prompt["corruption"] if prompt.get("corruption") else prompt["v1"]
            seq += self.tokenizer(v1_2_text, add_special_tokens=False,
                    padding="max_length", max_length=max_len["v1"], truncation=True)["input_ids"]

            # -> (v2)
            seq += MEANS

            # c (
            seq += self.tokenizer(prompt["v2"], add_special_tokens=False,
                    padding="max_length", max_length=max_len["v2"], truncation=True)["input_ids"]

            # =>
            seq += DED

            # c (second occurrence = conclusion b)
            seq += self.tokenizer(prompt["a"], add_special_tokens=False,
                    padding="max_length", max_length=max_len["a"], truncation=True)["input_ids"]
            
            seq += GEN

            # <- (END = v3)
            seq += self.tokenizer(prompt["b"], add_special_tokens=False,
                    padding="max_length", max_length=max_len["b"], truncation=True)["input_ids"]
            
            seq += END

            tokenised.append(seq)

        return t.tensor(tokenised, dtype=t.long)


class FrameSubAMRBuilder(BaseAMRBuilder):
    """Frame substitution: defaults to base behavior unless customized."""
    def __init__(self, df: pd.DataFrame, N: int, seed: int, tokenizer):
        super().__init__(df, N, seed, tokenizer)
        # Columns commonly used for corruptions (A,B,C refer to the classic syllogistic slots)
        self.A_col = df["Premise1_Subject"]
        self.B_col = df["Premise2_Subject"]   # middle term in classic ARG_SUB datasets
        self.C_col = df["Premise2_Object"]

        self.element_list = list(self.A_col) + list(self.B_col) + list(self.C_col)

        self.begin_str = "because"
        self.and_str = "and because"
        self.deduction_str = "thus"
        self.end_str = "causes"

        self.labels = [
            "BEGIN",
            "a",
            "∈",
            "b1",
            "∧",
            "b2",
            "->",
            "c",
            "=>",
            "c",
            "<-"
        ]


    def get_prompt_label_pair_from_row(self, row: pd.Series, corruption: Optional[str] = None) -> Dict:
        """
        Default mapping (ARG_SUB-friendly): 
        A = Premise1_Subject
        B = Premise1_Object  (middle term)
        C = Premise2_Object
        """
        a = " " + row["Premise1_Subject"]
        b = " " + row["Premise2_Subject"]       # middle term
        c = " " + row["Premise2_Object"]       # NOTE: fixed bug where 'c' got overwritten
        return self.get_prompt_label_pair_from_row_and_components(row, a, b, c, corruption=corruption)
    
    def get_prompt_label_pair_from_row_and_components(
        self, row: pd.Series, a: str, b: str, c: str, corruption: Optional[str] = None
    ) -> Dict:
        prompt = {}
        if corruption:
            premise_1 = f"{a} {row['Premise1_Verb']}{corruption}"
        else:
            premise_1 = f"{a} {row['Premise1_Verb']}{b}"
        premise_2 = f"{b} {row['Premise2_Verb']}{c}"
        
        conclusion_set_up = f"{c} {self.end_str}"

        prompt["input"] = f"{self.begin_str}{premise_1} {self.and_str}{premise_2} {self.deduction_str}{conclusion_set_up} "
        prompt["a"] = a
        prompt["b"] = b
        prompt["c"] = c
        prompt["v1"] = row["Premise1_Verb"]
        prompt["v2"] = row["Premise2_Verb"]
        prompt["v3"] = self.end_str
        prompt["corruption"] = corruption
        prompt["labels"] = (a, b)
        return prompt
    
    def corrupt_middle_term(self) -> List[Dict]:
        """
        Replace the middle term subject in premise 2 (a2) with a random alternative.
        """
        prompts = []
        for _, row in self.samples.iterrows():
            corruption = " " + get_filtered_sample(self.B_col, [row["Premise2_Subject"]])
            prompts.append(self.get_prompt_label_pair_from_row(row, corruption=corruption))
        return prompts

    def corrupt_all_terms(self) -> List[Dict]:
        """
        Independently replace A, B, and C with random alternatives (one-word form).
        """
        prompts = []
        for _, row in self.samples.iterrows():
            a = " " + get_filtered_sample(self.A_col, [row["Premise1_Subject"]])
            b = " " + get_filtered_sample(self.B_col, [row["Premise2_Subject"]])
            c = " " + get_filtered_sample(self.C_col, [row["Premise2_Object"]])
            prompts.append(self.get_prompt_label_pair_from_row_and_components(row, a, b, c))
        return prompts

    def tlen(self,text):
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def get_label_token_lengths(self, prompts):
        # init
        max_len = {lab: 0 for lab in self.labels}

        # fixed operators (unchanged text)
        max_len["BEGIN"] = self.tlen(self.begin_str)
        max_len["∧"]     = self.tlen(self.and_str)
        max_len["=>"]    = self.tlen(self.deduction_str)

        prompt_lens = []
        for prompt in prompts:
            pl = {lab: 0 for lab in self.labels}

            # fixed parts (constant per prompt)
            pl["BEGIN"] = max_len["BEGIN"]
            pl["∧"]     = max_len["∧"]
            pl["=>"]    = max_len["=>"]

            # variable parts
            a_len   = self.tlen(prompt["a"] + " ")
            v1_len  = self.tlen(prompt["v1"] + " ")
            b2_len  = self.tlen(prompt["b"] + " ")
            v2_len  = self.tlen(prompt["v2"] + " ")
            c_len  = self.tlen(prompt["c"] + " ")


            if prompt.get("corruption"):
                b1_len = self.tlen(prompt["corruption"] + " ")
            else:
                b1_len = self.tlen(prompt["b"] + " ")


            # END piece
            v3_len = self.tlen(prompt["v3"])

            # fill per-prompt
            pl["a"]  = a_len
            pl["∈"]  = v1_len
            pl["b1"] = b1_len
            pl["b2"] = b2_len
            pl["->"] = v2_len
            # store the larger since label "c" is reused
            pl["c"]  = c_len
            pl["<-"] = v3_len

            # update global maxima
            max_len["a"]  = max(max_len["a"],  a_len)
            max_len["∈"]  = max(max_len["∈"],  v1_len)
            max_len["b1"] = max(max_len["b1"], b1_len)
            max_len["b2"] = max(max_len["b2"], b2_len)
            max_len["->"] = max(max_len["->"], v2_len)
            max_len["c"]  = max(max_len["c"],  c_len)
            max_len["<-"] = max(max_len["<-"], v3_len)

            prompt_lens.append(pl)

        return prompt_lens, max_len
    

    def get_adjusted_token_sequences(self, max_len, prompts) -> t.Tensor:
        """
        Same assembly as before; only label names changed.
        Sequence (labels shown; text unchanged):
        BEGIN  a  ∈  b1  ∧  b2  ->  c(first=a_2)  =>  c(second=conclusion b)  <-
        """
        tokenised = []

        BEGIN = self.tokenizer(self.begin_str,               add_special_tokens=False)["input_ids"]
        AND   = self.tokenizer(self.and_str,                add_special_tokens=False)["input_ids"]
        DED   = self.tokenizer(self.deduction_str,add_special_tokens=False)["input_ids"]

        for prompt in prompts:
            seq = []

            # BEGIN
            seq += BEGIN

            # a
            seq += self.tokenizer(prompt["a"], add_special_tokens=False,
                    padding="max_length", max_length=max_len["a"], truncation=True)["input_ids"]

            # ∈ (v1)
            seq += self.tokenizer(prompt["v1"], add_special_tokens=False,
                    padding="max_length", max_length=max_len["∈"], truncation=True)["input_ids"]

            # b1 (premise b)
            b1_text = prompt["corruption"] if prompt.get("corruption") else prompt["b"]
            seq += self.tokenizer(b1_text, add_special_tokens=False,
                    padding="max_length", max_length=max_len["b1"], truncation=True)["input_ids"]

            # ∧
            seq += AND

            # b2 (premise c)
            
            seq += self.tokenizer(prompt["b"], add_special_tokens=False,
                    padding="max_length", max_length=max_len["b2"], truncation=True)["input_ids"]

            # -> (v2)
            seq += self.tokenizer(prompt["v2"], add_special_tokens=False,
                    padding="max_length", max_length=max_len["->"], truncation=True)["input_ids"]

            # c (
            seq += self.tokenizer(prompt["c"], add_special_tokens=False,
                    padding="max_length", max_length=max_len["c"], truncation=True)["input_ids"]

            # =>
            seq += DED

            # c (second occurrence = conclusion b)
            seq += self.tokenizer(prompt["c"], add_special_tokens=False,
                    padding="max_length", max_length=max_len["c"], truncation=True)["input_ids"]

            # <- (END = v3)
            seq += self.tokenizer(prompt["v3"], add_special_tokens=False,
                    padding="max_length", max_length=max_len["<-"], truncation=True)["input_ids"]

            tokenised.append(seq)

        return t.tensor(tokenised, dtype=t.long)


class ConditionalFrameInsertionSubstitutionAMRBuilder(BaseAMRBuilder):
    def __init__(self, df: pd.DataFrame, N: int, seed: int, tokenizer):
        super().__init__(df, N, seed, tokenizer)
        # Columns commonly used for corruptions (A,B,C refer to the classic syllogistic slots)
        self.P1_col = df["Property1"]
        self.P2_col = df["Property2"]  
        self.S_col = df["Premise_Subject"]

        self.element_list = list(self.P1_col) + list(self.P2_col) + list(self.S_col)

        self.if_str = "if something is"
        self.then_str = "then that something"
        self.because_str = "because"
        self.deduction_str = "thus"

        self.labels = [
            "IF",
            "p1_1",
            "THEN",
            "v1",
            "p2",
            "->",
            "s_1",
            "v2",
            "p1_2",
            "=>",
            "s_2",
            "v3",
        ]


    def get_prompt_label_pair_from_row(self, row: pd.Series, corruption: Optional[str] = None) -> Dict:
        
        p1 = " " + row["Property1"]
        p2 = " " + row["Property2"]
        s = " " + row["Premise_Subject"]        
        return self.get_prompt_label_pair_from_row_and_components(row, p1, p2, s, corruption=corruption)
    
    def get_prompt_label_pair_from_row_and_components(
        self, row: pd.Series, p1: str, p2: str, s: str, corruption: Optional[str] = None
    ) -> Dict:
        prompt = {}
        if corruption:
            property_1 = corruption
        else:
            property_1 = p1
        
        premise = f"{s} {row['Premise_Verb']}{p1}"
        conclusion_set_up = f"{s} {row['Conclusion_Verb']}{p1}"

        prompt["input"] = f"{self.if_str}{property_1} {self.then_str}{row['Then_Verb']}{p2} {self.because_str}{premise} {self.deduction_str}{conclusion_set_up}"
        prompt["p1"] = p1
        prompt["p2"] = p2
        prompt["s"] = s
        prompt["v1"] = row["Then_Verb"]
        prompt["v2"] = row["Premise_Verb"]
        prompt["v3"] = row["Conclusion_Verb"]
        prompt["corruption"] = corruption
        prompt["labels"] = (p2, p1)
        return prompt
    
    def corrupt_middle_term(self) -> List[Dict]:
        """
        Replace the middle term subject in premise 2 (a2) with a random alternative.
        """
        prompts = []
        for _, row in self.samples.iterrows():
            corruption = " " + get_filtered_sample(self.P1_col, [row["Property1"]])
            prompts.append(self.get_prompt_label_pair_from_row(row, corruption=corruption))
        return prompts

    def corrupt_all_terms(self) -> List[Dict]:
        """
        Independently replace A, B, and C with random alternatives (one-word form).
        """
        prompts = []
        for _, row in self.samples.iterrows():
            p1 = " " + get_filtered_sample(self.P1_col, [row["Property1"]])
            p2 = " " + get_filtered_sample(self.P2_col, [row["Property2"]])
            s = " " + get_filtered_sample(self.S_col, [row["Premise_Subject"]])
            prompts.append(self.get_prompt_label_pair_from_row_and_components(row, p1, p2, s))
        return prompts

    def tlen(self,text):
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def get_label_token_lengths(self, prompts):
        # init
        max_len = {lab: 0 for lab in self.labels}

        # fixed operators (unchanged text)
        max_len["IF"] = self.tlen(self.if_str)
        max_len["THEN"]     = self.tlen(self.then_str)
        max_len["->"]    = self.tlen(self.because_str)
        max_len["=>"]    = self.tlen(self.deduction_str)

        prompt_lens = []
        for prompt in prompts:
            pl = {lab: 0 for lab in self.labels}

            # fixed parts (constant per prompt)
            pl["IF"] = max_len["IF"]
            pl["THEN"]     = max_len["THEN"]
            pl["=>"]    = max_len["=>"]
            pl["->"]    = max_len["->"]

            # variable parts
            p1_len   = self.tlen(prompt["p1"])
            v1_len  = self.tlen(prompt["v1"])
            p2_len  = self.tlen(prompt["p2"])
            v2_len  = self.tlen(prompt["v2"])
            s_len  = self.tlen(prompt["s"])
            v3_len  = self.tlen(prompt["v3"])


            if prompt.get("corruption"):
                p1_1_len = self.tlen(prompt["corruption"])
            else:
                p1_1_len = self.tlen(prompt["p1"])


            # fill per-prompt
            pl["p1_1"]  = p1_1_len
            pl["p1_2"]  = p1_len
            pl["v1"]  = v1_len
            pl["p2"] = p2_len
            pl["v2"] = v2_len
            pl["s"]  = s_len
            pl["s_2"]  = s_len
            pl["v3"] = v3_len

            # update global maxima
            max_len["p1_1"]  = max(max_len["p1_1"],  p1_1_len)
            max_len["p1_2"]  = max(max_len["p1_2"],  p1_len)
            max_len["v1"]  = max(max_len["v1"],  v1_len)
            max_len["p2"] = max(max_len["p2"], p2_len)
            max_len["v2"] = max(max_len["v2"], v2_len)
            max_len["s_1"]  = max(max_len["s_1"],  s_len)
            max_len["s_2"] = max_len["s_1"]
            max_len["v3"] = max(max_len["v3"], v3_len)

            prompt_lens.append(pl)

        return prompt_lens, max_len


    
    def get_adjusted_token_sequences(self, max_len, prompts) -> t.Tensor:
        """
        Same assembly as before; only label names changed.
        Sequence (labels shown; text unchanged):
        BEGIN  a  ∈  b1  ∧  b2  ->  c(first=a_2)  =>  c(second=conclusion b)  <-
        """
        tokenised = []

        IF = self.tokenizer(self.if_str,               add_special_tokens=False)["input_ids"]
        THEN   = self.tokenizer(self.then_str,                add_special_tokens=False)["input_ids"]
        DED   = self.tokenizer(self.deduction_str,add_special_tokens=False)["input_ids"]
        BEC = self.tokenizer(self.because_str,               add_special_tokens=False)["input_ids"]
        for prompt in prompts:
            seq = []

            # BEGIN
            seq += IF

            # a
            p1_1_text = prompt["corruption"] if prompt.get("corruption") else prompt["p1"]
            seq += self.tokenizer(p1_1_text, add_special_tokens=False,
                    padding="max_length", max_length=max_len["p1_1"], truncation=True)["input_ids"]
            
            seq += THEN

            # ∈ (v1)
            seq += self.tokenizer(prompt["v1"], add_special_tokens=False,
                    padding="max_length", max_length=max_len["v1"], truncation=True)["input_ids"]

            # b1 (premise b)
            seq += self.tokenizer(prompt["p2"], add_special_tokens=False,
                    padding="max_length", max_length=max_len["p2"], truncation=True)["input_ids"]

            # ∧
            seq += BEC

            # b2 (premise c)
            seq += self.tokenizer(prompt["s"], add_special_tokens=False,
                    padding="max_length", max_length=max_len["s_1"], truncation=True)["input_ids"]



            # c (
            seq += self.tokenizer(prompt["v2"], add_special_tokens=False,
                    padding="max_length", max_length=max_len["v2"], truncation=True)["input_ids"]
            
            seq += self.tokenizer(prompt["p1"], add_special_tokens=False,
                    padding="max_length", max_length=max_len["p1_2"], truncation=True)["input_ids"]

            # =>
            seq += DED

            # c (second occurrence = conclusion b)
            seq += self.tokenizer(prompt["s"], add_special_tokens=False,
                    padding="max_length", max_length=max_len["s_2"], truncation=True)["input_ids"]
            

            # <- (END = v3)
            seq += self.tokenizer(prompt["v3"], add_special_tokens=False,
                    padding="max_length", max_length=max_len["v3"], truncation=True)["input_ids"]
            

            tokenised.append(seq)

        return t.tensor(tokenised, dtype=t.long)


class ArgInsAMRBuilder(BaseAMRBuilder):
    """Argument insertion: defaults to base behavior unless customized."""
    def __init__(self, df: pd.DataFrame, N: int, seed: int, tokenizer):
        super().__init__(df, N, seed, tokenizer)
        # Columns commonly used for corruptions (A,B,C refer to the classic syllogistic slots)
        self.A_col = df["Premise1_Subject"]
        self.B_col = df["Premise2_Subject"]   # middle term in classic ARG_SUB datasets
        self.C_col = df["Premise2_Object"]
        self.V_col = df["Premise1_Verb"]

        self.element_list = list(self.A_col) + list(self.B_col) + list(self.C_col)

        self.begin_str = "because"
        self.and_str = "and because"
        self.deduction_str = "thus"
        self.that_str = "that"
        self.kindof = "is a kind of"

        self.labels = [
            "BEGIN",
            "a_1",
            "v1",
            "b",
            "∧",
            "a_2",
            "∈",
            "c_1",
            "=>",
            "a_3",
            "∈_2",
            "c_2",
            "THAT"
        ]


    def get_prompt_label_pair_from_row(self, row: pd.Series, corruption: Optional[str] = None) -> Dict:
        """
        Default mapping (ARG_SUB-friendly): 
        A = Premise1_Subject
        B = Premise1_Object  (middle term)
        C = Premise2_Object
        """
        a = " " + row["Premise1_Subject"]
        b = " " + row["Premise1_Object"]       # middle term
        c = " " + row["Premise2_Object"]       # NOTE: fixed bug where 'c' got overwritten
        v = " " + row["Premise1_Verb"] 
        return self.get_prompt_label_pair_from_row_and_components(row, a, b, c, v, corruption=corruption)
    
    def get_prompt_label_pair_from_row_and_components(
        self, row: pd.Series, a: str, b: str, c: str, v: str, corruption: Optional[str] = None
    ) -> Dict:
        prompt = {}
        if corruption:
            premise_1 = f"{corruption} {v}{b}"
        else:
            premise_1 = f"{a} {v}{b}"
        premise_2 = f"{a} {self.kindof}{c}"
        
        conclusion_set_up = f"{a} {self.kindof}{c} {self.that_str}"

        prompt["input"] = f"{self.begin_str}{premise_1} {self.and_str}{premise_2} {self.deduction_str}{conclusion_set_up} "
        prompt["a"] = a
        prompt["b"] = b
        prompt["c"] = c
        prompt["v1"] = row["Premise1_Verb"]
        prompt["corruption"] = corruption
        prompt["labels"] = (v, self.kindof)
        return prompt
    
    def corrupt_middle_term(self) -> List[Dict]:
        """
        Replace the middle term subject in premise 2 (a2) with a random alternative.
        """
        prompts = []
        for _, row in self.samples.iterrows():
            corruption = " " + get_filtered_sample(self.A_col, [row["Premise1_Subject"]])
            prompts.append(self.get_prompt_label_pair_from_row(row, corruption=corruption))
        return prompts

    def corrupt_all_terms(self) -> List[Dict]:
        """
        Independently replace A, B, and C with random alternatives (one-word form).
        """
        prompts = []
        for _, row in self.samples.iterrows():
            a = " " + get_filtered_sample(self.A_col, [row["Premise1_Subject"]])
            b = " " + get_filtered_sample(self.B_col, [row["Premise2_Subject"]])
            c = " " + get_filtered_sample(self.C_col, [row["Premise2_Object"]])
            v = " " + row["Premise1_Verb"]
            prompts.append(self.get_prompt_label_pair_from_row_and_components(row, a, b, c, v))
        return prompts

    def tlen(self,text):
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def get_label_token_lengths(self, prompts):
        """
        Compute per-label token lengths using the label set:
        ["BEGIN","a_1","v1","b","∧","a_2","∈","c_1","=>","a_3","∈_2","c_2","THAT"].

        Notes:
        - a_1 uses the corruption (if present) else the original 'a'.
        - ∈ and ∈_2 are the fixed string self.kindof.
        - In your current conclusion template, c_2 corresponds to 'b' (since you build
        'a kindof b that ...'). If you later switch to 'c' in the conclusion, change
        c2_len to use prompt["c"] instead.
        """
        # init maxima for all labels
        max_len = {lab: 0 for lab in self.labels}

        # fixed operators (unchanged text)
        max_len["BEGIN"] = self.tlen(self.begin_str)
        max_len["∧"]     = self.tlen(self.and_str)
        max_len["=>"]    = self.tlen(self.deduction_str)
        max_len["∈"]     = self.tlen(self.kindof)
        max_len["∈_2"]   = self.tlen(self.kindof)
        max_len["THAT"]  = self.tlen(self.that_str)

        prompt_lens = []
        for prompt in prompts:
            pl = {lab: 0 for lab in self.labels}

            # fixed parts copied in
            pl["BEGIN"] = max_len["BEGIN"]
            pl["∧"]     = max_len["∧"]
            pl["=>"]    = max_len["=>"]
            pl["∈"]     = max_len["∈"]
            pl["∈_2"]   = max_len["∈_2"]
            pl["THAT"]  = max_len["THAT"]

            # variable parts
            a1_text = prompt["corruption"] if prompt.get("corruption") else prompt["a"]
            a1_len  = self.tlen(a1_text)
            v1_len  = self.tlen(prompt["v1"])
            b_len   = self.tlen(prompt["b"])
            a2_len  = self.tlen(prompt["a"])     # original A
            c1_len  = self.tlen(prompt["c"])
            a3_len  = self.tlen(prompt["a"])     # original A
            c2_len  = self.tlen(prompt["c"])     # matches your current conclusion template

            # fill per-prompt
            pl["a_1"] = a1_len
            pl["v1"]  = v1_len
            pl["b"]   = b_len
            pl["a_2"] = a2_len
            pl["c_1"] = c1_len
            pl["a_3"] = a3_len
            pl["c_2"] = c2_len

            # update global maxima
            max_len["a_1"] = max(max_len["a_1"], a1_len)
            max_len["v1"]  = max(max_len["v1"],  v1_len)
            max_len["b"]   = max(max_len["b"],   b_len)
            max_len["a_2"] = max(max_len["a_2"], a2_len)
            max_len["c_1"] = max(max_len["c_1"], c1_len)
            max_len["a_3"] = max(max_len["a_3"], a3_len)
            max_len["c_2"] = max(max_len["c_2"], c2_len)

            prompt_lens.append(pl)

        return prompt_lens, max_len


    def get_adjusted_token_sequences(self, max_len, prompts) -> t.Tensor:
        """
        Assemble token sequences in the order:
        BEGIN  a_1  v1  b  ∧  a_2  ∈  c_1  =>  a_3  ∈_2  c_2  THAT

        Fixed strings:
        BEGIN=self.begin_str, ∧=self.and_str, ∈=self.kindof, =>=self.deduction_str, ∈_2=self.kindof, THAT=self.that_str
        """
        tokenised = []

        BEGIN = self.tokenizer(self.begin_str,    add_special_tokens=False)["input_ids"]
        AND   = self.tokenizer(self.and_str,      add_special_tokens=False)["input_ids"]
        IN    = self.tokenizer(self.kindof,       add_special_tokens=False)["input_ids"]
        DED   = self.tokenizer(self.deduction_str,add_special_tokens=False)["input_ids"]
        THAT  = self.tokenizer(self.that_str,     add_special_tokens=False)["input_ids"]

        for prompt in prompts:
            seq = []

            # BEGIN
            seq += BEGIN

            # a_1 (corruption if present, else original a)
            a1_text = prompt["corruption"] if prompt.get("corruption") else prompt["a"]
            seq += self.tokenizer(a1_text, add_special_tokens=False,
                                padding="max_length", max_length=max_len["a_1"], truncation=True)["input_ids"]

            # v1
            seq += self.tokenizer(prompt["v1"], add_special_tokens=False,
                                padding="max_length", max_length=max_len["v1"], truncation=True)["input_ids"]

            # b
            seq += self.tokenizer(prompt["b"], add_special_tokens=False,
                                padding="max_length", max_length=max_len["b"], truncation=True)["input_ids"]

            # ∧
            seq += AND

            # a_2
            seq += self.tokenizer(prompt["a"], add_special_tokens=False,
                                padding="max_length", max_length=max_len["a_2"], truncation=True)["input_ids"]

            # ∈
            seq += IN

            # c_1
            seq += self.tokenizer(prompt["c"], add_special_tokens=False,
                                padding="max_length", max_length=max_len["c_1"], truncation=True)["input_ids"]

            # =>
            seq += DED

            # a_3
            seq += self.tokenizer(prompt["a"], add_special_tokens=False,
                                padding="max_length", max_length=max_len["a_3"], truncation=True)["input_ids"]

            # ∈_2
            seq += IN

            # c_2 (matches your conclusion template; currently using 'c')
            seq += self.tokenizer(prompt["c"], add_special_tokens=False,
                                padding="max_length", max_length=max_len["c_2"], truncation=True)["input_ids"]

            # THAT
            seq += THAT

            tokenised.append(seq)

        return t.tensor(tokenised, dtype=t.long)



class FrameConjAMRBuilder(BaseAMRBuilder):
    def __init__(self, df: pd.DataFrame, N: int, seed: int, tokenizer):
        super().__init__(df, N, seed, tokenizer)
        # Columns commonly used for corruptions (A,B,C refer to the classic syllogistic slots)
        self.A_col = df["Premise1_Subject"]
        self.B_col = df["Premise2_Subject"]   # middle term in classic ARG_SUB datasets
        self.C_col = df["Premise2_Object"]

        self.element_list = list(self.A_col) + list(self.B_col) + list(self.C_col)

        self.begin_str = "because"
        self.and_because_str = "and because"
        self.deduction_str = "thus"
        self.and_str = "and"
        self.influence_str = "is influenced both by"

        self.labels = [
            "BEGIN",
            "a_1",
            "v1",
            "b",
            "∧",
            "c",
            "v2",
            "b",
            "=>",
            "b",
            "<-",
            "a_2",
            "AND"
        ]


    def get_prompt_label_pair_from_row(self, row: pd.Series, corruption: Optional[str] = None) -> Dict:
        """
        Default mapping (ARG_SUB-friendly): 
        A = Premise1_Subject
        B = Premise1_Object  (middle term)
        C = Premise2_Object
        """
        a = " " + row["Premise1_Subject"]
        b = " " + row["Premise1_Object"]       
        c = " " + row["Premise2_Subject"]       
        return self.get_prompt_label_pair_from_row_and_components(row, a, b, c, corruption=corruption)
    
    def get_prompt_label_pair_from_row_and_components(
        self, row: pd.Series, a: str, b: str, c: str, corruption: Optional[str] = None
    ) -> Dict:
        prompt = {}
        if corruption:
            premise_1 = f"{corruption} {row['Premise1_Verb']}{b}"
        else:
            premise_1 = f"{a} {row['Premise1_Verb']}{b}"
        premise_2 = f"{c} {row['Premise2_Verb']}{b}"
        
        conclusion_set_up = f"{b} {self.influence_str}{a} {self.and_str}"

        prompt["input"] = f"{self.begin_str}{premise_1} {self.and_because_str}{premise_2} {self.deduction_str}{conclusion_set_up} "
        prompt["a"] = a
        prompt["b"] = b
        prompt["c"] = c
        prompt["v1"] = row["Premise1_Verb"]
        prompt["v2"] = row["Premise2_Verb"]
        prompt["corruption"] = corruption
        prompt["labels"] = (c, a)
        return prompt
    
    def corrupt_middle_term(self) -> List[Dict]:
        """
        Replace the middle term subject in premise 2 (a2) with a random alternative.
        """
        prompts = []
        for _, row in self.samples.iterrows():
            corruption = " " + get_filtered_sample(self.A_col, [row["Premise1_Subject"]])
            prompts.append(self.get_prompt_label_pair_from_row(row, corruption=corruption))
        return prompts

    def corrupt_all_terms(self) -> List[Dict]:
        """
        Independently replace A, B, and C with random alternatives (one-word form).
        """
        prompts = []
        for _, row in self.samples.iterrows():
            a = " " + get_filtered_sample(self.A_col, [row["Premise1_Subject"]])
            b = " " + get_filtered_sample(self.B_col, [row["Premise2_Subject"]])
            c = " " + get_filtered_sample(self.C_col, [row["Premise2_Object"]])
            prompts.append(self.get_prompt_label_pair_from_row_and_components(row, a, b, c))
        return prompts

    def tlen(self,text):
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def get_label_token_lengths(self, prompts):
        """
        Compute per-label token lengths for the sequence:
        BEGIN  a_1  v1  b  ∧  c  v2  b  =>  b  <-  a_2  AND

        Fixed strings:
        BEGIN=self.begin_str ("because")
        ∧=self.and_because_str ("and because")
        =>=self.deduction_str ("thus")
        <-=self.influence_str ("is influenced both by")
        AND=self.and_str ("and")

        Notes:
        - a_1 uses the corruption (if present) else the original 'a'.
        - 'b' appears three times; we track a single max length for 'b'.
        """
        # init maxima for all labels present in self.labels
        max_len = {lab: 0 for lab in self.labels}

        # fixed operators (unchanged text)
        max_len["BEGIN"] = self.tlen(self.begin_str)
        max_len["∧"]     = self.tlen(self.and_because_str)
        max_len["=>"]    = self.tlen(self.deduction_str)
        max_len["<-"]    = self.tlen(self.influence_str)
        max_len["AND"]   = self.tlen(self.and_str)

        prompt_lens = []
        for prompt in prompts:
            pl = {lab: 0 for lab in self.labels}

            # copy fixed parts
            pl["BEGIN"] = max_len["BEGIN"]
            pl["∧"]     = max_len["∧"]
            pl["=>"]    = max_len["=>"]
            pl["<-"]    = max_len["<-"]
            pl["AND"]   = max_len["AND"]

            # variable parts
            a1_text = prompt["corruption"] if prompt.get("corruption") else prompt["a"]
            a1_len  = self.tlen(a1_text)
            v1_len  = self.tlen(prompt["v1"])
            b_len   = self.tlen(prompt["b"])
            c_len   = self.tlen(prompt["c"])
            v2_len  = self.tlen(prompt["v2"])
            a2_len  = self.tlen(prompt["a"])  # original A in the conclusion

            # fill per-prompt
            pl["a_1"] = a1_len
            pl["v1"]  = v1_len
            pl["b"]   = b_len
            pl["c"]   = c_len
            pl["v2"]  = v2_len
            pl["a_2"] = a2_len

            # update global maxima
            max_len["a_1"] = max(max_len["a_1"], a1_len)
            max_len["v1"]  = max(max_len["v1"],  v1_len)
            max_len["b"]   = max(max_len["b"],   b_len)
            max_len["c"]   = max(max_len["c"],   c_len)
            max_len["v2"]  = max(max_len["v2"],  v2_len)
            max_len["a_2"] = max(max_len["a_2"], a2_len)

            prompt_lens.append(pl)

        return prompt_lens, max_len
    def get_adjusted_token_sequences(self, max_len, prompts) -> t.Tensor:
        """
        Assemble token sequences in the order:
        BEGIN  a_1  v1  b  ∧  c  v2  b  =>  b  <-  a_2  AND

        Fixed strings:
        BEGIN=self.begin_str, ∧=self.and_because_str, =>=self.deduction_str,
        <-=self.influence_str, AND=self.and_str
        """
        tokenised = []

        BEGIN_IDS = self.tokenizer(self.begin_str,       add_special_tokens=False)["input_ids"]
        AND_BECS  = self.tokenizer(self.and_because_str, add_special_tokens=False)["input_ids"]
        DED_IDS   = self.tokenizer(self.deduction_str,   add_special_tokens=False)["input_ids"]
        INFL_IDS  = self.tokenizer(self.influence_str,   add_special_tokens=False)["input_ids"]
        AND_IDS   = self.tokenizer(self.and_str,         add_special_tokens=False)["input_ids"]

        for prompt in prompts:
            seq = []

            # BEGIN
            seq += BEGIN_IDS

            # a_1 (corruption if present, else original a)
            a1_text = prompt["corruption"] if prompt.get("corruption") else prompt["a"]
            seq += self.tokenizer(
                a1_text, add_special_tokens=False,
                padding="max_length", max_length=max_len["a_1"], truncation=True
            )["input_ids"]

            # v1
            seq += self.tokenizer(
                prompt["v1"], add_special_tokens=False,
                padding="max_length", max_length=max_len["v1"], truncation=True
            )["input_ids"]

            # b
            seq += self.tokenizer(
                prompt["b"], add_special_tokens=False,
                padding="max_length", max_length=max_len["b"], truncation=True
            )["input_ids"]

            # ∧ ("and because")
            seq += AND_BECS

            # c
            seq += self.tokenizer(
                prompt["c"], add_special_tokens=False,
                padding="max_length", max_length=max_len["c"], truncation=True
            )["input_ids"]

            # v2
            seq += self.tokenizer(
                prompt["v2"], add_special_tokens=False,
                padding="max_length", max_length=max_len["v2"], truncation=True
            )["input_ids"]

            # b (again)
            seq += self.tokenizer(
                prompt["b"], add_special_tokens=False,
                padding="max_length", max_length=max_len["b"], truncation=True
            )["input_ids"]

            # => ("thus")
            seq += DED_IDS

            # b (third time)
            seq += self.tokenizer(
                prompt["b"], add_special_tokens=False,
                padding="max_length", max_length=max_len["b"], truncation=True
            )["input_ids"]

            # <- ("is influenced both by")
            seq += INFL_IDS

            # a_2 (original a)
            seq += self.tokenizer(
                prompt["a"], add_special_tokens=False,
                padding="max_length", max_length=max_len["a_2"], truncation=True
            )["input_ids"]

            # AND ("and")
            seq += AND_IDS

            tokenised.append(seq)

        return t.tensor(tokenised, dtype=t.long)




class ArgPredGenAMRBuilder(BaseAMRBuilder):
    def __init__(self, df: pd.DataFrame, N: int, seed: int, tokenizer):
        super().__init__(df, N, seed, tokenizer)
        # Columns commonly used for corruptions (A,B,C refer to the classic syllogistic slots)
        self.A_col = df["Premise1_Subject"]
        self.B_col = df["Premise2_Subject"]   
        self.C_col = df["Premise2_Object"]

        self.element_list = list(self.A_col) + list(self.B_col) + list(self.C_col)

        self.begin_str = "because"
        self.and_str = "and because"
        self.deduction_str = "thus"
        self.kindof = "is a kind of"

        self.labels = [
            "BEGIN",
            "a",
            "->",
            "c_1",
            "∧",
            "b",
            "-->",
            "c_2",
            "=>",
            "b",
            "∈",
        ]


    def get_prompt_label_pair_from_row(self, row: pd.Series, corruption: Optional[str] = None) -> Dict:
        """
        Default mapping (ARG_SUB-friendly): 
        A = Premise1_Subject
        B = Premise1_Object  (middle term)
        C = Premise2_Object
        """
        a = " " + row["Premise1_Subject"]
        b = " " + row["Premise2_Subject"]       
        c = " " + row["Premise2_Object"]      
        return self.get_prompt_label_pair_from_row_and_components(row, a, b, c, corruption=corruption)
    
    def get_prompt_label_pair_from_row_and_components(
        self, row: pd.Series, a: str, b: str, c: str, corruption: Optional[str] = None
    ) -> Dict:
        prompt = {}
        premise_1 = f"{a} {row['Premise1_Verb']}{c}"
        if corruption:
            premise_2 = f"{b} {row['Premise2_Verb']}{corruption}"
        else:
            premise_2 = f"{b} {row['Premise2_Verb']}{c}"
        
        conclusion_set_up = f"{b} {self.kindof}"

        prompt["input"] = f"{self.begin_str}{premise_1} {self.and_str}{premise_2} {self.deduction_str}{conclusion_set_up} "
        prompt["a"] = a
        prompt["b"] = b
        prompt["c"] = c
        prompt["v1"] = row["Premise1_Verb"]
        prompt["corruption"] = corruption
        prompt["labels"] = (a, b)
        return prompt
    
    def corrupt_middle_term(self) -> List[Dict]:
        """
        Replace the middle term subject in premise 2 (a2) with a random alternative.
        """
        prompts = []
        for _, row in self.samples.iterrows():
            corruption = " " + get_filtered_sample(self.C_col, [row["Premise1_Subject"]])
            prompts.append(self.get_prompt_label_pair_from_row(row, corruption=corruption))
        return prompts

    def corrupt_all_terms(self) -> List[Dict]:
        """
        Independently replace A, B, and C with random alternatives (one-word form).
        """
        prompts = []
        for _, row in self.samples.iterrows():
            a = " " + get_filtered_sample(self.A_col, [row["Premise1_Subject"]])
            b = " " + get_filtered_sample(self.B_col, [row["Premise2_Subject"]])
            c = " " + get_filtered_sample(self.C_col, [row["Premise2_Object"]])
            prompts.append(self.get_prompt_label_pair_from_row_and_components(row, a, b, c))
        return prompts

    def tlen(self,text):
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def get_label_token_lengths(self, prompts):
        """
        Compute per-label token lengths for labels in self.labels, assuming the sequence:
        BEGIN  a  ->  c_1  ∧  b  -->  c_2  =>  b  ∈
        Fixed strings: BEGIN=self.begin_str, ∧=self.and_str, =>=self.deduction_str, ∈=self.kindof
        '->' is v1; '-->' is v2 (extracted from prompt['input']); c_2 is corruption if present else c.
        """
        # init maxima for all labels present in self.labels
        max_len = {lab: 0 for lab in self.labels}

        # fixed operators (unchanged text)
        if "BEGIN" in max_len: max_len["BEGIN"] = self.tlen(self.begin_str)
        if "∧"     in max_len: max_len["∧"]     = self.tlen(self.and_str)
        if "=>"    in max_len: max_len["=>"]    = self.tlen(self.deduction_str)
        if "∈"     in max_len: max_len["∈"]     = self.tlen(self.kindof)

        prompt_lens = []
        for prompt in prompts:
            pl = {lab: 0 for lab in self.labels}

            # fixed parts copied in
            if "BEGIN" in pl: pl["BEGIN"] = max_len["BEGIN"]
            if "∧"     in pl: pl["∧"]     = max_len["∧"]
            if "=>"    in pl: pl["=>"]    = max_len["=>"]
            if "∈"     in pl: pl["∈"]     = max_len["∈"]

            # variable parts
            a_len   = self.tlen(prompt["a"])
            v1_len  = self.tlen(prompt["v1"])
            c1_len  = self.tlen(prompt["c"])
            b_len   = self.tlen(prompt["b"])
            c2_text = prompt.get("corruption") or prompt["c"]
            c2_len  = self.tlen(c2_text)

            # Extract v2 text from prompt["input"]: after "<and_str><b> " and before c2_text
            s = prompt["input"]
            anchor = f"{self.and_str}{prompt['b']} "
            try:
                start = s.index(anchor) + len(anchor)
                end   = s.index(c2_text, start)
                v2_text = s[start:end]
            except ValueError:
                v2_text = ""
            v2_len = self.tlen(v2_text)

            # fill per-prompt
            if "a"     in pl: pl["a"]     = a_len
            if "->"    in pl: pl["->"]    = v1_len
            if "c_1"   in pl: pl["c_1"]   = c1_len
            if "b"     in pl: pl["b"]     = b_len   # used twice; single max is fine
            if "-->"   in pl: pl["-->"]   = v2_len
            if "c_2"   in pl: pl["c_2"]   = c2_len

            # update global maxima
            if "a"   in max_len: max_len["a"]   = max(max_len["a"],   a_len)
            if "->"  in max_len: max_len["->"]  = max(max_len["->"],  v1_len)
            if "c_1" in max_len: max_len["c_1"] = max(max_len["c_1"], c1_len)
            if "b"   in max_len: max_len["b"]   = max(max_len["b"],   b_len)
            if "-->" in max_len: max_len["-->"] = max(max_len["-->"], v2_len)
            if "c_2" in max_len: max_len["c_2"] = max(max_len["c_2"], c2_len)

            prompt_lens.append(pl)

        return prompt_lens, max_len


    def get_adjusted_token_sequences(self, max_len, prompts) -> t.Tensor:
        """
        Assemble token sequences in the order:
        BEGIN  a  ->  c_1  ∧  b  -->  c_2  =>  b  ∈

        Fixed strings:
        BEGIN=self.begin_str, ∧=self.and_str, =>=self.deduction_str, ∈=self.kindof
        """
        tokenised = []

        BEGIN = self.tokenizer(self.begin_str,    add_special_tokens=False)["input_ids"]
        AND   = self.tokenizer(self.and_str,      add_special_tokens=False)["input_ids"]
        DED   = self.tokenizer(self.deduction_str,add_special_tokens=False)["input_ids"]
        IN    = self.tokenizer(self.kindof,       add_special_tokens=False)["input_ids"]

        for prompt in prompts:
            seq = []

            # compute v2 text and c2 text for this prompt
            c2_text = prompt.get("corruption") or prompt["c"]
            s = prompt["input"]
            anchor = f"{self.and_str}{prompt['b']} "
            try:
                start = s.index(anchor) + len(anchor)
                end   = s.index(c2_text, start)
                v2_text = s[start:end]
            except ValueError:
                v2_text = ""

            # helpers
            def tok(text, L):
                return self.tokenizer(
                    text, add_special_tokens=False,
                    padding="max_length", max_length=L, truncation=True
                )["input_ids"]

            # BEGIN
            seq += BEGIN

            # a
            if "a" in max_len: seq += tok(prompt["a"], max_len["a"])

            # ->
            if "->" in max_len: seq += tok(prompt["v1"], max_len["->"])

            # c_1
            if "c_1" in max_len: seq += tok(prompt["c"], max_len["c_1"])

            # ∧
            seq += AND

            # b
            if "b" in max_len: seq += tok(prompt["b"], max_len["b"])

            # -->
            if "-->" in max_len: seq += tok(v2_text, max_len["-->"])

            # c_2
            if "c_2" in max_len: seq += tok(c2_text, max_len["c_2"])

            # =>
            seq += DED

            # b (again)
            if "b" in max_len: seq += tok(prompt["b"], max_len["b"])

            # ∈
            seq += IN

            tokenised.append(seq)

        return t.tensor(tokenised, dtype=t.long)



class ArgSubPropAMRBuilder(BaseAMRBuilder):
    """Property inheritance: defaults to base behavior unless customized."""
    def __init__(self, df: pd.DataFrame, N: int, seed: int, tokenizer):
        super().__init__(df, N, seed, tokenizer)
        # Columns commonly used for corruptions (A,B,C refer to the classic syllogistic slots)
        self.A_col = df["Premise1_Subject"]
        self.B_col = df["Premise2_Subject"]   # middle term in classic ARG_SUB datasets
        self.C_col = df["Premise2_Object"]

        self.element_list = list(self.A_col) + list(self.B_col) + list(self.C_col)

        self.begin_str = "since "
        self.and_str = "and since "
        self.deduction_str = ", therefore "

        self.labels = [
            "BEGIN",
            "a",
            "∈",
            "b1",
            "∧",
            "b2",
            "->",
            "c",
            "=>",
            "c",
            "<-"
        ]


    def get_prompt_label_pair_from_row(self, row: pd.Series, corruption: Optional[str] = None) -> Dict:
        """
        Default mapping (ARG_SUB-friendly): 
        A = Premise1_Subject
        B = Premise1_Object  (middle term)
        C = Premise2_Object
        """
        a = row["Premise1_Subject"]
        b = row["Premise2_Subject"]       # middle term
        c = row["Premise2_Object"]       # NOTE: fixed bug where 'c' got overwritten
        return self.get_prompt_label_pair_from_row_and_components(row, a, b, c, corruption=corruption)
    
    def get_prompt_label_pair_from_row_and_components(
        self, row: pd.Series, a: str, b: str, c: str, corruption: Optional[str] = None
    ) -> Dict:
        prompt = {}
        if corruption:
            premise_1 = f"{a} {row['Premise1_Verb']} {corruption}"
        else:
            premise_1 = f"{a} {row['Premise1_Verb']} {b}"
        premise_2 = f"{b} {row['Premise2_Verb']} {c}"
        
        conclusion_set_up = f"{c} {row['Conclusion_Verb']}"

        prompt["input"] = f"{self.begin_str}{premise_1} {self.and_str}{premise_2}{self.deduction_str}{conclusion_set_up} "
        prompt["a"] = a
        prompt["b"] = b
        prompt["c"] = c
        prompt["v1"] = row["Premise1_Verb"]
        prompt["v2"] = row["Premise2_Verb"]
        prompt["v3"] = row["Conclusion_Verb"]
        prompt["corruption"] = corruption
        prompt["labels"] = (a, b)
        return prompt
    
    def corrupt_middle_term(self) -> List[Dict]:
        """
        Replace the middle term subject in premise 2 (a2) with a random alternative.
        """
        prompts = []
        for _, row in self.samples.iterrows():
            corruption = get_filtered_sample(self.B_col, [row["Premise2_Subject"]])
            prompts.append(self.get_prompt_label_pair_from_row(row, corruption=corruption))
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
            prompts.append(self.get_prompt_label_pair_from_row_and_components(row, a, b, c))
        return prompts

    def tlen(self,text):
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def get_label_token_lengths(self, prompts):
        # init
        max_len = {lab: 0 for lab in self.labels}

        # fixed operators (unchanged text)
        max_len["BEGIN"] = self.tlen(self.begin_str)
        max_len["∧"]     = self.tlen(self.and_str)
        max_len["=>"]    = self.tlen(self.deduction_str)

        prompt_lens = []
        for prompt in prompts:
            pl = {lab: 0 for lab in self.labels}

            # fixed parts (constant per prompt)
            pl["BEGIN"] = max_len["BEGIN"]
            pl["∧"]     = max_len["∧"]
            pl["=>"]    = max_len["=>"]

            # variable parts
            a_len   = self.tlen(prompt["a"] + " ")
            v1_len  = self.tlen(prompt["v1"] + " ")
            b2_len  = self.tlen(prompt["b"] + " ")
            v2_len  = self.tlen(prompt["v2"] + " ")
            c_len  = self.tlen(prompt["c"] + " ")


            if prompt.get("corruption"):
                b1_len = self.tlen(prompt["corruption"] + " ")
            else:
                b1_len = self.tlen(prompt["b"] + " ")


            # END piece
            v3_len = self.tlen(prompt["v3"])

            # fill per-prompt
            pl["a"]  = a_len
            pl["∈"]  = v1_len
            pl["b1"] = b1_len
            pl["b2"] = b2_len
            pl["->"] = v2_len
            # store the larger since label "c" is reused
            pl["c"]  = c_len
            pl["<-"] = v3_len

            # update global maxima
            max_len["a"]  = max(max_len["a"],  a_len)
            max_len["∈"]  = max(max_len["∈"],  v1_len)
            max_len["b1"] = max(max_len["b1"], b1_len)
            max_len["b2"] = max(max_len["b2"], b2_len)
            max_len["->"] = max(max_len["->"], v2_len)
            max_len["c"]  = max(max_len["c"],  c_len)
            max_len["<-"] = max(max_len["<-"], v3_len)

            prompt_lens.append(pl)

        return prompt_lens, max_len

class ExampleAMRBuilder(BaseAMRBuilder):
    def __init__(self, df: pd.DataFrame, N: int, seed: int, tokenizer):
        super().__init__(df, N, seed, tokenizer)
        # Columns commonly used for corruptions (A,B,C refer to the classic syllogistic slots)
        self.A_col = df["Premise1_Subject"]
        self.V_col = df["Premise1_Object"]  
        self.B_col = df["Premise2_Subject"]
        self.C_col = df["Premise2_Object"]


        self.element_list = list(self.V_col) 


        self.because_str = "because"
        self.and_str = "and because"
        self.can_be_str = "can be done by" #is done when things
        self.deduction_str = "thus"
        self.example_str = "an example of"
        self.is_str = "is"

        self.labels = [
            "BEGIN",
            "a_1",
            "->",
            "v_1",
            "∧",
            "b_1",
            "v_2",
            "c",
            "EXAMPLE",
            "a_2",
            "IS",
            "b_2",
        ]


    def get_prompt_label_pair_from_row(self, row: pd.Series, corruption: Optional[str] = None) -> Dict:
        
        a = " " + row["Premise1_Subject"]
        v = " " + row["Premise1_Object"]
        b = " " + row["Premise2_Subject"]
        c = " " + row["Premise2_Object"]          
        return self.get_prompt_label_pair_from_row_and_components(row, a, b, c, v, corruption=corruption)
    
    def get_prompt_label_pair_from_row_and_components(
        self, row: pd.Series, a: str, b: str, c: str, v: str, corruption: Optional[str] = None
    ) -> Dict:
        prompt = {}
        
        premise_1 = f"{a}{self.can_be_str}{v}"

        if corruption:
            premise_2 = f"{b}{corruption}{c}"
        else:
            premise_2 = f"{b}{v}{c}"
        
        conclusion_set_up = f"{self.example_str}{a}{self.is_str}{b}{v}"

        prompt["input"] = f"{self.because_str}{premise_1} {self.and_str}{premise_2}{self.deduction_str}{conclusion_set_up}"
        prompt["a"] = a
        prompt["b"] = b
        prompt["c"] = c
        prompt["v"] = v
        prompt["corruption"] = corruption
        prompt["labels"] = (c, a)
        return prompt
    
    def corrupt_middle_term(self) -> List[Dict]:
        """
        Replace the middle term subject in premise 2 (a2) with a random alternative.
        """
        prompts = []
        for _, row in self.samples.iterrows():
            corruption = " " + get_filtered_sample(self.V_col, [row["Premise1_Object"]])
            prompts.append(self.get_prompt_label_pair_from_row(row, corruption=corruption))
        return prompts

    def corrupt_all_terms(self) -> List[Dict]:
        """
        Independently replace A, B, and C with random alternatives (one-word form).
        """
        prompts = []
        for _, row in self.samples.iterrows():
            a = " " + get_filtered_sample(self.A_col, [row["Premise1_Subject"]])
            b = " " + get_filtered_sample(self.B_col, [row["Premise2_Subject"]])
            c = " " + get_filtered_sample(self.C_col, [row["Premise2_Object"]])
            v = " " + get_filtered_sample(self.V_col, [row["Premise1_Object"]])
            prompts.append(self.get_prompt_label_pair_from_row_and_components(row, a, b, c, v))
        return prompts

    def tlen(self,text):
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def get_label_token_lengths(self, prompts):
        # seed with declared labels
        max_len = {lab: 0 for lab in self.labels}

        # fixed operators (constant across prompts)
        max_len["BEGIN"]   = self.tlen(self.because_str)
        max_len["->"]      = self.tlen(self.can_be_str)
        max_len["∧"]       = self.tlen(self.and_str)
        max_len["THUS"]    = self.tlen(self.deduction_str)  # internal helper (not in self.labels)
        max_len["EXAMPLE"] = self.tlen(self.example_str)
        max_len["IS"]      = self.tlen(self.is_str)

        prompt_lens = []
        for prompt in prompts:
            pl = {lab: 0 for lab in self.labels}

            # fixed parts
            pl["BEGIN"]   = max_len["BEGIN"]
            pl["->"]      = max_len["->"]
            pl["∧"]       = max_len["∧"]
            pl["THUS"]    = max_len["THUS"]
            pl["EXAMPLE"] = max_len["EXAMPLE"]
            pl["IS"]      = max_len["IS"]

            # variable parts from the prompt dict
            a_len  = self.tlen(prompt["a"])
            v_len  = self.tlen(prompt["v"])
            b_len  = self.tlen(prompt["b"])
            v2_len = self.tlen(prompt["corruption"]) if prompt.get("corruption") else v_len
            c_len  = self.tlen(prompt["c"])

            # fill per-prompt
            pl["a_1"] = a_len
            pl["v_1"] = v_len
            pl["b_1"] = b_len
            pl["v_2"] = v2_len
            pl["c"]   = c_len
            pl["a_2"] = a_len
            pl["b_2"] = b_len
            pl["v_3"] = v_len  # final verb in the conclusion ("... b v"); internal helper

            # update global maxima
            for k in ["a_1", "v_1", "b_1", "v_2", "c", "a_2", "b_2", "v_3"]:
                max_len[k] = max(max_len.get(k, 0), pl[k])

            prompt_lens.append(pl)

        return prompt_lens, max_len


    def get_adjusted_token_sequences(self, max_len, prompts) -> t.Tensor:
        """
        Sequence layout (labels shown; fixed tokens mapped to your strings):
        BEGIN  a_1  ->  v_1  ∧  b_1  v_2  c  THUS  EXAMPLE  a_2  IS  b_2  v_3
        """
        tokenised = []

        # fixed-token pieces
        BEGIN = self.tokenizer(self.because_str,   add_special_tokens=False)["input_ids"]
        CAN   = self.tokenizer(self.can_be_str,    add_special_tokens=False)["input_ids"]
        AND   = self.tokenizer(self.and_str,       add_special_tokens=False)["input_ids"]
        THUS  = self.tokenizer(self.deduction_str, add_special_tokens=False)["input_ids"]
        EX    = self.tokenizer(self.example_str,   add_special_tokens=False)["input_ids"]
        IS    = self.tokenizer(self.is_str,        add_special_tokens=False)["input_ids"]

        for prompt in prompts:
            seq = []

            # BEGIN
            seq += BEGIN

            # a_1
            seq += self.tokenizer(
                prompt["a"], add_special_tokens=False,
                padding="max_length", max_length=max_len["a_1"], truncation=True
            )["input_ids"]

            # ->
            seq += CAN

            # v_1
            seq += self.tokenizer(
                prompt["v"], add_special_tokens=False,
                padding="max_length", max_length=max_len["v_1"], truncation=True
            )["input_ids"]

            # ∧
            seq += AND

            # b_1
            seq += self.tokenizer(
                prompt["b"], add_special_tokens=False,
                padding="max_length", max_length=max_len["b_1"], truncation=True
            )["input_ids"]

            # v_2 (possibly corrupted)
            v2_text = prompt["corruption"] if prompt.get("corruption") else prompt["v"]
            seq += self.tokenizer(
                v2_text, add_special_tokens=False,
                padding="max_length", max_length=max_len["v_2"], truncation=True
            )["input_ids"]

            # c
            seq += self.tokenizer(
                prompt["c"], add_special_tokens=False,
                padding="max_length", max_length=max_len["c"], truncation=True
            )["input_ids"]

            # THUS EXAMPLE
            seq += THUS
            seq += EX

            # a_2
            seq += self.tokenizer(
                prompt["a"], add_special_tokens=False,
                padding="max_length", max_length=max_len["a_2"], truncation=True
            )["input_ids"]

            # IS
            seq += IS

            # b_2
            seq += self.tokenizer(
                prompt["b"], add_special_tokens=False,
                padding="max_length", max_length=max_len["b_2"], truncation=True
            )["input_ids"]

            # v_3 (final verb in the conclusion)
            seq += self.tokenizer(
                prompt["v"], add_special_tokens=False,
                padding="max_length", max_length=max_len["v_3"], truncation=True
            )["input_ids"]

            tokenised.append(seq)

        return t.tensor(tokenised, dtype=t.long)



class IfThenAMRBuilder(BaseAMRBuilder):
    def __init__(self, df: pd.DataFrame, N: int, seed: int, tokenizer):
        super().__init__(df, N, seed, tokenizer)
        # Columns commonly used for corruptions (A,B,C refer to the classic syllogistic slots)
        self.A_col = df["Premise1_Subject"]
        self.B_col = df["Premise2_Subject"]
        self.C_col = df["Premise2_Object"]


        self.element_list = list(self.A_col) + list(self.B_col) 

        self.because_str = "because a"
        self.and_str = "and because"
        self.requires_str = "requires" 
        self.if_str = "thus if there is a"
        self.then_str = "then there is a"

        self.labels = [
            "BEGIN",
            "a",
            "<-",
            "b_1",
            "∧",
            "b_2",
            "<--",
            "c_1",
            "IF",
            "c_2",
            "THEN",
        ]


    def get_prompt_label_pair_from_row(self, row: pd.Series, corruption: Optional[str] = None) -> Dict:
        
        a = " " + row["Premise1_Subject"]
        b = " " + row["Premise2_Subject"]
        c = " " + row["Premise2_Object"]          
        return self.get_prompt_label_pair_from_row_and_components(row, a, b, c, corruption=corruption)
    
    def get_prompt_label_pair_from_row_and_components(
        self, row: pd.Series, a: str, b: str, c: str, corruption: Optional[str] = None
    ) -> Dict:
        prompt = {}
        if corruption:
            premise_1 = f"{a} {self.requires_str}{corruption}"
        else:
            premise_1 = f"{a} {self.requires_str}{b}"

        premise_2 = f"{b} {self.requires_str}{c}"
        
        conclusion_set_up = f"{self.if_str}{c} {self.then_str}"

        prompt["input"] = f"{self.because_str}{premise_1} {self.and_str}{premise_2} {conclusion_set_up}"
        prompt["a"] = a
        prompt["b"] = b
        prompt["c"] = c
        prompt["corruption"] = corruption
        prompt["labels"] = (a, b)
        return prompt
    
    def corrupt_middle_term(self) -> List[Dict]:
        """
        Replace the middle term subject in premise 2 (a2) with a random alternative.
        """
        prompts = []
        for _, row in self.samples.iterrows():
            corruption = " " + get_filtered_sample(self.B_col, [row["Premise1_Object"]])
            prompts.append(self.get_prompt_label_pair_from_row(row, corruption=corruption))
        return prompts

    def corrupt_all_terms(self) -> List[Dict]:
        """
        Independently replace A, B, and C with random alternatives (one-word form).
        """
        prompts = []
        for _, row in self.samples.iterrows():
            a = " " + get_filtered_sample(self.A_col, [row["Premise1_Subject"]])
            b = " " + get_filtered_sample(self.B_col, [row["Premise2_Subject"]])
            c = " " + get_filtered_sample(self.C_col, [row["Premise2_Object"]])
            prompts.append(self.get_prompt_label_pair_from_row_and_components(row, a, b, c))
        return prompts

    def tlen(self,text):
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def get_label_token_lengths(self, prompts):
        # seed with declared labels
        max_len = {lab: 0 for lab in self.labels}

        # fixed operators (constant across prompts)
        max_len["BEGIN"] = self.tlen(self.because_str)
        max_len["<-"]    = self.tlen(self.requires_str)   # appears twice; same length
        max_len["<--"]    = self.tlen(self.requires_str) 
        max_len["∧"]     = self.tlen(self.and_str)
        max_len["IF"]    = self.tlen(self.if_str)
        max_len["THEN"]  = self.tlen(self.then_str)

        prompt_lens = []
        for prompt in prompts:
            pl = {lab: 0 for lab in self.labels}

            # fixed parts
            pl["BEGIN"] = max_len["BEGIN"]
            pl["<-"]    = max_len["<-"]
            pl["<--"]    = max_len["<--"]
            pl["∧"]     = max_len["∧"]
            pl["IF"]    = max_len["IF"]
            pl["THEN"]  = max_len["THEN"]

            # variable parts from the prompt dict
            a_len   = self.tlen(prompt["a"])
            b1_len  = self.tlen(prompt["b"])
            b2_text = prompt["corruption"] if prompt.get("corruption") else prompt["b"]
            b2_len  = self.tlen(b2_text)
            c_len   = self.tlen(prompt["c"])

            # fill per-prompt
            pl["a"]   = a_len
            pl["b_1"] = b1_len
            pl["b_2"] = b2_len
            pl["c_1"]   = c_len  # used twice in the sequence
            pl["c_2"]   = c_len

            # update global maxima
            for k in ["a", "b_1", "b_2", "c_1", "c_2"]:
                max_len[k] = max(max_len.get(k, 0), pl[k])

            prompt_lens.append(pl)

        return prompt_lens, max_len


    def get_adjusted_token_sequences(self, max_len, prompts) -> t.Tensor:
        """
        Sequence layout (labels shown; fixed tokens mapped to your strings):
        BEGIN  a  <-  b_1  ∧  b_2  <-  c  IF  c  THEN
        """
        tokenised = []

        # fixed-token pieces
        BEGIN = self.tokenizer(self.because_str, add_special_tokens=False)["input_ids"]
        REQ   = self.tokenizer(self.requires_str, add_special_tokens=False)["input_ids"]
        AND   = self.tokenizer(self.and_str,     add_special_tokens=False)["input_ids"]
        IF    = self.tokenizer(self.if_str,      add_special_tokens=False)["input_ids"]
        THEN  = self.tokenizer(self.then_str,    add_special_tokens=False)["input_ids"]

        for prompt in prompts:
            seq = []

            # BEGIN
            seq += BEGIN

            # a
            seq += self.tokenizer(
                prompt["a"], add_special_tokens=False,
                padding="max_length", max_length=max_len["a"], truncation=True
            )["input_ids"]

            # <-
            seq += REQ

            # b_1
            seq += self.tokenizer(
                prompt["b"], add_special_tokens=False,
                padding="max_length", max_length=max_len["b_1"], truncation=True
            )["input_ids"]

            # ∧
            seq += AND

            # b_2 (possibly corrupted)
            b2_text = prompt["corruption"] if prompt.get("corruption") else prompt["b"]
            seq += self.tokenizer(
                b2_text, add_special_tokens=False,
                padding="max_length", max_length=max_len["b_2"], truncation=True
            )["input_ids"]

            # <-
            seq += REQ

            # c (premise)
            seq += self.tokenizer(
                prompt["c"], add_special_tokens=False,
                padding="max_length", max_length=max_len["c_1"], truncation=True
            )["input_ids"]

            # IF
            seq += IF

            # c (condition repeated)
            seq += self.tokenizer(
                prompt["c"], add_special_tokens=False,
                padding="max_length", max_length=max_len["c_2"], truncation=True
            )["input_ids"]

            # THEN
            seq += THEN

            tokenised.append(seq)

        return t.tensor(tokenised, dtype=t.long)



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
        self.type = type

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
