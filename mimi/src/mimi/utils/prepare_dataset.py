from __future__ import annotations
from enum import Enum



import torch as t
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
import random
import copy
import re
import csv
from itertools import permutations
from mimi.utils.global_variables import IMAGE_DIR, DATASET_DIR





class AMRType(str, Enum):
    ARG_SUB = "argument_substitution"
    PRED_SUB = "predicate_substitution"
    FRAME_SUB = "frame_substitution"
    COND_FRAME = "conditional_frame_insertion_substitution"
    ARG_INS = "argument_insertion"
    FRAME_CONJ = "frame_conjunction"
    ARG_PRED_GEN = "argument_or_predicate_generalisation"
    ARG_SUB_PROP = "argument_substitution_property_inheritance"
    EXAMPLE = "example"
    IFT = "if_then"
    UNK = "unknown"

class Corruption(Enum):
    NO = "no"
    MID = "middle"
    ALL = "all"



class MaterialInferenceDataset:

    def __init__(
        self,
        seed = 42,
        N = 100,
        type: AMRType = AMRType.ARG_SUB,
        corruption: Corruption = Corruption.NO,
        tokenizer= None,
    ):

        self.N = N
        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.prepend_bos = False


        self.df = pd.read_csv(f"{DATASET_DIR}/examples_100/{type.value}.csv")

        random.seed(self.seed)
        np.random.seed(self.seed)

        self.A = self.df["Premise1_Subject"]
        self.B = self.df["Premise2_Subject"]
        self.C = self.df["Premise2_Object"]


        if corruption == Corruption.NO:
            self.prompts = self.gen_prompt_label_pairs()

        elif corruption == Corruption.MID:
            self.prompts = self.corrupt_middle_term()

        elif corruption == Corruption.ALL:
            self.prompts = self.corrupt_all_terms()
        
        


        self.sentences = [
            prompt["input"] for prompt in self.prompts
        ]
        self.labels = [
            prompt["label"] for prompt in self.prompts
        ]
        self.A = [prompt["a"] for prompt in self.prompts]
        self.B = [prompt["b"] for prompt in self.prompts]


    def gen_prompt_label_pairs(self):
        prompts = []
        samples = self.df.sample(n=self.N, random_state=self.seed)
        for index, row in samples.iterrows():
            prompt = self.get_prompt_label_pair_from_row(row)
            prompts.append(prompt)
        return prompts
    
    def get_prompt_label_pair_from_row(self, row):
        a = row["Premise1_Subject"]
        b = row["Premise2_Subject"]
        c = row["Premise2_Object"]
        return self.get_prompt_label_pair_from_row_and_abc(row, a, b, c)
    
    def get_prompt_label_pair_from_row_and_abc(self, row, a, b, c):

        prompt = {}
        premise_1 = a + " " + row["Premise1_Verb"]+ " " + b
        premise_2 = b + " " + row["Premise2_Verb"] + " " + c
        conclusion_set_up = a + " " + row["Conclusion_Verb"]

        prompt["input"] = f"Since {premise_1} and {premise_2}, therefore {conclusion_set_up}"

        prompt["a"] = a
        prompt["b"] = b
        prompt["label"] = c.strip().lower()


        return prompt 

    def get_filtered_sample(self, iterable, excluded = []):
        sample_list = pd.Series(filter(lambda x: x not in excluded, iterable))
        return sample_list.sample().iloc[0]

    def corrupt_middle_term(self):
        prompts = []
        samples = self.df.sample(n=self.N, random_state=self.seed)
        for index, row in samples.iterrows():
            row["Premise2_Subject"] = self.get_filtered_sample(self.B, [row["Premise2_Subject"]])
            prompt = self.get_prompt_label_pair_from_row(row)
            prompts.append(prompt)
        return prompts
    
    def corrupt_all_terms(self):
        prompts = []
        samples = self.df.sample(n=self.N, random_state=self.seed)
        for index, row in samples.iterrows():
            a = self.get_filtered_sample(self.A, [row["Premise1_Subject"]])
            b = self.get_filtered_sample(self.B, [row["Premise2_Subject"]])
            c = self.get_filtered_sample(self.C, [row["Premise2_Object"]])
            prompt = self.get_prompt_label_pair_from_row_and_abc(row, a, b, c)
            prompts.append(prompt)
        return prompts
    

