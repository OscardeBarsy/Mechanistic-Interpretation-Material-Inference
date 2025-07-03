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
from global_variables import IMAGE_DIR, DATASET_DIR

bc_path = DATASET_DIR / "belief-consistent.csv"
bi_path = DATASET_DIR / "belief-inconsistent.csv"


SYLLOGISM_TEMPLATE = {
    'AAA1': 'All[A] are[B]. All[B] are[C]. Therefore, all[A] are',
    'EAE1': 'All[A] are[B]. No[B] are[C]. Therefore, no[A] are',
    'AII1': 'Some[A] are[B]. All[B] are[C]. Therefore, some[A] are',
    'EIO1': 'Some[A] are[B]. No[B] are[C]. Therefore, some[A] are not',
    'EAE2': 'All[A] are[B]. No[C] are[B]. Therefore, no[A] are',
    'AEE2': 'No[A] are[B]. All[C] are[B]. Therefore, no[A] are',
    'EIO2': 'Some[A] are[B]. No[C] are[B]. Therefore, some[A] are not',
    'AOO2': 'Some[A] are not[B]. All[C] are[B]. Therefore, some[A] are not',
    'IAI3': 'All[B] are[A]. Some[B] are[C]. Therefore, some[A] are',
    'AII3': 'Some[B] are[A]. All[B] are[C]. Therefore, some[A] are',
    'OAO3': 'All[B] are[A]. Some[B] are not[C]. Therefore, some[A] are not',
    'EIO3': 'Some[B] are[A]. No[B] are[C]. Therefore, some[A] are not',
    'AEE4': 'No[B] are[A]. All[C] are[B]. Therefore, no[A] are',
    'IAI4': 'All[B] are[A]. Some[B] are[C]. Therefore, some[A] are',
    'EIO4': 'Some[B] are[A]. No[B] are[C]. Therefore, some[A] are not',
}

ALPHABET_LIST = [' A', ' B', ' C', ' D', ' E', ' F', ' G', ' H', ' I', ' J', ' K', ' L', ' M', ' N', ' O', ' P', ' Q', ' R', ' S', ' T', ' U', ' V', ' W', ' X', ' Y', ' Z']
NUMBER_LIST = [' 1', ' 2', ' 3', ' 4', ' 5',' 6',' 7',' 8',' 9']


def gen_symbolic_prompt(N, seed=42, template_type = 'CAT'):
    random.seed(seed)
    cnt = 0
    prompts = []
    while cnt < N:
        template = SYLLOGISM_TEMPLATE[template_type]
        A = B = C = ""
        while len(set([A, B, C])) < 3:
            A = random.choice(ALPHABET_LIST)
            B = random.choice(ALPHABET_LIST)
            C = random.choice(ALPHABET_LIST)

        permutations_list = list(permutations([A, B, C]))
        for perm in permutations_list:
            prompt = {}
            prompt["A"] = perm[0]
            prompt["B"] = perm[1]
            prompt["label"] = perm[2]
            prompt["input"] = template.replace("[A]", perm[0]).replace("[B]", perm[1]).replace("[C]", perm[2])
            prompts.append(prompt)
        cnt += 1
    return prompts

def gen_numeric_prompt(N, seed=42, template_type = 'CAT'):
    random.seed(seed)
    cnt = 0
    prompts = []
   
    while cnt < N:
        template = SYLLOGISM_TEMPLATE[template_type]
        A = B = C = ""
        while len(set([A, B, C])) < 3:
            A = random.choice(NUMBER_LIST)
            B = random.choice(NUMBER_LIST)
            C = random.choice(NUMBER_LIST)

        permutations_list = list(permutations([A, B, C]))
        for perm in permutations_list:
            prompt = {}
            prompt["A"] = perm[0]
            prompt["B"] = perm[1]
            prompt["input"] = template.replace("[A]", perm[0]).replace("[B]", perm[1]).replace("[C]", perm[2])
            prompt["label"] = perm[2]
            prompts.append(prompt)
        cnt += 1
    return prompts

def gen_consistent_prompt(N, tokenizer, seed = 42,  template_type = 'CAT', sequence_length = 15):
    prompts = []
    with open(bc_path, mode='r') as file:
        cnt = 0
        csvFile = csv.reader(file)
        next(csvFile)
        for lines in csvFile:
            prompt = {}
            prompt["label"] = lines[1]
            prompt["A"] = lines[2]
            prompt["B"] = lines[3]
            template = SYLLOGISM_TEMPLATE[template_type]
            complete = template.replace('[A]', prompt["A"]).replace('[B]',prompt["B"]).replace('[C]', prompt["label"])
            prompt["input"] = complete
            prompts.append(prompt)
            cnt += 1
            if cnt == N:
                break

    return prompts

def gen_inconsistent_prompt(N, tokenizer, seed = 42,  template_type = 'CAT', sequence_length = 15):
    prompts = []
    with open(bi_path, mode='r') as file:
        cnt = 0
        csvFile = csv.reader(file)
        next(csvFile)
        for lines in csvFile:
            prompt = {}
            prompt["label"] = lines[1]
            prompt["A"] = lines[2]
            prompt["B"] = lines[3]

            template = SYLLOGISM_TEMPLATE[template_type]
            complete = template.replace('[A]', prompt["A"]).replace('[B]',prompt["B"]).replace('[C]', prompt["label"])
            prompt["input"] = complete
            prompts.append(prompt)
            cnt += 1
            if cnt == N:
                break
    return prompts


class SyllogismDataset:
    def __init__(
        self,
        seed = 0,
        N = 50,
        type = 'symbolic',
        device= 'mps',
        template_type = 'CAT',
        tokenizer= None,
    ):

        self.N = N
        self.seed = seed
        self.template_type = template_type
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device
        self.prepend_bos = False

        random.seed(self.seed)
        np.random.seed(self.seed)

        if type == 'symbolic':
            self.prompts = gen_symbolic_prompt(
                self.N, self.seed, self.template_type
            )
        elif type == 'consistent':
            self.prompts = gen_consistent_prompt( 
                self.N, self.tokenizer, self.seed,  self.template_type, sequence_length = 15
            )  
        elif type == 'inconsistent':
            self.prompts = gen_inconsistent_prompt(
                self.N, self.tokenizer, self.seed, self.template_type, sequence_length = 15
            )     
        else:
            self.prompts = gen_numeric_prompt(
                self.N, self.seed, self.template_type
            )            
    
        self.sentences = [
            prompt["input"] for prompt in self.prompts
        ]
        self.labels = [
            prompt["label"] for prompt in self.prompts
        ]
        self.A = [prompt["A"] for prompt in self.prompts]
        self.B = [prompt["B"] for prompt in self.prompts]