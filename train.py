import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from model.Data import DataLoader
from model.Graph import KnowledgeGraph

import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

from datasets import load_dataset, Dataset
import transformers
from transformers import LlamaTokenizer, BitsAndBytesConfig
from torch.nn.utils.rnn import pad_sequence
from accelerate import load_checkpoint_and_dispatch

import sys
sys.path.append("..")

from model.stater import Stater
from model.stater_t5 import StaterT5
from torch.utils.tensorboard import SummaryWriter

class Tracker:
    def __init__(self, option, graph=None, device=None):
        super(Tracker, self).__init__()
        self.option = option
        if device:
            self.device = device
        else:
            if self.option.use_cuda:
                self.device =  torch.device("cuda:0")
            else:
                self.device =  torch.device("cpu")
        # quant_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # )
        if self.option.stater_type == "llama":
            self.stater = Stater.from_pretrained(
                self.option.stater_path,
                cache_dir=self.option.stater_cache_dir,
                use_safetensors=True,
                torch_dtype=self.option.compute_dtype
            )
            
            #self.stater = load_checkpoint_and_dispatch(model, self.option.stater_path, device_map="auto", no_split_module_classes=["LlamaDecoderLayer"])
        elif self.option.stater_type == "flant5-large":
            self.stater = StaterT5.from_pretrained(
                self.option.stater_path,
                cache_dir=self.option.stater_cache_dir,
                use_safetensors=True,
                torch_dtype=self.option.compute_dtype
            )
            
        elif self.option.stater_type == "bert-base":
            self.stater = StaterBert.from_pretrained(
                self.option.stater_path,
                cache_dir=self.option.stater_cache_dir,
                use_safetensors=True,
                torch_dtype=self.option.compute_dtype
            )
        else:
            raise("check you stater_type hyparameters")
        # 加载权重
        # self.stater = model.to(self.device)
        self.stater.eval()
        if self.option.instruction_type == "0":
            self.instruction = """
Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.


### Instruction:

You are now an assistant and are answering a user's Utterance. Starting with the Current Entity as the starting point, performing one or two-hop reasoning on the knowledge graph based on the query and Dialog History, and the Path History is a set of triples that consisting of [Starting Entity, Relation, Target Entity]


### Input:
Dialog History: {}
Utterance: {}
Path History: {}
Current Entity: {}
Current Step: {}


{}


### Response:\n"
"""   
        elif self.option.instruction_type == "1":
            self.instruction = """
### Task Background:
Performing one-hop reasoning on the knowledge graph.

### Instruction:
Given the Task Background and the Environment, directly output this path in triplet format without any other content.


### Environment:
Dialog History: {}
Utterance: {}
Path History: {}
Current Entity: {}
Current Step: {}


{}


### Response:
"""
        else:
            self.instruction = """
Dialog History: {}
Utterance: {}
Path History: {}
Current Entity: {}
Current Step: {}


{}
"""
        print(self.instruction)
        self.init_tokenizer()
        # self.writer = SummaryWriter(log_dir=self.option.log_dir)
        self.counter = 0
        if self.option.use_cuda:
            self.stater = self.stater.to(self.device)
        
    def init_tokenizer(self):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.option.stater_path,
                padding_side="right",
                use_fast=False,
                model_max_length=2048,
            )
        # LLaMA tokenizer 特殊处理
        if isinstance(self.tokenizer, LlamaTokenizer):
            if self.tokenizer._pad_token is None:
                smart_tokenizer_and_embedding_resize(
                    special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                    tokenizer=self.tokenizer,
                    model=self.stater,
            )
            # LLaMA tokenizer may not have correct special tokens set.
            # Check and add them if missing to prevent them from being parsed into different tokens.
            # Note that these are present in the vocabulary.
            # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
            print('Adding special tokens.')
            self.tokenizer.add_special_tokens({
                    "eos_token": self.tokenizer.convert_ids_to_tokens(self.stater.config.eos_token_id),
                    "bos_token": self.tokenizer.convert_ids_to_tokens(self.stater.config.bos_token_id),
                    "unk_token": self.tokenizer.convert_ids_to_tokens(
                        self.stater.config.pad_token_id if self.stater.config.pad_token_id != -1 else self.tokenizer.pad_token_id
                    ),
            })

    def update_state(self, dialog_history, query, path_history, current_entity, step, inp=None, target_entity=None):
        inps = []
        if inp:
            for i in inp:
                i = i[:750]
                self.counter += 1
                # self.writer.add_scalar('total/input_len{}'.format(self.option.exp_name), len(i), self.counter)
                if self.option.stater_type == "llama":
                    inps.append(f"{self.tokenizer.bos_token}{i}")
                else:
                    inps.append(i)
        else:
            for i in range(len(current_entity)):
                if dialog_history:
                    d_h = dialog_history[i]
                else:
                    d_h = []
                if path_history:
                    p_h = path_history[i]
                else:
                    p_h = []
                if step == 2:
                    step_str = "Now is the second step. Now is the second step. Now is the second step."
                else:
                    step_str = str(step)
                    
                tmp = self.instruction.format(d_h, query[i], p_h, current_entity[i], step_str, '')
                tmp = tmp[:750]

                self.counter += 1
                # self.writer.add_scalar('total/input_len{}'.format(self.option.exp_name), len(tmp), self.counter)
                # tmp = self.prompt_input_template.format(instruction=instruction[i], input=tmp)
                if self.option.stater_type == "llama":
                    inps.append(f"{self.tokenizer.bos_token}{tmp}")
                else:
                    inps.append(tmp)

        # if inp or mode=="train":
        #     tokenized_inp_with_prompt = self.tokenizer(
        #         inps,
        #         max_length=2048,
        #         truncation=True,
        #         add_special_tokens=False,
        #         return_tensors="pt"
        #     )
        #     data_dict = {
        #         'input_ids': tokenized_inp_with_prompt['input_ids'],
        #         'attention_mask':tokenized_inp_with_prompt['attention_mask'],
        #     }
        # else:
        # print(inps)
        tokenized_inp_with_prompt = self.tokenizer(
            inps,
            max_length=2048,
            truncation=True,
            add_special_tokens= not isinstance(self.tokenizer, LlamaTokenizer),
        )
        input_ids = []
        for tokenized_source in tokenized_inp_with_prompt['input_ids']:
            input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if self.option.use_cuda:
            data_dict = {k: v.to(self.stater.device) for k, v in data_dict.items()}
        return self.stater(**data_dict)

import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from pprint import pprint

from torch.distributions import Categorical
from torch.utils.data import DataLoader as DL

class Manager(nn.Module):
    def __init__(self, option, dataloader, graph, character, tracker):
        super(Manager, self).__init__()
        self.option = option
        self.character = character
        self.data_loader = dataloader
        self.graph = graph
        self.train_data = self.data_loader.get_reason_train_data()
        self.tracker = tracker


        self.start_entities = None
        self.start_relations = None
        self.current_entities = None
        self.current_relations = None
        self.target_entities = None
        # self.split_target = None
        self.contexts = None
        self.utterances = None
        # self.intent = None
        # ultra append

        self.inps = None
        self.dialog_histories = None
        self.queries = None
        self.path_histories = None

        self.start_relations_embedding = None
        self.start_entities_embedding = None
        self.target_entities_embedding = None
        
        self.reached_mask = None
        self.positive_reward = 1
        self.negative_reward = 0
        self.dones = torch.tensor([False] * 2)
        self.dws = torch.tensor([False] * 2)
        self.steps = 0
        print(f"num train data {len(self.train_data)}")
        self.data_iter = iter(DL(dataset=self.train_data, batch_size=self.option.train_batch_size, shuffle=True, collate_fn=self.collate_fn))
        # self.load()
        self.device = torch.device("cuda") if self.option.use_cuda else torch.device("cpu")
        self.epoch = 0
        self.ground_truth_step = None
        option.max_train_steps = math.ceil(len(self.train_data) * option.train_times / option.batch_size) * \
                                 (option.batch_size / option.mini_batch_size * option.K_epochs / option.gradient_accumulation) * option.epoch
        # if self.option.use_cuda:
        #     self.relation_embedding = self.relation_embedding.cuda()
        #     self.entity_embedding = self.entity_embedding.cuda()
            
    def collate_fn(self, examples):
        """
        对batch数据进行处理
        :param batch: [一个getitem的结果，getitem的结果,getitem的结果]
        :return: 元组
        """
        batch = {
            "input": [],
            "dialog_history": [],
            "query": [],
            "path_history": [],
            "current_entity": None,
            "target_entity": None,
            "step": []
        }
        current_entities = []
        target_entities = []
        # pprint(examples)
        for item in examples:
            # pprint(item)
            batch["input"].append(item["input"])
            batch["dialog_history"].append(item["dialog_history"])
            batch["query"].append(item["query"])
            batch["path_history"].append(item["path_history"])
            current_entities.append(item["current_entity"])
            target_entities.append(item["target_entity"])
            batch["step"].append(item["step"])
        batch["current_entity"] = torch.LongTensor(current_entities)
        if self.option.dataset == "OpenDialKG":
            batch["target_entity"] = torch.LongTensor(target_entities)
        else:
            batch["target_entity"] = target_entities
        return batch

    def reset(self):
        with torch.no_grad():
            try:
                batch = next(self.data_iter)
                self.current_entities = batch["current_entity"]
                self.target_entities = batch["target_entity"]
                if self.character == "Assistant" or self.character == "User" or self.character == "Random" or self.character == "Reason":
                    self.dialog_histories = batch["dialog_history"]
                    self.queries = batch["query"]
                    self.path_histories = batch["path_history"]
                    self.inps = batch["input"]
                    self.ground_truth_step = batch["step"]
            except StopIteration:
                self.data_iter = iter(DL(dataset=self.train_data, batch_size=self.option.train_batch_size, shuffle=True, collate_fn=self.collate_fn))
                self.epoch += 1
                batch = next(self.data_iter)
                self.current_entities = batch["current_entity"]
                self.target_entities = batch["target_entity"]
                if self.character == "Assistant" or self.character == "User" or self.character == "Random" or self.character == "Reason":
                    self.dialog_histories = batch["dialog_history"]
                    self.queries = batch["query"]
                    self.path_histories = batch["path_history"]
                    self.inps = batch["input"]
                    self.ground_truth_step = batch["step"]
                    
            self.current_entities = self.current_entities.repeat_interleave(self.option.train_times, dim=0)
            if self.option.dataset == "OpenDialKG":
                self.target_entities = self.target_entities.repeat_interleave(self.option.train_times, dim=0)
            else:
                self.target_entities = [item for s in self.target_entities for item in [s]*self.option.train_times]
            # self.current_entities = [item for s in self.current_entities for item in [s]*self.option.train_times]
            # self.target_entities = [item for s in self.target_entities for item in [s]*self.option.train_times]
            self.dialog_histories = [item for s in self.dialog_histories for item in [s]*self.option.train_times]
            self.queries = [item for s in self.queries for item in [s]*self.option.train_times]
            self.path_histories = [item for s in self.path_histories for item in [s]*self.option.train_times]
            self.inps = [item for s in self.inps for item in [s]*self.option.train_times]
            self.ground_truth_step = [item for s in self.ground_truth_step for item in [s]*self.option.train_times]
            # print(self.inps)
            # from pprint import pprint
            # pprint(self.dialog_histories)
            # pprint(self.path_histories)
            
            self.real_batch_size = len(self.queries)
            step_embedding = self.get_step_embedding(0, self.real_batch_size)
            if self.option.use_cuda:
                self.current_entities = self.current_entities.to(self.tracker.device)
                if self.option.dataset == "OpenDialKG":
                    self.target_entities = self.target_entities.to(self.tracker.device)
                step_embedding = step_embedding.to(self.tracker.device)
            # self.target_entities_embedding = self.entity_embedding(self.target_entities)
            
            # pprint(self.current_entities)
            state_queries = self.tracker.update_state(
                dialog_history=None,
                query=None,
                path_history=None,
                current_entity=None,
                step=1,
                inp=self.inps
            )
            state_queries = state_queries + step_embedding
            
            self.done = False
            self.dw = False
            self.steps = 0
            # print(self.current_entities)

        return state_queries
    
    def GetPosEncodingMatrix(self, max_len, d_emb):
        # 位置编码
        pos_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0 else np.zeros(d_emb)
            for pos in range(max_len)
        ])
        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
        return pos_enc

    def get_step_embedding(self, step, real_batch_size):
        np_pos_embedding = self.GetPosEncodingMatrix(self.option.max_out, self.option.state_embed_size)[step]
        # if step % 2 == 0:
        #     step_base = torch.sin(torch.tensor([step], dtype=self.option.compute_dtype))
        # else:
        #     step_base = torch.cos(torch.tensor([step], dtype=self.option.compute_dtype))
            
        # step_embedding = torch.ones(real_batch_size, self.option.state_embed_size, dtype=self.option.compute_dtype) * step_base
        step_embedding = torch.tensor(np_pos_embedding, dtype=self.option.compute_dtype)
        return step_embedding
    
    def cal_reward(self, next_entity):
        # print(next_relation)
        # print(self.data_loader.relation2num["Equal"])
        # print(next_entity.item())
        # print(self.split_target[self.steps].item())

        # if self.option.type == "Reason":
        #     # print(self.split_target)
        #     # print(self.split_target[0] == self.split_target[1])
        #     # print(next_relation.item() == self.data_loader.relation2num["equal"])
        #     # encourage stop at step 2
        #     if self.steps == 1 and self.split_target[0] == self.split_target[1] and next_relation.item() == \
        #             self.data_loader.relation2num["equal"]:
        #         return 200
        #     if next_entity == self.split_target[self.steps]:
        #         return 1
        #     else:
        #         if self.steps == 0 and self.start_entity == next_entity:
        #             return -200
        #         if self.steps == 1 and self.split_target[0] == self.split_target[1] and next_relation.item() != \
        #                 self.data_loader.relation2num["equal"]:
        #             return -200
        #         return -1
        # else:
        wc = self.option.coh_weight
        ws = self.option.sim_weight
        c_embed = self.entity_embedding(self.current_entity)
        n_embed = self.entity_embedding(next_entity)
        coh_cos_dis = F.cosine_similarity(c_embed, n_embed)
        if coh_cos_dis >= 0.5:
            coherence_reward = 1
        else:
            coherence_reward = -1
        tar_cos_dis_old = F.cosine_similarity(c_embed, self.target_entity_embedding)
        tar_cos_dis_new = F.cosine_similarity(n_embed, self.target_entity_embedding)
        if tar_cos_dis_new >= tar_cos_dis_old:
            similarity_reward = 1
        else:
            similarity_reward = -1
        # print(f"coherence_reward: {coherence_reward}\tsimilarity_reward: {similarity_reward}")
        cs_reward = wc * coherence_reward + ws * similarity_reward

        return cs_reward

    def step(self, action):
        with torch.no_grad():
            current_entities = self.current_entities
            # print(current_entities)
            # print(action)
            # print(self.target_entities)
            next_relations, next_entities = self.graph.get_nexts(self.current_entities, action)
            # print(next_entities)
            # state_query = self.tracker.update_state(
            #     instruction=self.instruction,
            #     dialog_history=self.dialog_history,
            #     query=self.query,
            #     path_history=self.path_history,
            #     current_entity=self.data_loader.num2entity[self.current_entity],
            #     inp=self.inp
            # )
            # if next_entity == self.target_entity:
            #     self.done = True
            #     self.dw = True
            # reward = self.cal_reward(next_entity)
            self.steps += 1
            # if self.steps >= self.option.train_step_length:
            #     self.dones = True
            #     if next_entity == self.target_entity:
            #         self.dws = True
            
            # current_entity_str = self.data_loader.num2entity[current_entity.item()]
            # next_relation_str = self.data_loader.num2relation[next_relation.item()]
            # next_entity_str = self.data_loader.num2entity[next_entity.item()]
            # self.path_history.append(f"{current_entity_str},{next_relation_str},{next_entity_str}")
            
            # print(action_prob)
            # print(self.instruction_arr)
            # print(self.dialog_history_arr)
            # print(self.query_arr)
            # print(self.path_history)
            # print(self.path_history_arr)
            # print(chosen_entities_arr)
            current_entities_arr = [""] * len(next_entities)
            next_relations_arr = [""] * len(next_entities)
            next_entities_arr = [""] * len(next_entities)
            for i in range(len(next_entities)):
                # print(self.current_entities[i])
                # print(chosen_relations[i])
                # print(chosen_entities[i])
                # print(chosen_entities[i].item())
                # print(self.data_loader.num2entity[chosen_entities[i].item()])
                current_entity_str = self.data_loader.num2entity[self.current_entities[i].item()]
                next_relation_str = self.data_loader.num2relation[next_relations[i].item()]
                next_entity_str = self.data_loader.num2entity[next_entities[i].item()]
                
                current_entities_arr[i] = current_entity_str
                next_relations_arr[i] = next_relation_str
                if self.path_histories[i]:
                    if not isinstance(self.path_histories[i], list):
                        self.path_histories[i] = list(self.path_histories[i])
                    self.path_histories[i].append(f"{current_entity_str},{next_relation_str},{next_entity_str}")
                else:
                    self.path_histories[i] = [f"{current_entity_str},{next_relation_str},{next_entity_str}"]
                
                next_entities_arr[i] = next_entity_str
                
            # print(next_relations)
            # print(next_entities)
            # print(current_entities_arr)
            # print(next_relations_arr)
            # print(next_entities_arr)
            # print("\n\n\n\n")
            step_embedding = self.get_step_embedding(self.steps, self.real_batch_size)
            if self.option.use_cuda:
                step_embedding = step_embedding.to(self.tracker.device)
                
            state_queries = self.tracker.update_state(
                dialog_history=self.dialog_histories,
                query=self.queries,
                path_history=self.path_histories,
                current_entity=next_entities_arr,
                step=self.steps+1
            )
            state_queries = state_queries + step_embedding
            self.current_entities = next_entities
            
        return state_queries, current_entities, self.target_entities, next_entities

    def get_final_reward(self):
        rewards = []
        
        for i in range(len(self.current_entities)):
            if self.option.dataset == "MetaQA":
                if self.current_entities[i].cpu().item() in self.target_entities[i]:
                    rewards.append(1)
                else:
                    rewards.append(-1)
            else:
                if self.current_entities[i] == self.target_entities[i]:
                    rewards.append(1)
                else:
                    rewards.append(-1)
        return rewards
        
    def clear_gpu(self):
        self.start_entities = None
        self.start_relations = None
        self.current_entities = None
        self.current_relations = None
        self.target_entities = None
        # self.split_target = None
        self.contexts = None
        self.utterances = None
        # self.intent = None
        # ultra append
        self.inps = None
        self.dialog_histories = None
        self.queries = None
        self.path_histories = None

        self.start_relations_embedding = None
        self.start_entities_embedding = None
        self.target_entities_embedding = None
        
        self.reached_mask = None
        self.dones = torch.tensor([False] * 2)
        self.dws = torch.tensor([False] * 2)
        self.steps = 0
        self.ground_truth_step = None
        
    def choose_action(self, agent, state, step=None, mask=None):
        with torch.no_grad():
            if state.device !=  agent.device:
                state =  state.to(agent.device)
            plm = agent.actor(state, self.current_entities, step)
            # real_a_p = agent.get_real_scores(plm, self.current_entities, step)
            dist = Categorical(logits=plm)
            action_id = dist.sample()
            # if mask != None:
            #     action_id[mask] = 0
            # action_id = torch.multinomial(real_a_p, 1)
            log_action_prob = dist.log_prob(action_id)

        return action_id, log_action_prob

    def evaluate(self, agent, s):  # When evaluating the policy, we select the action with the highest probability
        with torch.no_grad():
            plm = agent.actor(s)
            a_prob = self.get_real_scores(plm, self.current_entity)
            a = torch.argmax(a_prob, dim=1)
            chosen_action = self.graph.get_action(self.current_entity, a)
            next_relation, next_entity = self.graph.get_next(self.current_entity, a)
        return a, chosen_action, next_relation, next_entity

    def get_cos_similar(self, v1: list, v2: list):
        num = float(np.dot(v1, v2))  # 向量点乘
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
        return 0.5 + 0.5 * (num / denom) if denom != 0 else 0

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical
from collections import OrderedDict

# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    # nn.init.constant_(layer.bias, 0)


class Actor(nn.Module):
    def __init__(self, option, state_dim, hidden_dim, graph, device, use_bias=False):
        super(Actor, self).__init__()
        self.option = option
        self.graph = graph
        self.device = device
        self.fc1 = nn.Linear(state_dim, hidden_dim, bias=use_bias)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.activate_func = nn.Tanh()
        if self.option.out_path_aware:
            dim = self.option.relation_embed_size + self.option.entity_embed_size
        else:
            dim = self.option.max_out
        self.actor = nn.Linear(hidden_dim, dim, bias=use_bias)
        # self.actor = nn.Sequential(OrderedDict([
        #                 ('fc1', nn.Linear(5120, 5120, bias=False)),
        #                  ('silu1', nn.SiLU()),
        #                   ('fc2', nn.Linear(5120, 5120, bias=False)),
        #                ('silu2', nn.SiLU()),
        #                 ('score', nn.Linear(5120, 400, bias=False)),
        #                ]))
        print("random initialized path embedding...")
        self.relation_embedding = nn.Embedding(self.option.num_relation, self.option.relation_embed_size)
        torch.nn.init.xavier_uniform_(self.relation_embedding.weight)
        self.entity_embedding = nn.Embedding(self.option.num_entity, self.option.entity_embed_size)
        torch.nn.init.xavier_uniform_(self.entity_embedding.weight)
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.actor, gain=0.01)
    
    def load_pretrained_embeddings(self):
        path_entity = self.option.entity_embedding_path
        path_relation = self.option.relation_embedding_path
        print(f"loading pretrained entity embeddings from:{path_entity}")
        # entity_embedding = np.load(path_entity)
        print(f"loading pretrained relation embeddings from:{path_relation}")
        # relation_embedding = np.load(path_relation)
        self.entity_embedding = torch.nn.Embedding.from_pretrained(torch.load(path_entity), freeze=True).to(self.device)
        self.relation_embedding = torch.nn.Embedding.from_pretrained(torch.load(path_relation), freeze=True).to(self.device)
    
    def get_real_scores(self, action_logit, current_entities, step=None):
        if self.option.out_path_aware:
            # pprint(current_entities)
            # pprint(current_entities.view(-1))
            actions_entities = self.graph.get_out(current_entities.view(-1).cpu())
            if actions_entities.device != action_logit.device:
                actions_entities = actions_entities.to(action_logit.device)
            # print(actions_entities)
            # pprint(actions_entities)
            out_relations_id = actions_entities[:, :, 0]
            # print(out_relations_id)
            out_entities_id = actions_entities[:, :, 1]
            # print(out_entities_id)
            out_entities_embedding = self.entity_embedding(out_entities_id)
            out_relations_embedding = self.relation_embedding(out_relations_id)
            path_embedding = torch.cat([out_relations_embedding, out_entities_embedding], -1)
            # path_embedding = out_relations_embedding + out_entities_embedding
            # pprint(action_logit)
            # pprint(action_logit.shape)
            # pprint(path_embedding.shape)
            prelim_scores = torch.sum(torch.mul(action_logit.unsqueeze(1), path_embedding), dim=-1)
            dummy_entities_id = torch.ones_like(out_entities_id, dtype=torch.int64) * 100718
            # dummy_entities_id = torch.ones_like(out_entities_id, dtype=torch.int64) * 43234
            mask = torch.eq(out_entities_id, dummy_entities_id)
            if step == 1 and self.option.dataset == "OpenDialKG":
                mask[:, 0] = True
            dummy_scores = torch.ones_like(prelim_scores) * (-99999)
            prelim_scores = torch.where(mask, dummy_scores, prelim_scores)
            # action_logit = torch.softmax(prelim_scores, dim=1)
            action_logit = prelim_scores 
            
        return action_logit

    def forward(self, s, ce=None, step=None):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        a_prob = self.actor(s)
        if ce is not None:
            a_prob = self.get_real_scores(a_prob, ce, step)
        return a_prob

class Critic(nn.Module):
    def __init__(self, option, state_dim, hidden_dim, use_bias=False):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim, bias=use_bias)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.activate_func = nn.Tanh()
        self.critic = nn.Linear(hidden_dim, 1, bias=use_bias)
        # self.actor = nn.Sequential(OrderedDict([
        #                 ('fc1', nn.Linear(5120, 5120, bias=False)),
        #                  ('silu1', nn.SiLU()),
        #                   ('fc2', nn.Linear(5120, 5120, bias=False)),
        #                ('silu2', nn.SiLU()),
        #                 ('score', nn.Linear(5120, 400, bias=False)),
        #                ]))
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.critic, gain=0.01)
    
    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.critic(s)
        return v_s
    
# class Critic(nn.Module):
#     def __init__(self, option, state_dim):
#         super(Critic, self).__init__()
#         self.actor = nn.Sequential(OrderedDict([
#                         ('fc1', nn.Linear(5120, 5120, bias=False)),
#                          ('silu1', nn.SiLU()),
#                           ('fc2', nn.Linear(5120, 5120, bias=False)),
#                        ('silu2', nn.SiLU()),
#                         ('critic', nn.Linear(5120, 1, bias=False)),
#                        ]))
        
#     def forward(self, s):
#         a_prob = self.actor(s)
#         return a_prob

class PPO:
    def __init__(self, option, character, graph):
        self.option = option
        self.character = character
        self.batch_size = option.batch_size
        self.graph = graph
        self.mini_batch_size = option.mini_batch_size
        self.max_train_steps = option.max_train_steps
        self.lr_a = option.lr_a  # Learning rate of actor
        self.lr_c = option.lr_c  # Learning rate of critic
        self.gamma = option.gamma  # Discount factor
        self.lamda = option.lamda  # GAE parameter
        self.epsilon = option.epsilon  # PPO clip parameter
        self.K_epochs = option.K_epochs  # PPO parameter
        self.entropy_coef = option.entropy_coef  # Entropy coefficient
        self.set_adam_eps = option.set_adam_eps
        self.use_grad_clip = option.use_grad_clip
        self.use_lr_decay = option.use_lr_decay
        self.use_adv_norm = option.use_adv_norm
        if self.option.use_cuda:
            self.device =  torch.device("cuda:0")
        else:
            self.device =  torch.device("cpu")
        self.actor = Actor(option, self.option.state_embed_size, self.option.hidden_dim, self.graph, self.device, self.option.use_bias)
        self.critic = Critic(option, self.option.state_embed_size, self.option.hidden_dim, self.option.use_bias)
        if option.compute_dtype == torch.bfloat16:
            self.actor.to(torch.bfloat16)
            self.critic.to(torch.bfloat16)
        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)
        lam = lambda f: 1 - f / self.max_train_steps
        self.actor_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer_actor, lr_lambda=lam)
        self.critic_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer_critic, lr_lambda=lam)
        self.update_times = 0
        self.mini_step = 0
        # self.load_pretrained_embeddings()
        # 首次加载 sft 后的参数
        if os.path.exists(self.option.actor_path):
            self.load()
        # 加载 rl 训练过的参数
        if os.path.exists(self.option.actor_checkpoint_path):
            self.load_all()
        if self.option.use_trans_e:
            self.actor.load_pretrained_embeddings()            
        if option.use_cuda:
            self.actor = self.actor.to(self.device)
            # if self.option.mode=="train":
            self.critic = self.critic.to(self.device)

#     def choose_action(self, state, step, mask=None):
#         with torch.no_grad():
#             plm = self.actor(state)
#             real_a_p = self.get_real_scores(plm, self.current_entities, step)
#             dist = Categorical(probs=real_a_p)
#             action_id = dist.sample()
#             # if mask != None:
#             #     action_id[mask] = 0
#             # action_id = torch.multinomial(real_a_p, 1)
#             log_action_prob = dist.log_prob(action_id)

#         return action_id, log_action_prob
    
#     def get_real_scores(self, action_prob, current_entities, step=None):
#         # pprint(current_entities)
#         # pprint(current_entities.view(-1))
#         actions_entities = self.graph.get_out(current_entities.view(-1).cpu())
#         if actions_entities.device != action_prob.device:
#             actions_entities = actions_entities.to(action_prob.device)
#         # print(actions_entities)
#         # pprint(actions_entities)
#         out_relations_id = actions_entities[:, :, 0]
#         # print(out_relations_id)
#         out_entities_id = actions_entities[:, :, 1]
#         # print(out_entities_id)
#         out_entities_embedding = self.entity_embedding(out_entities_id)
#         out_relations_embedding = self.relation_embedding(out_relations_id)
#         path_embedding = torch.cat([out_relations_embedding, out_entities_embedding], -1)
#         # path_embedding = out_relations_embedding + out_entities_embedding
#         # pprint(action_prob)
#         # pprint(action_prob.shape)
#         # pprint(path_embedding.shape)
#         prelim_scores = torch.sum(torch.mul(action_prob.unsqueeze(1), path_embedding), dim=-1)
#         dummy_entities_id = torch.ones_like(out_entities_id, dtype=torch.int64) * 100718
#         # dummy_entities_id = torch.ones_like(out_entities_id, dtype=torch.int64) * 43234
#         mask = torch.eq(out_entities_id, dummy_entities_id)
#         if step == 1 and self.option.dataset == "OpenDialKG":
#             mask[:, 0] = True
#         dummy_scores = torch.ones_like(prelim_scores) * (-99999)
#         prelim_scores = torch.where(mask, dummy_scores, prelim_scores)
#         action_prob = torch.softmax(prelim_scores, dim=1)

#         return action_prob
    
#     def load_pretrained_embeddings(self):
#         path_entity = self.option.entity_embedding_path
#         path_relation = self.option.relation_embedding_path
#         print(f"loading pretrained entity embeddings from:{path_entity}")
#         # entity_embedding = np.load(path_entity)
#         print(f"loading pretrained relation embeddings from:{path_relation}")
#         # relation_embedding = np.load(path_relation)
#         self.entity_embedding = torch.nn.Embedding.from_pretrained(torch.load(path_entity), freeze=True).to(self.device)
#         self.relation_embedding = torch.nn.Embedding.from_pretrained(torch.load(path_relation), freeze=True).to(self.device)
        
    def update(self, replay_buffer, manager):
        s, a, a_log_prob, r, s_, dw, done, ce = replay_buffer.numpy_to_tensor()  # Get training data
        if self.option.use_cuda:
            s = s.to(self.device)
            a = a.to(self.device)
            a_log_prob = a_log_prob.to(self.device)
            r = r.to(self.device)
            ce = ce.to(self.device)
            s_ = s_.to(self.device)
            dw = dw.to(self.device)
            done = done.to(self.device)
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().cpu().tolist()), reversed(done.flatten().cpu().tolist())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype = self.option.compute_dtype).view(-1, 1)
            if self.option.use_cuda:
                adv = adv.to(self.device)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for K epochs:
        actor_buffer = []
        critic_buffer = []
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                self.mini_step += 1
                plm = self.actor(s[index], ce[index])
                # real_sm = self.get_real_scores(plm, ce[index])
                dist_now = Categorical(logits=plm)
                dist_entropy = dist_now.entropy().view(-1, 1)  # shape(mini_batch_size X 1)
                a_log_prob_now = dist_now.log_prob(a[index].squeeze()).view(-1, 1)  # shape(mini_batch_size X 1)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_log_prob_now - a_log_prob[index])  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_log_prob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # shape(mini_batch_size X 1)
                self.option.writer.add_scalar('train/actor_loss', actor_loss.mean().float(), self.update_times)

                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                self.option.writer.add_scalar('train/critic_loss', critic_loss.float(), self.update_times)
                
                # actor_buffer.append(actor_loss.mean())
                # critic_buffer.append(critic_loss)
                # total_loss = actor_loss.mean() + critic_loss
                # if len(actor_buffer) == self.option.gradient_accumulation:
                # Update actor
                self.optimizer_actor.zero_grad()
                # actor_loss = torch.stack(actor_buffer)
                # actor_loss = torch.mean(actor_loss)
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.option.writer.add_scalar('train/actor_learning_rate', self.optimizer_actor.param_groups[0]['lr'], self.update_times)
                self.optimizer_actor.step()
                self.actor_scheduler.step()

                # Update critic
                self.optimizer_critic.zero_grad()
                # critic_loss = torch.stack(critic_buffer)
                # critic_loss = torch.mean(critic_loss)
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.option.writer.add_scalar('train/critic_learning_rate', self.optimizer_critic.param_groups[0]['lr'], self.update_times)
                self.optimizer_critic.step()
                self.critic_scheduler.step()
                # actor_buffer = []
                # critic_buffer = []
                self.update_times += 1
                    
                    
                    
                # else:
                #     actor_buffer.append(actor_loss.mean())
                #     critic_buffer.append(critic_loss)

        # if self.use_lr_decay:  # Trick 6:learning rate Decay
        #     self.lr_decay(self.update_times)
        return self.update_times

    # def lr_decay(self, total_steps):
    #     lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
    #     lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
    #     for p in self.optimizer_actor.param_groups:
    #         p['lr'] = lr_a_now
    #     for p in self.optimizer_critic.param_groups:
    #         p['lr'] = lr_c_now

    def check_optimizer(self, optimizer):
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

    def load(self):
        actor_model_path = self.option.actor_path
        print(f"loading actor form: {actor_model_path}")
        self.actor.load_state_dict(torch.load(actor_model_path), strict=False)
        self.critic.load_state_dict(torch.load(actor_model_path), strict=False)
        # print(f"loading critic form: {actor_model_path}")
        # self.critic.load_state_dict(torch.load(actor_model_path), strict=False)

    def load_all(self):
        checkpoint_path = self.option.actor_checkpoint_path
        print(f"loading checkpoint form: {checkpoint_path}")
        s_d = torch.load(checkpoint_path)
        self.update_times = s_d["update_times"]
        self.actor.load_state_dict(s_d["actor"])
        self.critic.load_state_dict(s_d["critic"])
        self.optimizer_actor.load_state_dict(s_d["optimizer_actor"])
        self.optimizer_critic.load_state_dict(s_d["optimizer_critic"])
        self.actor_scheduler.load_state_dict(s_d["actor_scheduler"])
        self.critic_scheduler.load_state_dict(s_d["critic_scheduler"])
        # move to gpu
        self.check_optimizer(self.optimizer_actor)
        self.check_optimizer(self.optimizer_critic)
        # self.check_optimizer(self.actor_scheduler)
        # self.check_optimizer(self.critic_scheduler)
    
    def save(self):
        agent_save_path = os.path.join(self.option.log_dir, "actor.pkt")
        print(f"saving model at: {agent_save_path}")
        torch.save({
            'update_times': self.update_times,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer_actor': self.optimizer_actor.state_dict(),
            'optimizer_critic': self.optimizer_critic.state_dict(),
            'actor_scheduler': self.actor_scheduler.state_dict(),
            'critic_scheduler': self.critic_scheduler.state_dict(),
        }, agent_save_path)

        
import copy
import os
import torch
from torch.utils.data import DataLoader as DL
import torch.nn.functional as F
from tricks.normalization import Normalization
from tqdm import tqdm
from pprint import pprint
import time
class Tester:
    def __init__(self, option, manager, agent, character):
        self.manager = manager
        self.option = option
        self.agent = agent
        self.character = character
        self.data_loader = manager.data_loader
        self.option.num_entity = self.data_loader.num_entity
        self.option.num_relation = self.data_loader.num_relation
        self.graph = manager.graph
        if self.option.mode == "train":
            self.test_data = self.data_loader.get_reason_valid_data()
            self.num_test_data = len(self.test_data)
            print(f"num valid data {self.num_test_data}")
        else:
            self.test_data = self.data_loader.get_reason_test_data()
            self.num_test_data = len(self.test_data)
            print(f"num test data {self.num_test_data}")   
        self.tracker = manager.tracker
        self.relation_embedding = agent.actor.relation_embedding
        self.entity_embedding = agent.actor.entity_embedding

        self.start_relation = self.get_dummy_start_relation(1)
        self.start_relation_embedding = None

        self.current_entity = None

        self.start_entity = None
        self.start_entity_embedding = None

        self.context = None
        self.contexts = None
        self.utterance = None
        self.utterances = None
        
        # ultra append
        self.inp = None
        self.dialog_history = None
        self.query = None
        self.path_history = None

        self.dialog_history_arr = None
        self.query_arr = None
        self.path_history_arr = None
        
        # self.split_target = []
        self.target_entity = None
        self.target_entities = None
        self.target_entity_embedding = None
        self.target_entities_embedding = None

        self.path = None

        # self.intent = None
        # self.intents = None
        # self.intent_embedding = None
        # self.intents_embedding = None
        self.pads = torch.ones(self.option.test_times) * self.data_loader.entity2num["Pad"]
        self.state_query = None
        self.reached_mask = None
        self.positive_reward = 1
        self.negative_reward = 0
        self.done = False
        self.dw = False
        self.steps = 0
        self.data_iter = iter(DL(dataset=self.test_data, batch_size=1, shuffle=True))
        self.init_log_current_prob = torch.zeros(1)
        self.paths_e = None
        self.paths_r = None
        self.father_e = None
        self.agent.actor.eval()
        
        # if self.option.use_state_norm:
        #     self.state_norm = Normalization(shape=self.option.state_embed_size)
        if self.option.use_cuda:
            # self.relation_embedding = self.relation_embedding.to(self.agent.device)
            # self.entity_embedding = self.entity_embedding.to(self.agent.device)
            self.init_log_current_prob = self.init_log_current_prob.to(self.agent.device)
            self.pads = self.pads.to(self.agent.device)
            
    def GetPosEncodingMatrix(self, max_len, d_emb):
        # 位置编码
        pos_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0 else np.zeros(d_emb)
            for pos in range(max_len)
        ])
        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
        return pos_enc

    def get_step_embedding(self, step, real_batch_size):
        np_pos_embedding = self.GetPosEncodingMatrix(self.option.max_out, self.option.state_embed_size)[step]
        # if step % 2 == 0:
        #     step_base = torch.sin(torch.tensor([step], dtype=self.option.compute_dtype))
        # else:
        #     step_base = torch.cos(torch.tensor([step], dtype=self.option.compute_dtype))
            
        # step_embedding = torch.ones(real_batch_size, self.option.state_embed_size, dtype=self.option.compute_dtype) * step_base
        step_embedding = torch.tensor(np_pos_embedding, dtype=self.option.compute_dtype)
        return step_embedding
    
    def get_dummy_start_relation(self, batch_size):
        dummy_start_item = self.data_loader.relation2num["Start"]
        dummy_start = torch.ones(batch_size, dtype=torch.int64) * dummy_start_item
        return dummy_start
        
    def collate_fn(self, examples):
        """
        对batch数据进行处理
        :param batch: [一个getitem的结果，getitem的结果,getitem的结果]
        :return: 元组
        """
        batch = {
            "input": [],
            "dialog_history": [],
            "query": [],
            "path_history": [],
            "current_entity": None,
            "target_entity": None,
            "action": None
        }
        current_entities = []
        target_entities = []
        actions = []
        # pprint(examples)
        for item in examples:
            # pprint(item)
            batch["input"].append(item["input"])
            batch["dialog_history"].append(item["dialog_history"])
            batch["query"].append(item["query"])
            batch["path_history"].append(item["path_history"])
            current_entities.append(item["current_entity"])
            target_entities.append(item["target_entity"])

        batch["current_entity"] = torch.LongTensor(current_entities)
        if self.option.dataset == "OpenDialKG":
            batch["target_entity"] = torch.LongTensor(target_entities)
        else:
            batch["target_entity"] = target_entities
        return batch
        
    def yield_next(self):
        data_iter = iter(DL(self.test_data, batch_size=1))
        # c = 0
        while True:
            try:
                batch = next(data_iter)
                self.start_entity = batch["current_entity"]
                self.target_entity = batch["target_entity"]
                self.dialog_history = batch["dialog_history"]
                self.query = batch["query"]
                self.path_history = batch["path_history"]
                self.path = batch["path"]
                # if self.path_history:
                #     self.path_history[0] = list(self.path_history[0])
                self.inp = batch["input"]
                # c += 1
                # if c >= 100:
                #     break
            except StopIteration:
                break
            yield 0

    def reset(self, st: list = None):
        with torch.no_grad():
            if st is not None:
                self.start_entity = torch.LongTensor([st[0]])
                self.target_entity = torch.LongTensor([st[1]])
                if self.character == "Reason" or self.character == "User" or self.character == "Assistant":
                    if self.option.ablation == "Context":
                        self.context = st[2]
                    elif self.option.ablation == "Utterance":
                        self.utterance = st[2]
                    elif self.option.ablation == "Proposed":
                        pass
                    else:
                        raise Exception("Please check your ablation settings!")
            self.log_current_prob = self.init_log_current_prob
            # print(self.instruction)
            # print(self.path_history)
            real_batch_size = 1
            step_embedding = self.get_step_embedding(0, real_batch_size)
            
            self.state_query = self.tracker.update_state(
                dialog_history=None,
                query=None,
                path_history=None,
                current_entity=None,
                step=1,
                inp=self.inp
            )
            # if self.option.use_state_norm:
            #     self.state_query = self.state_norm(self.state_query, update=False)
                
            if self.option.use_cuda:
                self.start_relation = self.start_relation.to(self.agent.device)
                self.start_entity = self.start_entity.to(self.agent.device)
                self.target_entity = self.target_entity.to(self.agent.device)
                step_embedding = step_embedding.to(self.tracker.device)
                # self.state_query = self.state_query.to(self.agent.device)

            self.state_query = self.state_query + step_embedding
            self.current_entity = self.start_entity
            # self.start_relation_embedding = self.relation_embedding(self.start_relation)
            # self.start_entity_embedding = self.entity_embedding(self.start_entity)
            # self.target_entity_embedding = self.entity_embedding(self.target_entity)


            self.steps = 0
            self.done = False
            self.dw = False
            self.start_relations = self.start_relation.repeat_interleave(self.option.test_times, dim=0)
            self.target_entities = self.target_entity.repeat_interleave(self.option.test_times, dim=0)
            # self.target_entities_embedding = self.entity_embedding(self.target_entities)
            self.current_entities = self.current_entity.repeat_interleave(self.option.test_times, dim=0)
            self.paths_e = self.current_entities
            self.paths_r = self.start_relations
            self.father_e = self.paths_e
            
            
            # arr
            if self.dialog_history:
                self.dialog_history_arr = self.dialog_history * self.option.test_times
            else:
                self.dialog_history_arr = [self.dialog_history] * self.option.test_times
            self.query_arr = self.query * self.option.test_times
            if self.path_history:
                self.path_history_arr = self.path_history * self.option.test_times
            else:
                self.path_history_arr = [self.path_history] * self.option.test_times
            # pprint(self.dialog_history_arr)
            # pprint(self.query_arr)
            # pprint(self.path_history_arr)
    
    def test_once(self):
        with torch.no_grad():
            pbar = tqdm(total=len(self.test_data))
            # pbar = tqdm(total=5)
            data_iter = iter(DL(dataset=self.test_data, batch_size=self.option.test_batch_size, shuffle=True, collate_fn=self.collate_fn))
            all_final_reward_1 = 0
            all_final_reward_0 = 0
            all_action_reward_1 = 0
            count = 0
            action_distribution = [0] * self.option.max_out
            while True:
                try:
                    batch = next(data_iter)
                    current_entities = batch["current_entity"]
                    target_entities= batch["target_entity"]
                    dialog_histories = batch["dialog_history"]
                    queries = batch["query"]
                    path_histories = batch["path_history"]
                    # if self.path_history:
                    #     self.path_history[0] = list(self.path_history[0])
                    inps = batch["input"]
                    step = 1
                    state_queries = None
                    # torch.cuda.empty_cache()
                    real_batch_size = len(queries)
                    step_embedding = self.get_step_embedding(step - 1, real_batch_size)
                    state_queries = self.tracker.update_state(
                        dialog_history=dialog_histories,
                        query=queries,
                        path_history=path_histories,
                        current_entity=current_entities,
                        step=step,
                        inp=inps,
                    )
                    # if self.option.use_cuda:
                    #     state_queries = state_queries.to(self.agent.device)
                    # if self.option.use_state_norm:
                    #     state_queries = self.state_norm(state_queries, update=False)
                    done = False
                    # mask = current_entities.eq(target_entities)
                    if self.option.use_cuda:
                        current_entities = current_entities.to(self.agent.device)
                        if self.option.dataset == "OpenDialKG":
                            target_entities = target_entities.to(self.agent.device)
                        step_embedding = step_embedding.to(self.tracker.device)
                        
                    state_queries = state_queries + step_embedding
                        
                    start = time.time()
                    # print(current_entities)
                    # print(target_entities)
                    for i in range(self.option.max_step_length):
                        if state_queries.device != self.agent.device:
                            state_queries = state_queries.to(self.agent.device)
                        plm = self.agent.actor(state_queries, current_entities, step)
                        # a_prob = self.agent.get_real_scores(plm, 
                        action = torch.argmax(plm, dim=1)
                        for action_item in action:
                            #self.option.writer.add_scalar('valid/test_action_distribution{}'.format(self.option.exp_name), action_item.item(), action_id)
                            action_distribution[action_item.item()] += 1
                        # if i ==0:
                        #     action = torch.argmax(a_prob, dim=1)
                        #     # print(action)
                        # else:
                        #     action = torch.tensor([0] * a_prob.shape[0]).cuda()

                        # max_step_length = 1
                        # print(action)
                        # print(current_entities)
                        # print(actions)
                        # print("\n")
                        stater_start = time.time()
                        # all_action_reward_1 += action.eq(actions).long().sum().item()
                        next_relations, next_entities = self.graph.get_nexts(current_entities, action)
                        # if self.option.use_cuda:
                        #     next_relations = next_relations.to(self.agent.device)
                        #     next_entities = next_entities.to(self.agent.device)
                        stater_end = time.time()
                        # print(stater_end - stater_start)
                        # print(next_entities)
                        # print(target_entities)
                        
                        if self.option.dataset == "OpenDialKG":
                            reward = next_entities.eq(target_entities).long().sum().item()
                            if i==0:
                                all_final_reward_0 += reward
                            elif i==1:
                                all_final_reward_1 += reward
                            else:
                                raise("not implement yet!")
                            # print(f"{i}: {reward}")
                        # reward[mask] = True
                        # mask = reward
                        if i < self.option.max_step_length - 1:
                            # print(i)
                            # current_entities_arr = []
                            current_entities_arr = [""] * len(next_entities)
                            next_relations_arr = [""] * len(next_entities)
                            next_entities_arr = [""] * len(next_entities)
                            for i in range(len(next_entities)):
                                # print(self.current_entities[i])
                                # print(chosen_relations[i])
                                # print(chosen_entities[i])
                                # print(chosen_entities[i].item())
                                # print(self.data_loader.num2entity[chosen_entities[i].item()])
                                current_entity_str = self.data_loader.num2entity[current_entities[i].item()]
                                next_relation_str = self.data_loader.num2relation[next_relations[i].item()]
                                next_entity_str = self.data_loader.num2entity[next_entities[i].item()]

                                if path_histories[i]:
                                    if not isinstance(path_histories[i], list):
                                        path_histories[i] = list(path_histories[i])
                                    path_histories[i].append(f"{current_entity_str},{next_relation_str},{next_entity_str}")
                                else:
                                    path_histories[i] = [f"{current_entity_str},{next_relation_str},{next_entity_str}"]
                                # current_entities_arr.append(current_entity_str)
                                current_entities_arr[i] = current_entity_str
                                next_relations_arr[i] = next_relation_str
                                next_entities_arr[i] = next_entity_str
                                
                        
                            # print(current_entities_arr)
                            # print(next_relations_arr)
                            # print(next_entities_arr)
                    
                            state_queries = None
                            # torch.cuda.empty_cache()
                            step += 1
                            step_embedding = self.get_step_embedding(step - 1, real_batch_size)
                            state_queries = self.tracker.update_state(
                                dialog_history=dialog_histories,
                                query=queries,
                                path_history=path_histories,
                                current_entity=next_entities_arr,
                                step=step
                            )
                            
                            if self.option.use_cuda:
                                step_embedding = step_embedding.to(self.tracker.device)
                                
                            state_queries = state_queries + step_embedding
                            
                            # if self.option.use_state_norm:
                            #     state_queries = self.state_norm(state_queries, update=False)
     
                        current_entities = next_entities

                    # mask = current_entities.eq(target_entities)
                    # final_reward_1 = mask.long().sum().item()

                    # all_final_reward_1 += final_reward_1
                    end = time.time()
                    # print(end-start)
                    # print(final_reward_1)
                    if self.option.dataset == "MetaQA":
                        for i, v in enumerate(current_entities.cpu()):
                            if v.item() in target_entities[i]:
                                all_final_reward_0 += 1
                    pbar.update(self.option.test_batch_size)
                    # count += 1
                    # if count >= 20:
                    #     break
                    # sub_count -= 40
                except StopIteration:
                    break
            result = all_final_reward_1 / len(self.test_data)
            # resilt_action = all_action_reward_1 / len(self.test_data)
            state_queries = None
            current_entities = None
            target_entities = None
            # 清空GPU缓存
            # torch.cuda.empty_cache()
            with open(os.path.join(self.option.log_dir, "action_distribution.txt"), "w", encoding='UTF-8') as f:
                json.dump(action_distribution, f, ensure_ascii=True, indent=4)
            return result, all_final_reward_0 / len(self.test_data)
            
#     def test_llama(self):
#         with torch.no_grad():
#             pbar = tqdm(total=len(self.test_data))
#             # pbar = tqdm(total=5)
#             data_iter = iter(DL(dataset=self.test_data, batch_size=24, shuffle=True, collate_fn=self.collate_fn))
#             all_final_reward_1 = 0
#             all_action_reward_1 = 0
#             # sub_count = 200
#             while True:
#                 try:
#                     batch = next(data_iter)
#                     current_entities = batch["current_entity"]
#                     target_entities= batch["target_entity"]
#                     dialog_histories = batch["dialog_history"]
#                     queries = batch["query"]
#                     path_histories = batch["path_history"]
#                     # if self.path_history:
#                     #     self.path_history[0] = list(self.path_history[0])
#                     inps = batch["input"]
#                     # state_queries = 
#                     done = False
#                     mask = current_entities.eq(target_entities)
#                     if self.option.use_cuda:
#                         current_entities = current_entities
#                         target_entities = target_entities
                        
#                     a_prob = self.tracker.update_state(
#                         dialog_history=dialog_histories,
#                         query=queries,
#                         path_history=path_histories,
#                         current_entity=current_entities,
#                         step=1,
#                         inp=inps
#                         # current_entities=current_entities
#                     )
#                     # plm, _ = self.agent.actor(state_queries)
#                     # a_prob = self.manager.tracker.stater(state_queries)
#                     action = torch.argmax(a_prob, dim=1)
#                     # max_step_length = 1
#                     # print(action)
#                     # print(current_entities)
#                     # print(actions)
#                     # print("\n")
#                     # all_action_reward_1 += action.eq(actions).long().sum().item()
#                     next_relations, next_entities = self.graph.get_nexts(current_entities, action)
#                     # print(next_entities)
#                     # print(target_entities)
#                     reward = next_entities.eq(target_entities)
#                     print(f"{i}: {reward.long().sum().item()}")
#                     # print(reward)
#                     reward[mask] = True
#                     mask = reward
#                     current_entities_arr = []
#                     next_entities_arr = []
#                     for i in range(len(next_entities)):
#                         # print(self.current_entities[i])
#                         # print(chosen_relations[i])
#                         # print(chosen_entities[i])
#                         # print(chosen_entities[i].item())
#                         # print(self.data_loader.num2entity[chosen_entities[i].item()])
#                         current_entity_str = self.data_loader.num2entity[current_entities[i].item()]
#                         next_relation_str = self.data_loader.num2relation[next_relations[i].item()]
#                         next_entity_str = self.data_loader.num2entity[next_entities[i].item()]
                        
#                         if len(path_histories) != 0:
#                             if not isinstance(path_histories[i], list):
#                                 path_histories[i] = list(path_histories[i])
#                             path_histories[i].append(f"{current_entity_str},{next_relation_str},{next_entity_str}")
#                         else:
#                             path_histories.append([f"{current_entity_str},{next_relation_str},{next_entity_str}"])
#                         next_entities_arr.append(next_entity_str)
#                         current_entities_arr.append(current_entity_str)
#                     # print(path_histories)
#                     # print(current_entities_arr)
#                     # print(next_entities_arr)
#                     # print(next_paths)
#                     # state_queries = self.tracker.update_state(
#                     #     instruction=instructions,
#                     #     dialog_history=dialog_histories,
#                     #     query=queries,
#                     #     path_history=path_histories,
#                     #     current_entity=next_entities_arr,
#                     # )
#                     # current_entities = next_entities
#                     final_reward_1 = mask.long().sum().item()
#                     all_final_reward_1 += final_reward_1
#                     pbar.update(24)
#                     # sub_count -= 40
#                 except StopIteration:
#                     break
#             result = all_final_reward_1 / len(self.test_data)
#             resilt_action = all_action_reward_1 / len(self.test_data)
#             return result, resilt_action
            
    def test(self):
        with torch.no_grad():
            all_path_recall_1 = 0
            all_path_recall_3 = 0
            all_path_recall_5 = 0
            all_path_recall_10 = 0
            all_path_recall_25 = 0

            all_final_reward_1 = 0
            all_final_reward_3 = 0
            all_final_reward_5 = 0
            all_final_reward_10 = 0
            all_final_reward_25 = 0
            all_r_rank = 0

            all_coherence_reward = 0
            all_similarity_reward = 0
            pbar = tqdm(total=len(self.test_data))
            with open(os.path.join(self.option.log_dir, "paths_log.txt"), "w", encoding='UTF-8') as f:
                f.write("Start Test:")
                f.write("\n\n")
                for _ in self.yield_next():
                    torch.cuda.empty_cache()
                    self.reset()
                    done = False
                    all_paths = []
                    mask = self.current_entities.eq(self.target_entities)
                    mask_pad = self.current_entities.eq(self.pads)
                    f.write(
                        f"Start Entity: {self.data_loader.num2entity[self.start_entity.item()]}\t\t\t\t\t\tTarget Entity: {self.data_loader.num2entity[self.target_entity.item()]}")
                    f.write("\n\n")
                    # recall = mask
                    self.paths_e = self.current_entities.unsqueeze(dim=0)
                    self.paths_r = self.start_relations.unsqueeze(dim=0)
                    while not done:
                        self.steps += 1
                        # print(self.state_query.device)
                        action_prob = self.agent.actor(self.state_query)
                        if self.steps == 1:
                            chosen_state, chosen_entities, chosen_relations = self.step(action_prob,
                                                                                        self.current_entity)
                        else:
                            chosen_state, chosen_entities, chosen_relations = self.step(action_prob,
                                                                                        self.current_entities)
                        # relation_embedding = self.relation_embedding(chosen_relations)
                        # entity_embedding = self.entity_embedding(chosen_entities)
                        chosen_entities_arr = [""] * len(chosen_entities)
                        # print(action_prob)
                        # print(self.instruction_arr)
                        # print(self.dialog_history_arr)
                        # print(self.query_arr)
                        # print(self.path_history)
                        # print(self.path_history_arr)
                        # print(chosen_entities_arr)
                        # no_path_history = len(self.path_history_arr) == 0
                        for i in range(len(chosen_entities)):
                            # print(self.current_entities[i])
                            # print(chosen_relations[i])
                            # print(chosen_entities[i])
                            # print(chosen_entities[i].item())
                            # print(self.data_loader.num2entity[chosen_entities[i].item()])
                            current_entity_str = self.data_loader.num2entity[self.current_entities[i].item()]
                            next_relation_str = self.data_loader.num2relation[chosen_relations[i].item()]
                            next_entity_str = self.data_loader.num2entity[chosen_entities[i].item()]
                            
                            if self.path_history_arr[i]:
                                if not isinstance(self.path_history_arr[i], list):
                                    self.path_history_arr[i] = list(self.path_history_arr[i])
                                self.path_history_arr[i].append(f"{current_entity_str},{next_relation_str},{next_entity_str}")
                            else:
                                self.path_history_arr[i] = [f"{current_entity_str},{next_relation_str},{next_entity_str}"]
                            
                            chosen_entities_arr[i] = next_entity_str
                            
                        # print(self.path_history_arr) 
                        # print(self.state_query)
                        # print(action_prob)
                        # print(self.instruction_arr)
                        # print(self.dialog_history_arr)
                        # print(self.query_arr)
                        # print(self.path_history_arr)
                        # print(chosen_entities_arr)
                        
                        self.state_query = self.tracker.update_state(
                            dialog_history=self.dialog_history_arr,
                            query=self.query_arr,
                            path_history=self.path_history_arr,
                            current_entity=chosen_entities_arr,
                            step = self.steps+1
                        )
                        real_batch_size = len(chosen_entities)
                        step_embedding = self.get_step_embedding(self.steps, real_batch_size)
                        if self.option.use_cuda:
                            step_embedding = step_embedding.to(self.tracker.device)
                        self.state_query = self.state_query + step_embedding
                        # if self.option.use_state_norm:
                        #     self.state_query = self.state_norm(self.state_query, update=False)
                            
                        # if self.option.use_cuda and self.state_query.device != self.agent.device:
                        #     self.state_query = self.state_query.to(self.agent.device)
                        
                        reward = chosen_entities.eq(self.target_entities)
                        # if reward.long().sum().item() > 0:
                        #     done = True
                        reward[mask_pad] = False
                        mask_pad_process = chosen_entities.eq(self.pads)
                        mask_pad[mask_pad_process] = True
                        self.paths_e = torch.cat([self.paths_e, chosen_entities.unsqueeze(dim=0)])
                        self.paths_r = torch.cat([self.paths_r, chosen_relations.unsqueeze(dim=0)])
                        reward[mask] = True
                        if self.steps >= 2:
                            done = True
                        mask = reward
                        self.current_entities = chosen_entities

                    self.paths_e = self.paths_e.T[~mask_pad]
                    self.paths_r = self.paths_r.T[~mask_pad]
                    start = self.start_entity.item()
                    for path_e, path_r in zip(self.paths_e, self.paths_r):
                        paths = []
                        for step in range(len(path_r)):
                            relation_id = path_r[step].item()
                            entity_id = path_e[step].item()
                            if step > 0:
                                f.write(f"{self.data_loader.num2relation[relation_id]}\t\t")
                                paths.append([start, relation_id, entity_id])
                            f.write(f"{self.data_loader.num2entity[entity_id]}\t\t")
                            # if entity_id == self.target_entity.item():
                            #     break
                            start = entity_id
                        all_paths.append(paths)
                        f.write("\n")
                        f.write("\n")

                    # try:
                    #     c_r, s_r = self.cal_coherence_similarity_reward(all_paths[0])
                    # except IndexError:
                    #     c_r, s_r = -1, -1
                    # all_coherence_reward += c_r
                    # all_similarity_reward += s_r

                    if self.character == "Reason" or self.character == "User" or self.character == "Assistant":
                        for pos, v in enumerate(mask):
                            if v.item() is True:
                            # print(self.target_entity.item())
                            # print(p)
                            # print(self.data_loader.num2entity[p[-1].item()])
                            # print(self.data_loader.num2entity[self.target_entity.item()])
                            # print("\n")
                            # if p[-1] == self.target_entity:
                                # print(self.target_entity.item())
                                # print(p)
                                # print(self.data_loader.num2entity[p[-1].item()])
                                # print(self.data_loader.num2entity[self.target_entity.item()])
                                # print("\n")
                                if pos < 25:
                                    # print("25+")
                                    all_final_reward_25 += 1
                                    if pos < 10:
                                        # print("10+")
                                        all_final_reward_10 += 1
                                        if pos < 5:
                                            # print("5+")
                                            all_final_reward_5 += 1
                                            if pos < 3:
                                                # print("3+")
                                                all_final_reward_3 += 1
                                                if pos < 1:
                                                    # print("1+")
                                                    all_final_reward_1 += 1
                                break
                            else:
                                all_r_rank += 1.0 / (pos + 1)
                        for pos, p in enumerate(all_paths):
                            # print(p)
                            # print(self.path)
                            # print(p[:2] == self.path)
                            # print("\n")
                            # print("\n")
                            # print("\n")
                            # print("\n")
                            if p[:2] == self.path:
                                # print("find path")
                                if pos < 25:
                                    # print("25+")
                                    all_path_recall_25 += 1
                                    if pos < 10:
                                        # print("10+")
                                        all_path_recall_10 += 1
                                        if pos < 5:
                                            # print("5+")
                                            all_path_recall_5 += 1
                                            if pos < 3:
                                                # print("3+")
                                                all_path_recall_3 += 1
                                                if pos < 1:
                                                    # print("1+")
                                                    all_path_recall_1 += 1
                                break
                    else:
                        if self.character == "Target":
                            for pos, p in enumerate(all_paths):
                                # print(p)
                                # print(p[:path_len])
                                # print(p)
                                # print(self.path)
                                for i, t in enumerate(p):
                                    if t[2] == self.target_entity.item():
                                        # print("finded!")
                                        break
                                # print(i)
                                # print(p[i][2])
                                # print(p[:i+1])
                                # exit(0)
                                if p[:i+1] == self.path:
                                    # print("find path")
                                    if pos < 25:
                                        # print("25+")
                                        all_path_recall_25 += 1
                                        if pos < 10:
                                            # print("10+")
                                            all_path_recall_10 += 1
                                            if pos < 5:
                                                # print("5+")
                                                all_path_recall_5 += 1
                                                if pos < 3:
                                                    # print("3+")
                                                    all_path_recall_3 += 1
                                                    if pos < 1:
                                                        # print("1+")
                                                        all_path_recall_1 += 1
                                    break
                        for pos, v in enumerate(mask):
                            if v.item() is True:
                                if pos < 25:
                                    # print("25+")
                                    all_final_reward_25 += 1
                                    if pos < 10:
                                        # print("10+")
                                        all_final_reward_10 += 1
                                        if pos < 5:
                                            # print("5+")
                                            all_final_reward_5 += 1
                                            if pos < 3:
                                                # print("3+")
                                                all_final_reward_3 += 1
                                                if pos < 1:
                                                    # print("1+")
                                                    all_final_reward_1 += 1
                                break
                            else:
                                all_r_rank += 1.0 / (pos + 1)
                    pbar.update(1)
            all_coherence_reward /= self.num_test_data
            all_similarity_reward /= self.num_test_data

            if self.character != "MultiHop":
                all_path_recall_1 /= self.num_test_data
                all_path_recall_3 /= self.num_test_data
                all_path_recall_5 /= self.num_test_data
                all_path_recall_10 /= self.num_test_data
                all_path_recall_25 /= self.num_test_data

            all_final_reward_1 /= self.num_test_data
            all_final_reward_3 /= self.num_test_data
            all_final_reward_5 /= self.num_test_data
            all_final_reward_10 /= self.num_test_data
            all_final_reward_25 /= self.num_test_data
            all_r_rank /= self.num_test_data

            with open(os.path.join(self.option.log_dir, "test_log.txt"), "a", encoding='UTF-8') as f:
                f.write("#" * 60 + "\n")
                f.write(f"all_coherence_reward:\t{all_coherence_reward}\n")
                f.write(f"all_similarity_reward:\t{all_similarity_reward}\n")

                if self.character != "MultiHop":
                    f.write(f"all_path_recall_1:\t{all_path_recall_1}\n")
                    f.write(f"all_path_recall_3:\t{all_path_recall_3}\n")
                    f.write(f"all_path_recall_5:\t{all_path_recall_5}\n")
                    f.write(f"all_path_recall_10:\t{all_path_recall_10}\n")
                    f.write(f"all_path_recall_25:\t{all_path_recall_25}\n\n")

                f.write(f"all_final_reward_1:\t{all_final_reward_1}\n")
                f.write(f"all_final_reward_3:\t{all_final_reward_3}\n")
                f.write(f"all_final_reward_5:\t{all_final_reward_5}\n")
                f.write(f"all_final_reward_10:\t{all_final_reward_10}\n")
                f.write(f"all_final_reward_25:\t{all_final_reward_25}\n")
                f.write(f"all_r_rank:\t{all_r_rank}\n")
                f.write("\n\n")

            # print(f"all_final_reward_1:{all_final_reward_1}", )
            # print(f"all_final_reward_3:{all_final_reward_3}", )
            # print(f"all_final_reward_5:{all_final_reward_5}", )
            # print(f"all_final_reward_10:{all_final_reward_10}", )
            # print(f"all_final_reward_25:{all_final_reward_25}", )
            # print(f"all_r_rank:{all_r_rank}", )

        return all_final_reward_1

    def reason(self, steps, rank=True, intent=None):
        with torch.no_grad():
            dw = False
            if self.character != "Reason":
                mask = self.current_entities.eq(self.target_entities)
            mask_pad = self.current_entities.eq(self.pads)
            self.paths_e = self.current_entities.unsqueeze(dim=0)
            self.paths_r = self.start_relations.unsqueeze(dim=0)
            while self.steps < steps:
                self.steps += 1
                action_prob, _ = self.agent.actor(self.state_query)
                if self.steps == 1:
                    chosen_state, chosen_entities, chosen_relations = self.step(action_prob,
                                                                                self.current_entity)
                else:
                    chosen_state, chosen_entities, chosen_relations = self.step(action_prob,
                                                                                self.current_entities)
                relation_embedding = self.relation_embedding(chosen_relations)
                entity_embedding = self.entity_embedding(chosen_entities)
                
                # self.state_query = self.state_norm(self.state_query)

                self.paths_e = torch.cat([self.paths_e, chosen_entities.unsqueeze(dim=0)])
                self.paths_r = torch.cat([self.paths_r, chosen_relations.unsqueeze(dim=0)])
                if self.character == "Reason" or self.character == "User" or self.character == "Assistant":
                    mask_pad_process = chosen_entities.eq(self.pads)
                    mask_pad[mask_pad_process] = True
                else:
                    reward = chosen_entities.eq(self.target_entities)
                    reward[mask_pad] = False
                    mask_pad_process = chosen_entities.eq(self.pads)
                    mask_pad[mask_pad_process] = True
                    reward[mask] = True
                    mask = reward
                    if reward.long().sum().item() > 0:
                        dw = True
                        break
                    self.current_entities = chosen_entities

                # print("当前实体：")
                # print(self.graph.get_out(self.current_entities[~mask_pad][0]))
                # print("选择的动作：")
                # print(chosen_relations[~mask_pad][0])
                # print("下一个实体：")
                # print(chosen_entities)
                # print(mask_pad)
                # print("不是PAD的实体：")
                # print(chosen_entities[~mask_pad])
                # print("第一个不是PAD的实体：")
                # print(chosen_entities[~mask_pad][0])
            target = self.target_entity.item()
            paths = []
            if dw:
                self.paths_e = self.paths_e.T[reward]
                self.paths_r = self.paths_r.T[reward]
            else:
                self.paths_e = self.paths_e.T[~mask_pad]
                self.paths_r = self.paths_r.T[~mask_pad]
            if len(self.paths_e) == 0 or len(self.paths_r) == 0:
                return []
            max_turn = 0
            if rank:
                max_score = 0
                for i, path in enumerate(self.paths_e):
                    score = 0
                    if intent is not None:
                        # print(path[1])
                        # print(intent)
                        # exit(0)
                        # print(self.paths_r[i])
                        # print(self.paths_r[i][1].item())
                        # print(self.paths_r[i][1].item() == intent)
                        # exit(0)
                        if self.paths_r[i][1].item() == intent:
                            score += 1
                    temp_score = []
                    for c, p in enumerate(path[1:]):
                        temp_score.append(F.cosine_similarity(self.entity_embedding(p), self.target_entity_embedding))
                        if p == target:
                            break
                    score += torch.mean(torch.stack(temp_score))
                    if score > max_score:
                        max_score = score
                        max_turn = i

            path_e = self.paths_e[max_turn]
            path_r = self.paths_r[max_turn]

            start = self.start_entity.item()
            for step in range(1, self.steps + 1):
                relation_id = path_r[step].item()
                entity_id = path_e[step].item()
                paths.append((start, relation_id, entity_id))
                start = entity_id
            return paths

    def cal_coherence_similarity_reward(self, path: list):
        c_reward = 0
        s_reward = 0
        len_path = 0
        for p in path:
            len_path += 1
            h = torch.LongTensor([p[0]])
            t = torch.LongTensor([p[2]])
            if self.option.use_cuda:
                h = h.to(self.agent.device)
                t = t.to(self.agent.device)
            h_e = self.entity_embedding(h)
            t_e = self.entity_embedding(t)
            c_cs = F.cosine_similarity(h_e, t_e)
            if c_cs >= 0.5:
                c_reward += 1
            else:
                c_reward += -1
            ht_cs = F.cosine_similarity(h_e, self.target_entity_embedding)
            tt_cs = F.cosine_similarity(t_e, self.target_entity_embedding)
            if tt_cs >= ht_cs:
                s_reward += 1
            else:
                s_reward += -1
            if t == self.target_entity:
                break
        if len_path != 0:
            c_reward /= len_path
            s_reward /= len_path
        return c_reward, s_reward

    def step(self, action_logit, current_entities):
        actions_entities = self.graph.get_out(current_entities.view(-1).cpu())
        out_relations_id = actions_entities[:, :, 0]
        out_entities_id = actions_entities[:, :, 1]
        if self.option.out_path_aware:
            out_entities_embedding = self.entity_embedding(out_entities_id)
            out_relations_embedding = self.relation_embedding(out_relations_id)
            path_embedding = torch.cat([out_relations_embedding, out_entities_embedding], -1)
            # path_embedding = out_relations_embedding + out_entities_embedding
            prelim_scores = torch.sum(torch.mul(action_logit.unsqueeze(1), path_embedding), dim=-1)
            dummy_relations_id = torch.ones_like(out_relations_id, dtype=torch.int64) * self.data_loader.relation2num["Pad"]
            mask = torch.eq(out_relations_id, dummy_relations_id)
            if self.steps == 1:
                mask[:, 0] = True
            dummy_scores = torch.ones_like(prelim_scores) * (-99999)
            scores = torch.where(mask, dummy_scores, prelim_scores)
        else:
            scores = action_logit
        action_prob = torch.softmax(scores, dim=1)
        log_action_prob = torch.log(action_prob)
        chosen_state, chosen_relation, chosen_entities = self.test_search(
            self.state_query,
            log_action_prob,
            out_relations_id,
            out_entities_id,
            1
        )

        return chosen_state, chosen_entities, chosen_relation

    def test_search(self, new_state, log_action_prob, out_relations_id, out_entities_id, batch_size):
        log_current_prob = self.log_current_prob.repeat_interleave(self.option.max_out).view(batch_size, -1)
        log_action_prob = log_action_prob.view(batch_size, -1)
        log_trail_prob = torch.add(log_action_prob, log_current_prob)
        top_k_log_prob, top_k_action_id = torch.topk(log_trail_prob, self.option.test_times)

        new_state = new_state.repeat_interleave(self.option.max_out) \
            .view(batch_size, -1, self.option.state_embed_size)

        out_relations_id = out_relations_id.view(batch_size, -1)
        out_entities_id = out_entities_id.view(batch_size, -1)
        #print(f"steps\t{self.steps}")
        #print(top_k_action_id)
        #print(self.paths_e)
        if self.steps != 1 and self.option.mode != "train":
            paths_e_cp = copy.deepcopy(self.paths_e)
            paths_r_cp = copy.deepcopy(self.paths_r)
            for i, v in enumerate(top_k_action_id.squeeze()):
                #print(i)
                #print(v)
                fa = torch.div(v, self.option.max_out, rounding_mode='floor')
                #print(fa)
                for j in range(len(self.paths_e)):
                    paths_e_cp[j][i] = self.paths_e[j][fa]
                    paths_r_cp[j][i] = self.paths_r[j][fa]
            self.paths_e = paths_e_cp
            self.paths_r = paths_r_cp
        chosen_relation = torch.gather(out_relations_id, dim=1, index=top_k_action_id).view(-1)
        chosen_entities = torch.gather(out_entities_id, dim=1, index=top_k_action_id).view(-1)
        self.log_current_prob = torch.gather(log_trail_prob, dim=1, index=top_k_action_id).view(-1)

        top_k_action_id_state = top_k_action_id.unsqueeze(2).repeat(1, 1, self.option.state_embed_size)
        chosen_state = torch.gather(new_state, dim=1, index=top_k_action_id_state).view(-1, self.option.state_embed_size)

        return chosen_state, chosen_relation, chosen_entities

import numpy as np
import torch


from model.replaybuffer import ReplayBuffer
# from test import Tester
from tqdm import tqdm


# from tricks.normalization import Normalization
class Trainer:
    def __init__(self, option, manager, agent, character):
        self.option = option
        self.manager = manager
        self.agent = agent
        self.character = character

    def train(self):
        np.random.seed(self.option.seed)
        self.option.max_episode_steps = self.option.train_step_length  # Maximum number of steps per episode
        evaluate_num = 0  # Record the number of evaluations
        total_steps = 0  # Record the total steps during the training
        replay_buffer = ReplayBuffer(self.option, self.option.state_embed_size)
        train_times = 0
        tester = Tester(self.option, self.manager, self.agent, self.character)
        if os.path.exists(self.option.actor_checkpoint_path) or os.path.exists(self.option.actor_path):
            all_final_reward_1, all_final_reward_0 = tester.test_once()
        else:
            all_final_reward_1, all_final_reward_0 = 0.0, 0.0
        # all_final_reward_1, all_final_reward_0 = tester.test_once()
        # 清空GPU缓存
        torch.cuda.empty_cache()
        #all_final_reward_1 = 0.03
        self.option.writer.add_scalar('valid/step_rewards_0{}'.format(self.option.exp_name), all_final_reward_0, evaluate_num)
        self.option.writer.add_scalar('valid/step_rewards_1{}'.format(self.option.exp_name), all_final_reward_1, evaluate_num)
        pbar = tqdm(total=self.agent.max_train_steps)
        update_times = self.agent.update_times
        # state_norm = Normalization(shape=self.option.state_embed_size)  # Trick 2:state normalization
        if update_times > 0:
            pbar.update(update_times)
        max_reward = all_final_reward_1
        impatience = 0
        pbar.set_postfix(impatience=impatience, times=evaluate_num, m_reward=max_reward,
                         reward=all_final_reward_1)
        pbar_buffer = tqdm(total=self.option.batch_size)
        samples_per_explore_per_agent = self.option.train_times * 2
        sample_inds = 0
        
        while self.agent.update_times < self.agent.max_train_steps:
            pbar.set_description("Epoch %s" % str(self.manager.epoch + 1))
            
            torch.cuda.empty_cache()
            s = self.manager.reset()
            # if self.option.use_state_norm:
            #     # print(s.device)
            #     # print(state_norm.mean.device)
            #     s = state_norm(s)

            episode_steps = 0
            done = False
            batch_size = s.shape[0]
            # print(batch_size)
            all_s =  np.zeros([batch_size, self.option.train_step_length], dtype=np.int32).tolist()
            all_a = np.zeros([batch_size, self.option.train_step_length], dtype=np.int32).tolist()
            all_log = np.zeros([batch_size, self.option.train_step_length], dtype=np.int32).tolist()
            all_s_ = np.zeros([batch_size, self.option.train_step_length], dtype=np.int32).tolist()
            all_dw = np.zeros([batch_size, self.option.train_step_length], dtype=np.int32).tolist()
            all_done = np.zeros([batch_size, self.option.train_step_length], dtype=np.int32).tolist()
            all_ce = np.zeros([batch_size, self.option.train_step_length], dtype=np.int32).tolist()
            process_reward = np.zeros([batch_size, self.option.train_step_length], dtype=np.int32).tolist()
            all_r = []
            arr_achieve = [False] * batch_size
            equal_arr=[False] * batch_size
            old_size = replay_buffer.get_size()
            # print(all_a)
            with torch.no_grad():
                for i in range(self.option.train_step_length):
                    episode_steps += 1
                    mask = None
                    # if i == 1:
                    #     mask = torch.tensor(arr_achieve)
                    a, a_log_prob = self.manager.choose_action(self.agent, s, i+1)
                    # print(a)
                        # a[mask] = 0
                    s_, ce, te, ne = self.manager.step(a)
                    # if self.option.use_state_norm:
                    #     s_ = state_norm(s_)
                    # print(s)
                    # print(s_)
                    # print(final_reward)
                    # print(done)
                    # print(dw)
                    # print(ce)
                    # print(s_.shape[0])
                    
                    # if i == 0:
                    #     for j in range(s_.shape[0]):
                    #         # print("hh")
                    #         if replay_buffer.get_size() < self.option.batch_size:
                    #             # print(f"{ce[j]}\t{a[j]}\t{ne[j]}\t{te[j]}")
                    #             # if a[j] == 0:
                    #             #     # print("equal")
                    #             #     equal_arr[j] = True
                    #             #     replay_buffer.store(
                    #             #         s[j].cpu(),
                    #             #         a[j].cpu(),
                    #             #         a_log_prob[j].cpu(),
                    #             #         -1 * self.option.gamma,
                    #             #         s_[j].cpu(),
                    #             #         False,
                    #             #         False,
                    #             #         ce[j].cpu()
                    #             #     )
                    #             #     continue
                    #             # print(ne[j])
                    #             # print(te[j])
                    #             # print(ne[j] in te[j])
                    #             # if ne[j].item() in te[j]:
                    #             if ne[j] == te[j]:
                    #                 arr_achieve[j] = True
                    #                 replay_buffer.store(
                    #                     s[j].cpu(),
                    #                     a[j].cpu(),
                    #                     a_log_prob[j].cpu(),
                    #                     1 * self.option.gamma,
                    #                     s_[j].cpu(),
                    #                     True,
                    #                     True,
                    #                     ce[j].cpu()
                    #             )
                    #                 self.option.writer.add_scalar('train/reward_per_sample_{}'.format(self.option.exp_name), 1 * self.option.gamma, sample_inds)
                    #                 sample_inds += 1
                    #             # else:
                    #             #     replay_buffer.store(
                    #             #         s[j].cpu(),
                    #             #         a[j].cpu(),
                    #             #         a_log_prob[j].cpu(),
                    #             #         -1,
                    #             #         s_[j].cpu(),
                    #             #         True,
                    #             #         True,
                    #             #         ce[j].cpu()
                    #             #     )
                    #             # print("update 1")
                    #             # pbar_buffer.update(1)
                    #         # if replay_buffer.get_size() < self.option.batch_size and a[j] == 0:
                    #         #     replay_buffer.store(
                    #         #         s[j].cpu(),
                    #         #         a[j].cpu(),
                    #         #         a_log_prob[j].cpu(),
                    #         #         -0.95,
                    #         #         s_[j].cpu(),
                    #         #         False,
                    #         #         False,
                    #         #         ce[j].cpu()
                    #         # )
                    # if i == 1:
                    #     for j in range(len(s_)):
                    #         # if equal_arr[j]:
                    #         #     # if replay_buffer.get_size() < self.option.batch_size:
                    #         #     #     replay_buffer.store(
                    #         #     #             s[j].cpu(),
                    #         #     #             a[j].cpu(),
                    #         #     #             a_log_prob[j].cpu(),
                    #         #     #             -1,
                    #         #     #             s_[j].cpu(),
                    #         #     #             True,
                    #         #     #             True,
                    #         #     #             ce[j].cpu()
                    #         #     # )
                    #         #     continue
                    #         if arr_achieve[j]:
                    #             if replay_buffer.get_size() < self.option.batch_size:
                    #                 if ne[j] == te[j]:
                    #                     replay_buffer.store(
                    #                         s[j].cpu(),
                    #                         a[j].cpu(),
                    #                         a_log_prob[j].cpu(),
                    #                         1,
                    #                         s_[j].cpu(),
                    #                         True,
                    #                         True,
                    #                         ce[j].cpu()
                    #                     )
                    #                     self.option.writer.add_scalar('train/reward_per_sample_{}'.format(self.option.exp_name), 1, sample_inds)
                    #                     sample_inds += 1
                    #                 else:
                    #                     replay_buffer.store(
                    #                         s[j].cpu(),
                    #                         a[j].cpu(),
                    #                         a_log_prob[j].cpu(),
                    #                         -1,
                    #                         s_[j].cpu(),
                    #                         True,
                    #                         True,
                    #                         ce[j].cpu()
                    #                     )
                    #                     self.option.writer.add_scalar('train/reward_per_sample_{}'.format(self.option.exp_name), -1, sample_inds)
                    #                     sample_inds += 1
                    #         else:
                    #             if replay_buffer.get_size() + 1 < self.option.batch_size:
                    #                 if ne[j] == te[j]:
                    #                     replay_buffer.store(
                    #                         all_s[j][0],
                    #                         all_a[j][0],
                    #                         all_log[j][0],
                    #                         0.1 * self.option.gamma,
                    #                         all_s_[j][0],
                    #                         False,
                    #                         False,
                    #                         all_ce[j][0]
                    #                     )
                    #                     self.option.writer.add_scalar('train/reward_per_sample_{}'.format(self.option.exp_name), 0.1 * self.option.gamma, sample_inds)
                    #                     sample_inds += 1
                    #                     replay_buffer.store(
                    #                         s[j].cpu(),
                    #                         a[j].cpu(),
                    #                         a_log_prob[j].cpu(),
                    #                         0.1,
                    #                         s_[j].cpu(),
                    #                         True,
                    #                         True,
                    #                         ce[j].cpu()
                    #                     )
                    #                     self.option.writer.add_scalar('train/reward_per_sample_{}'.format(self.option.exp_name), 0.1, sample_inds)
                    #                     sample_inds += 1
                    #                 else:
                    #                     replay_buffer.store(
                    #                         all_s[j][0],
                    #                         all_a[j][0],
                    #                         all_log[j][0],
                    #                         -0.1 * self.option.gamma,
                    #                         all_s_[j][0],
                    #                         False,
                    #                         False,
                    #                         all_ce[j][0]
                    #                     )
                    #                     self.option.writer.add_scalar('train/reward_per_sample_{}'.format(self.option.exp_name), -0.1 * self.option.gamma, sample_inds)
                    #                     sample_inds += 1
                    #                     replay_buffer.store(
                    #                         s[j].cpu(),
                    #                         a[j].cpu(),
                    #                         a_log_prob[j].cpu(),
                    #                         -0.1,
                    #                         s_[j].cpu(),
                    #                         True,
                    #                         True,
                    #                         ce[j].cpu()
                    #                     )
                    #                     self.option.writer.add_scalar('train/reward_per_sample_{}'.format(self.option.exp_name), -0.1, sample_inds)
                    #                     sample_inds += 1
                    
                    for j in range(len(s_)):
                        all_s[j][i] = s[j].cpu()
                        all_a[j][i] = a[j].cpu()
                        all_log[j][i] = a_log_prob[j].cpu()
                        all_s_[j][i] = s_[j].cpu()
                        all_ce[j][i] = ce[j].cpu()
                        if i < self.option.train_step_length - 1:
                            all_done[j][i] = False
                            all_dw[j][i] = False
                        else:
                            # all_done[j][i] = True
                            # if ne[j] == te[j]:
                            #     all_dw[j][i] = True
                            # else:
                            #     all_dw[j][i] = True
                            all_done[j][i] = True
                            all_dw[j][i] = True
                            
                    # new_size = replay_buffer.get_size()
                    # pbar_buffer.update(new_size - old_size)
                    s = s_
                    # torch.cuda.empty_cache()
            # print(replay_buffer.get_size())
            
            final_rewards = self.manager.get_final_reward()
            for i in range(len(final_rewards)):
                final_reward = final_rewards[i]
                temp = [final_reward]
                # all_r[i].append(final_reward)
                for _ in range(self.option.train_step_length - 1):
                    final_reward = final_reward * self.option.gamma
                    temp.insert(0, final_reward)
                all_r.append(temp)
            
            # print(all_r)
            append_able = replay_buffer.get_size() < self.option.batch_size
            
            for i in range(len(all_s)):
                if not append_able:
                    break
                for j in range(len(all_s[i])):
                    # if arr_achieve[i] and j==1 and all_a[i][j] == 0:
                    #     continue
                    if replay_buffer.get_size() < self.option.batch_size:
                        replay_buffer.store(
                            all_s[i][j],
                            all_a[i][j],
                            all_log[i][j],
                            all_r[i][j],
                            all_s_[i][j],
                            all_dw[i][j],
                            all_done[i][j],
                            all_ce[i][j]
                        )
                        self.option.writer.add_scalar('train/reward_per_sample_{}'.format(self.option.exp_name), all_r[i][j], sample_inds)
                        sample_inds += 1
                        pbar_buffer.update(1)
                    else:
                        append_able = False
                        break
                    
            # zhugeliang_state_query = self.manager.zhugeliang()
            # if zhugeliang_state_query is not None:
            #     if replay_buffer.get_size() < self.option.batch_size:
            #         replay_buffer.store(
            #             all_s[-1],
            #             all_a[-1],
            #             all_log[-1],
            #             1,
            #             zhugeliang_state_query.squeeze().cpu(),
            #             True,
            #             True,
            #             all_ce[-1]
            #         )
            # When the number of transitions in buffer reaches batch_size,then update
            # print(replay_buffer.get_size())
            if replay_buffer.get_size() >= self.option.batch_size:
                # print(replay_buffer.r)
                self.manager.clear_gpu()
                all_s =  None
                all_a = None
                all_log = None
                all_s_ = None
                all_dw = None
                all_done = None
                all_ce = None
                process_reward = None
                all_r = []
                arr_achieve = None
                equal_arr= None
                # 清空GPU缓存
                torch.cuda.empty_cache()

                current_step = self.agent.update(replay_buffer, self.manager)
                # pbar.set_description("Epoch %s" % str(self.manager.epoch + 1))
                pbar.update(current_step - update_times)
                update_times = current_step
                replay_buffer.re_init()
                
                pbar_buffer.reset()
                train_times += 1
                # Evaluate the policy every 'evaluate_freq' steps
                if train_times % self.option.evaluate_freq == 0:
                    evaluate_num += 1
                    # tester = Tester(self.option, self.manager, self.agent, self.character)
                    all_final_reward_1, all_final_reward_0 = tester.test_once()
                    # tester = None
                    # 清空GPU缓存
                    torch.cuda.empty_cache()
                    # self.option.writer.add_scalar('valid/step_rewards_0{}'.format(self.option.exp_name), all_final_reward_0, evaluate_num)
                    self.option.writer.add_scalar('valid/step_rewards_1{}'.format(self.option.exp_name), all_final_reward_1, evaluate_num)
                    if all_final_reward_1 > max_reward:
                        impatience = 0
                        # print("saving models...")
                        max_reward = all_final_reward_1
                        self.agent.save()
                        # self.manager.save()
                    else:
                        impatience += 1
                        if impatience >= self.option.max_patience:
                            print(f"No patience! Final reward:\t{max_reward}")
                            return max_reward
                    pbar.set_postfix(impatience=impatience, times=evaluate_num, m_reward=max_reward, reward=all_final_reward_1)

if __name__ == '__main__':
    import datetime
    import json
    import os
    import torch
    import argparse

    from model.utils import check_dir
    from model.Data import DataLoader
    from model.Graph import KnowledgeGraph
    from torch.utils.tensorboard import SummaryWriter

    parser = argparse.ArgumentParser("Hyperparameter Setting for LLM-ARK")
    parser.add_argument('--exp_name', default="LLM-ARK", type=str)
    parser.add_argument('--data_dir', default="datasets", type=str)
    parser.add_argument('--dataset', default="OpenDialKG", type=str)
    parser.add_argument('--output_dir', default="output", type=str)
    parser.add_argument('--mode', default="train", type=str)
    parser.add_argument('--model', default="checkpoint", type=str)
    parser.add_argument('--character', default="Assistant", type=str, help="Target / MultiHop / Assistant / User / Reason")
    parser.add_argument('--use_trans_e', type=bool, default=False)
    parser.add_argument('--out_path_aware', type=bool, default=False)
    parser.add_argument('--out_path_shuffle', type=bool, default=False)
    parser.add_argument('--max_patience', default=10, type=int)
    parser.add_argument('--state_embed_size', default=4096, type=int)
    parser.add_argument("--use_bias", type=bool, default=False, help="whether to use bias for actor")
    parser.add_argument("--fp16", type=bool, default=False, help="whether to use fp16")
    parser.add_argument("--bf16", type=bool, default=False, help="whether to use bf16")
    parser.add_argument("--hidden_dim", type=int, default=4096, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument('--relation_embed_size', default=200, type=int)
    parser.add_argument('--entity_embed_size', default=200, type=int)
    parser.add_argument('--max_out', default=50, type=int)
    parser.add_argument('--train_step_length', default=1, type=int)
    parser.add_argument('--max_step_length', default=2, type=int)
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--train_times', default=8, type=int)
    parser.add_argument('--test_times', default=20, type=int)
    parser.add_argument("--epoch", type=int, default=10, help="Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=1024, help="Minibatch size")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--test_batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=1, help="Batch size")
    parser.add_argument("--stater_type", type=str, default="llama", help="stater type: llama, flant5-large, bert-base")
    parser.add_argument("--instruction_type", type=str, default="1", help="0: alpaca v1 prompt; 1: gpt normal prompt; 2: uninstruction prompt")
    parser.add_argument("--stater_path", type=str, default="/root/autodl-tmp/huggingface/hub/models--NousResearch--Llama-2-7b-hf/snapshots/dacdfcde31297e34b19ee0e7532f29586d2c17bc", help="stater path")
    parser.add_argument("--stater_cache_dir", type=str, default=None, help="stater cache dir")
    parser.add_argument("--actor_path", type=str, default="", help="actor path")
    parser.add_argument("--actor_checkpoint_path", type=str, default="")
    parser.add_argument("--rl_train_data_path", type=str, default="datasets/OpenDialKG/Reason/train_type_1.json")
    parser.add_argument("--rl_valid_data_path", type=str, default="datasets/OpenDialKG/Reason/valid_type_1.json")
    parser.add_argument("--rl_test_data_path", type=str, default="datasets/OpenDialKG/Reason/test_type_1.json")
    parser.add_argument("--entity_embedding_path", type=str, default="checkpoint/OpenDialKG/TransE/entity.pth", help="")
    parser.add_argument("--relation_embedding_path", type=str, default="checkpoint/OpenDialKG/TransE/relation.pth", help="actor checkpoint path")
    parser.add_argument("--positive_reward", type=float, default=1, help="positive reward")
    parser.add_argument("--negative_reward", type=float, default=-1, help="negative reward")
    parser.add_argument("--coh_weight", type=float, default=0.0, help="weight of coherence reward")
    parser.add_argument("--sim_weight", type=float, default=0.0, help="weight of similarity reward")
    parser.add_argument("--tar_weight", type=float, default=1, help="weight of target reward")
    parser.add_argument("--lr_a", type=float, default=5e-5, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=5e-5, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    option = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(option.seed)
        option.use_cuda = True
    else:
        torch.manual_seed(option.seed)
        option.use_cuda = False

    option.exp_dir = os.path.join('runs', option.exp_name)
    option.log_dir = '{}/{}/{}'.format(option.exp_dir, str.upper(option.mode + option.character),
                                       datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    option.graph_dir = os.path.join(option.data_dir, option.dataset, "Graph")
    option.generator_dir = os.path.join(option.data_dir, option.dataset, "Generator")

    check_dir(option.log_dir)

    with open(os.path.join(option.log_dir, "option.txt"), "w", encoding="utf8") as f:
        json.dump(option.__dict__, f, indent=4, ensure_ascii=True)

    option.compute_dtype = (
        torch.float16
        if option.fp16
        else (torch.bfloat16 if option.bf16 else torch.float32)
    )
    
    option.writer = SummaryWriter(log_dir=option.log_dir)

    data_loader = DataLoader(option)

    graph = KnowledgeGraph(option, data_loader)

    tracker = Tracker(option, graph)

    manager = Manager(option, data_loader, graph, option.character, tracker)

    agent = PPO(option, option.character, graph=graph)

    graph.out_array = graph.out_array.to(agent.device)

    trainer = Trainer(option, manager, agent, option.character)

    tester = Tester(option, manager, agent, option.character)

    if option.mode == "train":
        trainer.train()

    if option.mode == "test":
        # tester.test_once()
        tester.test()