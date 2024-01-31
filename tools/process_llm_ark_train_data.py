import datetime
import json
import os
import torch
import argparse
import copy
import matplotlib.pyplot as plt
import numpy as np

import sys
# 获取当前模块的路径
current_module = __file__
parent_dir = os.path.dirname(os.path.abspath(current_module))
grandfather_dir = os.path.join(parent_dir, "..") # 上两级目录为父模块所在位置
 
# 将父模块所在的路径添加到系统路径列表中
sys.path.append(grandfather_dir)

from model.utils import check_dir
from model.Data import DataLoader
from model.Graph import KnowledgeGraph
from pprint import pprint

parser = argparse.ArgumentParser("Hyperparameter Setting for LLM-ARK")
parser.add_argument('--exp_name', default="LLM-ARK", type=str)
parser.add_argument('--data_dir', default="datasets", type=str)
parser.add_argument('--dataset', default="OpenDialKG", type=str)
parser.add_argument('--output_dir', default="output", type=str)
parser.add_argument('--mode', default="train", type=str)
parser.add_argument('--model', default="checkpoint", type=str)
parser.add_argument('--character', default="Assistant", type=str, help="Target / MultiHop / Assistant / User / Reason")
parser.add_argument('--use_trans_e', type=bool, default=True)
parser.add_argument('--out_path_aware', type=bool, default=True)
parser.add_argument('--out_path_shuffle', type=bool, default=True)
parser.add_argument('--max_patience', default=10, type=int)
parser.add_argument('--state_embed_size', default=4096, type=int)
parser.add_argument("--use_bias", type=bool, default=False, help="whether to use bias for actor")
parser.add_argument("--fp16", type=bool, default=False, help="whether to use fp16")
parser.add_argument("--bf16", type=bool, default=True, help="whether to use bf16")
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

torch.cuda.manual_seed_all(option.seed)
option.use_cuda = False

option.exp_dir = os.path.join('runs', option.exp_name)
option.log_dir = '{}/{}/{}'.format(option.exp_dir, str.upper(option.mode + option.character),
                                   datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
option.graph_dir = os.path.join(option.data_dir, option.dataset, "Graph")
option.generator_dir = os.path.join(option.data_dir, option.dataset, "Generator")
option.reason_dir = os.path.join(option.data_dir, option.dataset, "Reason")

option.action_dim = option.relation_embed_size

check_dir(option.reason_dir)

data_loader = DataLoader(option)
graph = KnowledgeGraph(option, data_loader)

data_path = "datasets/OpenDialKG/Generator/all.json"

with open(data_path, "r", encoding="utf8") as f:
    data = json.load(f)

def getout(current):
    arr = []
    ce = data_loader.entity2num[current]
    outs = graph.get_out(ce)
    for relation, target in outs.squeeze().cpu().numpy():
        r = data_loader.num2relation[relation]
        t = data_loader.num2entity[target]
        if t != "Pad":
            arr.append(f"{current},{r},{t}")
    return arr


instruction = """
You are now an assistant and are answering a user's Utterance. Starting with the Current Entity as the starting point, performing one or two-hop reasoning on the knowledge graph based on the query and Dialog History, and the Path History is a set of triples that consisting of [Starting Entity, Relation, Target Entity]
"""

inputs = """
Dialog History: {}
Utterance: {}
Path History: {}
Current Entity: {}
Current Step: {}


{}

"""

task_background = "Performing one-hop reasoning on the knowledge graph."

normal_example = """

### Examples

Environment:
Dialog History: []
Utterance: What do you think about the Washinton Redskins? Are you a fan?
Path History: ['Washington Redskins,~Team coached,Mike Shanahan']
Current Entity: Washington Redskins
Current Step: 2

Response:
Washington Redskins,~Team coached,Mike Shanahan

"""

opa_example="""

### Examples

Environment:
Dialog History: []
Utterance: What do you think about the Washinton Redskins? Are you a fan?
Path History: ['Washington Redskins,~Team coached,Mike Shanahan']
Current Entity: Washington Redskins
Current Step: 2
Out Paths: ['Washington Redskins,Equal,Washington Redskins', 'Washington Redskins,~Team coached,Mike Shanahan', 'Washington Redskins,~Champion,Super Bowl XXVI', 'Washington Redskins,~Team,National Football League', 
 'Washington Redskins,~Runner-up,Super Bowl VII', 'Washington Redskins,~Team Owned,Dwight Schar', 'Washington Redskins,~Game,Mike Sellers', 'Washington Redskins,~Team coached,Jay Gruden', 'Washington Redskins,~Current team head coached,Jay Gruden', 'Washington Redskins,~Runner-up,Super Bowl XVIII', 
 'Washington Redskins,~Coaching history,Vince Lombardi', 'Washington Redskins,~Game,Jason Taylor', 'Washington Redskins,~Game,Todd Collins', 
 'Washington Redskins,~Game,Santana Moss', 'Washington Redskins,~Game,Brian Orakpo', 'Washington Redskins,~Game,Ladell Betts', 'Washington 
 'Washington Redskins,~Game,Kedric Golston']

Response:
Washington Redskins,~Team coached,Mike Shanahan

"""

normal_prompt="""
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

out_path_aware_prompt="""
### Task Background:
Performing one-hop reasoning on the knowledge graph.


### Instruction:
Given the Task Background and the Environment, please choose a properate KG path from a set of Out Paths, directly output this path in triplet format without any other content.


### Environment:
Dialog History: {}
Utterance: {}
Path History: {}
Current Entity: {}
Current Step: {}
Out Paths: {}


{}


### Response:
"""


## RL data

alpaca_format = """
Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.\n\n
### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:
"""

base_threshold = 1600 - len(alpaca_format.format(instruction=instruction, input=""))


rl_data = []
rl_data_len_path_1 = []
rl_data_len_path_2 = []
lens = []
if option.instruction_type == "0":
    rl_lens_threshold  = base_threshold + len(alpaca_format.format(instruction=instruction, input=""))
elif option.instruction_type == "1":
    rl_lens_threshold  = base_threshold + len(normal_prompt)
else:
    rl_lens_threshold  = base_threshold


path_len_distribution = [0] * 2
for i in data:
    context = []
    path_history = []
    for item in i["dialog"]:
        sender = item["sender"]
        text = item["text"]
        label = item["label"]
        if "current_entity" in item.keys() and sender == "user":
            path = item["path"]
            ce = path[0][0]
            te = path[0][-1]
            # if len(path) == 1:
            #     continue
            # 去掉不连贯数据
            if len(path) == 2 and path[0][-1] != path[1][0]: 
                continue
            # s_p = ','.join(path[0])
            sample = {
                "input": "",
                "dialog_history": copy.deepcopy(context),
                "query": text,
                "path_history": copy.deepcopy(path_history),
                "current_entity": data_loader.entity2num[ce],
                "current_entity_str": ce,
                "target_entity": data_loader.entity2num[te],
                "target_entity_str": te,
                "step": len(path)
            }
            if option.instruction_type == "0":
                temp_input = inputs.format(sample["dialog_history"], sample["query"], sample["path_history"], ce, 1, '')
                final_prompt = alpaca_format.format(instruction=instruction, input=temp_input)
                sample["input"] = final_prompt
            elif option.instruction_type == "1":
                sample["input"] = normal_prompt.format(sample["dialog_history"], sample["query"], sample["path_history"], ce, 1, '')
            else:
                sample["input"] = inputs.format(context, text, path_history, ce, 1, '')
            input_len = len(sample["input"])
            lens.append(input_len)
            if input_len <= rl_lens_threshold:
                # if len(path) == 2:
                #     for i in range(3):
                #         path_len_distribution[1] += 1
                #         rl_data.append(copy.deepcopy(sample))
                # else:
                path_len_distribution[len(path)-1] += 1
                # rl_data.append(copy.deepcopy(sample))
                # rl_data.append(copy.deepcopy(sample))
                path_num = []
                for p in path:
                    h_ = p[0].strip()
                    r_ = p[1].strip()
                    t_ = p[2].strip()
                    path_num.append([data_loader.entity2num[h_], data_loader.relation2num[r_], data_loader.entity2num[t_]])
                if len(path) == 1:
                    h_ = te
                    r_ = "Equal"
                    t_ = te
                    path_num.append([data_loader.entity2num[h_], data_loader.relation2num[r_], data_loader.entity2num[t_]])
                    # rl_data_len_path_2.append(copy.deepcopy(sample))
                # else:
                    # rl_data_len_path_1.append(copy.deepcopy(sample))
                sample["path"] = path_num
                rl_data.append(copy.deepcopy(sample))
                
                if len(path) == 2:
                    rl_data_len_path_2.append(copy.deepcopy(sample))
                else:
                    rl_data_len_path_1.append(copy.deepcopy(sample))
                    
        if "current_entity" in item.keys() and sender == "assistant":
            user_path = item["path"]
            for u_p in user_path:
                u_p_str = ','.join(u_p)
                path_history.append(u_p_str)
        context.append(f"{sender}: {text}")

def split_dataset(data, gamma):
    train_size=int(gamma*len(data))
    # print(train_size)
    test_size=len(data) - train_size
    # print(test_size)
    train_dataset, test_dataset=torch.utils.data.random_split(data,[train_size, test_size])
    return list(train_dataset), list(test_dataset)

# len_path_1_train_dataset, len_path_1_test_dataset =  split_dataset(rl_data_len_path_1, 0.85)

# len_path_2_train_dataset, len_path_2_test_dataset =  split_dataset(rl_data_len_path_2, 0.85)

# rl_train_dataset = len_path_1_train_dataset + len_path_2_train_dataset
# rl_test_dataset = len_path_1_test_dataset + len_path_2_test_dataset

np.random.seed(option.seed)
np.random.shuffle(rl_data)

rl_train_dataset, rl_test_val_dataset = split_dataset(rl_data, 0.7)

len_test_dataset = len(rl_test_val_dataset) // 2

rl_valid_dataset = rl_test_val_dataset[:len_test_dataset]

rl_test_dataset = rl_test_val_dataset[len_test_dataset:]

pprint(rl_train_dataset[0])

print(f"saving train data at {option.rl_train_data_path}")
with open(option.rl_train_data_path, "w", encoding="utf8") as f:
    json.dump(rl_train_dataset, f, indent=4, ensure_ascii=False)
    
print(f"saving valid data at {option.rl_valid_data_path}")
with open(option.rl_valid_data_path, "w", encoding="utf8") as f:
    json.dump(rl_valid_dataset, f, indent=4, ensure_ascii=False)
    
print(f"saving test data at {option.rl_test_data_path}")   
with open(option.rl_test_data_path, "w", encoding="utf8") as f:
    json.dump(rl_test_dataset, f, indent=4, ensure_ascii=False)
