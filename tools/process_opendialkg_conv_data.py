import random
import re
import pandas as pd
import json
import os
from thefuzz import process
from tqdm import tqdm
import torch
import pickle
from thefuzz import process

def check_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

file = "opendialkg/data/opendialkg.csv"
generator_folder = "datasets/OpenDialKG/Generator"
graph_folder = "datasets/OpenDialKG/Conversation"
check_dir(generator_folder)
check_dir(graph_folder)

df = pd.read_csv(file,header = 0,usecols= ["Messages"])
df.shape

raw_data = []

for index, row in tqdm(df.iterrows()):
    raw_data.append(json.loads(row["Messages"]))

test_size = int(0.15 * len(raw_data))
train_size = len(raw_data) - test_size * 2
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(raw_data, [train_size, test_size, test_size])

def subset_to_arr(all_data, subset):
    sub_arr = []
    for i in subset.indices:
        sub_arr.append(all_data[i])
    return sub_arr

with open(os.path.join("train.json"),"w") as f:
    json.dump(subset_to_arr(raw_data, train_dataset),f,indent = 4, ensure_ascii=True)

with open(os.path.join("valid.json"),"w") as f:
    json.dump(subset_to_arr(raw_data, valid_dataset),f,indent = 4, ensure_ascii=True)

with open(os.path.join("test.json"),"w") as f:
    json.dump(subset_to_arr(raw_data, test_dataset),f,indent = 4, ensure_ascii=True)

path2knowledge = {}
knowledge2path = {}
source = []

def path2know(path:list,knowledge:str):
    not_exist_null = True
    nums = len(path)
    for p in path:
        s = p[0].strip()
        r = p[1].strip()
        t = p[2].strip()
        if s == '' or r == '' or t == '':
            not_exist_null = False
            continue
        k = knowledge
        if nums > 1:
            patt=t
            pattern = re.compile(patt)
            result = pattern.search(knowledge)
            k_span = result.span()
            k_real = k_span[-1]
            if len(knowledge)-1 > k_real:
                k_real += 1
            k = knowledge[:k_real]
            knowledge = knowledge[k_real+1:]
        str_path = s+"\t"+r+"\t"+t
        item_p2k = {str_path:k}
        item_k2p = {k:str_path}
        path2knowledge.update(item_p2k)
        knowledge2path.update(item_k2p)        
    return not_exist_null

def make_generator_data(data):
    datasets = []
    NO_KNOWLEDGE_TOKEN = "no_knowledge_used"
    for index, items in enumerate(data):
        episode = {}
        all_paths = []
        parallel_paths = []
        user_paths = []
        assistant_paths = []
        dialogs = []
        for i,v in enumerate(items):
            lens = len(items)
            dialog = {}
            knowledge = None
            # v.__contains__("message") and v["type"] == "chat" and
    #         if i==0 and v["sender"] == "assistant":
    #             #print("lalala...")
    #             continue
            if v.__contains__("message") and i < lens-1:
                if items[i+1].__contains__("metadata") and i < lens - 2:
                    try:
                        metadata = items[i+1]["metadata"]
        #                 if items[i+1]["action_id"] == "meta_thread/send_meta_message" and items[i+2].__contains__("message"):
        #                     dialog["text"] = v["message"]
        #                     dialog["knowledge"] = NO_KNOWLEDGE_TOKEN    
        #                     dialog["label"] = items[i+2]["message"]
        #                     dialogs.append(dialog)
        #                     continue
        #                 try:
        #                     if items[i+2].__contains__("metadata") and items[i+3].__contains__("metadata"):
        #                         dialog["text"] = items[i+2]["metadata"]["text"]
        #                         dialog["knowledge"] = NO_KNOWLEDGE_TOKEN
        #                         dialog["label"] = items[i+3]["metadata"]["text"]
        #                         dialogs.append(dialog)
        #                         i+=2
        #                 except:
        #                     i+=2
        #                     print("ggggggggg")
                        # 如果存在知识
                        if metadata.__contains__("path"):
                            path_arr = metadata["path"][1]
                            # 不存在空知识
                            if path2know(path_arr,metadata["path"][-1]):
                                all_paths.extend(path_arr)
                                parallel_paths.append(path_arr)
                                if v["sender"] == "user":
                                    assistant_paths.extend(path_arr)
                                else:
                                    user_paths.extend(path_arr)
                                knowledge = metadata["path"][-1]
                                dialog["current_entity"] = path_arr[0][0].strip()
    #                             dialog["chose_relation"] = path_arr[0][1].strip()
                                dialog["target_entity"] = path_arr[-1][2].strip()
                                dialog["path"] = path_arr
                            # 存在空知识
                            else:
                                knowledge = NO_KNOWLEDGE_TOKEN
                            dialog["text"] = v["message"]
                            dialog["knowledge"] = knowledge
                            dialog["label"] = items[i+2]["message"]
                            dialog["sender"] = v["sender"]
                            if dialog["text"] != "" and dialog["label"] != "":
                                dialogs.append(dialog)
                        # 不存在知识
                        else:
                            dialog["text"] = v["message"]
                            dialog["knowledge"] = NO_KNOWLEDGE_TOKEN
                            dialog["label"] = items[i+2]["message"]
                            dialog["sender"] = v["sender"]
                            if dialog["text"] != "" and dialog["label"] != "":
                                dialogs.append(dialog)
                            # print("hhhhhhhhhhhhh.....")
                    except:
                        pass
                        # print("ggggggggggg")
                else:
                    try:
                        dialog["label"] = items[i+1]["message"]
                        dialog["text"] = v["message"]
                        dialog["knowledge"] = NO_KNOWLEDGE_TOKEN
                        dialog["sender"] = v["sender"]
                        if dialog["text"] != "" and dialog["label"] != "":
                            dialogs.append(dialog)
                    except:
                        #print(items[i])
                        #print("oooooooooooooooooooo")
                        pass

        if len(dialogs) > 1:
            episode["dialog"] = dialogs
            episode["all_paths"] = all_paths
            episode["parallel_paths"] = parallel_paths
            episode["user_paths"] = user_paths
            episode["assistant_paths"] = assistant_paths
            datasets.append(episode)
    
    return datasets

all_corpus = make_generator_data(raw_data)

train_corpus = make_generator_data(train_dataset)
valid_corpus = make_generator_data(valid_dataset)
test_corpus = make_generator_data(test_dataset)

relations = []
entities = []
triples = []

def make_kg(corpus):
    for j in tqdm(corpus):
        for i in j["all_paths"]:
            i_0 = i[0].strip()
            i_1 = i[1].strip()
            i_2 = i[2].strip()
#             assert i_0 in all_entities
#             assert i_2 in all_entities
#             assert i_1 in all_relations
            if i_0 not in entities:
                entities.append(i_0)
            if i_2 not in entities:
                entities.append(i_2)
            if i_1 not in relations:
                relations.append(i_1)
            triple = f"{i_0}\t{i_1}\t{i_2}"
            if triple not in triples:
                triples.append(triple)
#             assert triple in all_triples

make_kg(train_corpus)
make_kg(valid_corpus)
make_kg(test_corpus)

print(len(entities))
print(len(relations))
print(len(triples))

with open(os.path.join(generator_folder,"all.json"),"w") as f:
    json.dump(all_corpus, f, indent=4, ensure_ascii=True)

with open(os.path.join(generator_folder,"train.json"),"w") as f:
    json.dump(train_corpus,f,indent = 4, ensure_ascii=True)

with open(os.path.join(generator_folder,"valid.json"),"w") as f:
    json.dump(valid_corpus,f,indent = 4, ensure_ascii=True)

with open(os.path.join(generator_folder,"test.json"),"w") as f:
    json.dump(test_corpus,f,indent = 4, ensure_ascii=True)

with open(os.path.join(graph_folder,"sub_entities.txt"),"w") as f:
    for i in entities:
        f.write(i.strip())
        f.write("\n")

with open(os.path.join(graph_folder,"sub_relations.txt"),"w") as f:
    for i in relations:
        f.write(i.strip())
        f.write("\n")

with open(os.path.join(graph_folder,"sub_triples.txt"),"w") as f:
    for i in triples:
        f.write(i.strip())
        f.write("\n")

with open(os.path.join(graph_folder,"path2knowledge.json"),"w") as f:
    json.dump(path2knowledge,f,indent = 4, ensure_ascii=True)

with open(os.path.join(graph_folder,"knowledge2path.json"),"w") as f:
    json.dump(knowledge2path,f,indent = 4, ensure_ascii=True)

