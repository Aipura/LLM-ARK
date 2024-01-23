from model.Graph import KnowledgeGraph
import json
import os
import numpy as np
import torch

from collections import defaultdict
from torch.utils.data import Dataset
from tqdm import tqdm


class OpendialKGData(Dataset):
    def __init__(self, data_path):
        self.instruction = []
        self.input = []
        self.dialog_history = []
        self.query = []
        self.path_history = []
        self.current_entity = []
        self.target_entity = []
        self.action = []
        self.next_path = []
        self.step = []
        self.path = []
        with open(data_path, "r", encoding="utf8") as f:
            data = json.load(f)
        for item in data:
            if item["step"] <= 2:
                # self.instruction.append(item["instruction"])
                self.input.append(item["input"])
                self.dialog_history.append(item["dialog_history"])
                self.query.append(item["query"])
                self.path_history.append(item["path_history"])
                self.current_entity.append(item["current_entity"])
                self.target_entity.append(item["target_entity"])
                # self.next_path.append(item["next_path"])
                self.step.append(item["step"])
                if item.__contains__("path"):
                    self.path.append(item["path"])
            # self.output.append(item["output"])

    def __getitem__(self, idx):
        # c = self.current_entity[idx]
        # t = self.target_entity[idx]
        # c_t = torch.LongTensor([c])
        # t_t = torch.LongTensor([t])
        batch = {
            # "instruction": self.instruction[idx],
            "input": self.input[idx],
            "dialog_history": self.dialog_history[idx],
            "query": self.query[idx],
            "path_history": self.path_history[idx],
            "current_entity": self.current_entity[idx],
            "target_entity": self.target_entity[idx],
            # "next_path": self.next_path[idx],
            "step": self.step[idx],
        }
        if self.path:
            batch["path"] = self.path[idx]
        else:
            batch["path"] = []
        return batch

    def __len__(self):
        return len(self.query)


# class OpendialKGData(Dataset):
#     def __init__(self, data_path):
#         self.heads = []
#         self.targets = []
#         self.utterances = []
#         self.contexts = []
#         self.paths = []
#         with open(data_path, "r", encoding="utf8") as f:
#             data = json.load(f)
#         for dialog in data:
#             self.heads.append(dialog["start_entity"])
#             self.targets.append(dialog["target_entity"])
#             self.contexts.append(dialog["context"])
#             self.utterances.append(dialog["utterance"])
#             self.paths.append(dialog["path"])

#     def __getitem__(self, idx):
#         h = self.heads[idx]
#         t = self.targets[idx]
#         c = self.contexts[idx]
#         u = self.utterances[idx]
#         p = self.paths[idx]
#         batch = {
#             "start_entity": torch.LongTensor([h]),
#             "target_entity": torch.LongTensor([t]),
#             "context": torch.FloatTensor(c),
#             "utterance": torch.FloatTensor(u),
#             "path": p,
#             # "split_target": torch.LongTensor([p[0][2], p[1][2]]).squeeze()
#         }
#         return batch

#     def __len__(self):
#         return len(self.heads)
    
# class MetaQAData(Dataset):
#     def __init__(self, data_path):

#         self.queries = []
#         self.current_entities = []
#         self.target_entities = []
        
#         with open(data_path, "r", encoding="utf8") as f:
#             data = json.load(f)
            
#         for item in data:
#             self.queries.append(item["query"])
#             self.current_entities.append(item["current_entity"])
#             self.target_entities.append(item["target_entity"])

#     def __getitem__(self, idx):
#         batch = {
#             "query": self.queries[idx],
#             "current_entity": self.current_entities[idx],
#             "target_entity": self.target_entities[idx],
#         }
#         return batch

#     def __len__(self):
#         return len(self.queries)
    
class DataLoader:
    def __init__(self, option):
        self.sub_graph_data = None
        self.test_dataset = None
        self.option = option
        self.include_reverse = False

        self.graph_data = None
        self.train_data = None
        self.test_data = None
        self.valid_data = None

        self.reason_train_data = None
        self.reason_test_data = None
        self.reason_valid_data = None

        self.entity2num = None
        self.num2entity = None

        self.relation2num = None
        self.num2relation = None
        self.relation2inv = None

        self.num_relation = 0
        self.num_entity = 0
        self.num_operator = 0

        self.knowledge2path = {}
        self.path2knowledge = {}

        # self.train_data_path = os.path.join(self.option.target_dir, "train.json")
        # self.valid_data_path = os.path.join(self.option.target_dir, "valid.json")
        # self.test_data_path = os.path.join(self.option.target_dir, "test.json")

        # self.multi_hop_train_data_path = os.path.join(self.option.multi_hop_dir, "train.json")
        # self.multi_hop_valid_data_path = os.path.join(self.option.multi_hop_dir, "valid.json")
        # self.multi_hop_test_data_path = os.path.join(self.option.multi_hop_dir, "test.json")

        # self.reason_train_data_path = os.path.join(self.option.reason_dir, "train.json")
        # self.reason_valid_data_path = os.path.join(self.option.reason_dir, "valid.json")
        # self.reason_test_data_path = os.path.join(self.option.reason_dir, "test.json")

        # self.user_reason_train_data_path = os.path.join(self.option.user_reason_dir, "train.json")
        # self.user_reason_valid_data_path = os.path.join(self.option.user_reason_dir, "valid.json")
        # self.user_reason_test_data_path = os.path.join(self.option.user_reason_dir, "test.json")

        # self.assistant_reason_train_data_path = os.path.join(self.option.assistant_reason_dir, "train.json")
        # self.assistant_reason_valid_data_path = os.path.join(self.option.assistant_reason_dir, "valid.json")
        # self.assistant_reason_test_data_path = os.path.join(self.option.assistant_reason_dir, "test.json")

        self.load_data_all()

    def load_data_all(self):
        graph_data_path = os.path.join(self.option.graph_dir, "triples.txt")
        sub_graph_data_path = os.path.join(self.option.graph_dir, "sub_triples.txt")
        entity_path = os.path.join(self.option.graph_dir, "entities.txt")
        relations_path = os.path.join(self.option.graph_dir, "relations.txt")

        # knowledge2path_path = os.path.join(self.option.graph_dir, "knowledge2path.json")
        # path2knowledge_path = os.path.join(self.option.graph_dir, "path2knowledge.json")

        self.entity2num, self.num2entity = self._load_dict(entity_path)
        self.relation2num, self.num2relation = self._load_dict(relations_path)

        # self._load_k2p_dict(knowledge2path_path)
        # self._load_p2k_dict(path2knowledge_path)

        if self.include_reverse:
            self._augment_reverse_relation()
        if self.option.dataset == "OpenDialKG":
            self._add_item(self.relation2num, self.num2relation, "Equal")
            self._add_item(self.entity2num, self.num2entity, "Equal")
            self._add_item(self.relation2num, self.num2relation, "Pad")
            # self._add_item(self.relation2num, self.num2relation, "Start")
            # self._add_item(self.relation2num, self.num2relation, "Stop")

            self._add_item(self.entity2num, self.num2entity, "Pad")
        # self._add_item(self.entity2num, self.num2entity, "Stop")

        self.num_relation = len(self.relation2num)
        self.num_entity = len(self.entity2num)

        self.option.num_entity = self.num_entity
        self.option.num_relation = self.num_relation

        print("num_relation", self.num_relation)
        print("num_entity", self.num_entity)

        self.graph_data = self._load_graph_data(graph_data_path)
        self.sub_graph_data = self._load_graph_data(sub_graph_data_path)

    def _load_graph_data(self, path):
        data = [l.strip().split("\t") for l in open(path, "r").readlines()]
        triplets = list()
        for item in data:
            head = self.entity2num[item[0]]
            tail = self.entity2num[item[2]]
            relation = self.relation2num[item[1]]
            triplets.append([head, relation, tail])
            if self.include_reverse:
                inv_relation = self.relation2num["inv_" + item[1]]
                triplets.append([tail, inv_relation, head])
        return triplets

    # def _load_double_data(self, path):
    #     data = [l.strip().split("\t") for l in open(path, "r").readlines()]
    #     doublets = list()
    #     for item in data:
    #         head = self.entity2num[item[0].strip()]
    #         tail = self.entity2num[item[1].strip()]
    #         doublets.append([head, tail])
    #     return doublets

    def _tokenize(self, data_path, file_path):
        if not os.path.exists(file_path):
            print(f"tokenizing... file will be saved at: {file_path}")
            with open(data_path, "r", encoding="utf8") as f1, open(file_path, "w", encoding="utf8") as f2:
                data = json.load(f1)
                tokenizer_data = []
                for dialog in tqdm(data):
                    start_entity = self.entity2num[dialog["current_entity"].strip()]
                    target_entity = self.entity2num[dialog["target_entity"].strip()]
                    context = self.predictor.tokenize(dialog["context"])
                    utterance = self.predictor.tokenize(dialog["utterance"])
                    path = []
                    for p in dialog["path"]:
                        h_ = p[0].strip()
                        r_ = p[1].strip()
                        t_ = p[2].strip()
                        path.append([self.entity2num[h_], self.relation2num[r_], self.entity2num[t_]])
                    item = {
                        "start_entity": start_entity,
                        "target_entity": target_entity,
                        "context": context,
                        "utterance": utterance,
                        "path": path,
                    }
                    tokenizer_data.append(item)
                json.dump(tokenizer_data, f2, indent=4, ensure_ascii=True)

    def _load_dict(self, path):
        obj2num = defaultdict(int)
        num2obj = defaultdict(str)
        data = [l.strip() for l in open(path, "r").readlines()]
        for num, obj in enumerate(data):
            obj2num[obj] = num
            num2obj[num] = obj
        return obj2num, num2obj

    def _augment_reverse_relation(self):
        num_relation = len(self.num2relation)
        temp = list(self.num2relation.items())
        self.relation2inv = defaultdict(int)
        for n, r in temp:
            rel = "inv_" + r
            num = num_relation + n
            self.relation2num[rel] = num
            self.num2relation[num] = rel
            self.relation2inv[n] = num
            self.relation2inv[num] = n

    def _add_item(self, obj2num, num2obj, item):
        count = len(obj2num)
        obj2num[item] = count
        num2obj[count] = item

    def _load_k2p_dict(self, path):
        with open(path, 'r', encoding='utf8') as f:
            self.knowledge2path = json.load(f)

    def _load_p2k_dict(self, path):
        with open(path, 'r', encoding='utf8') as f:
            self.path2knowledge = json.load(f)

    def get_graph_data(self):
        # with open(os.path.join(self.option.log_dir, "train_log.txt"), "a+", encoding='UTF-8') as f:
        #     f.write("Train graph contains " + str(len(self.graph_data)) + " triples\n")
        return np.array(self.graph_data, dtype=np.int64)

    def get_sub_graph_data(self):
        # with open(os.path.join(self.option.log_dir, "train_log.txt"), "a+", encoding='UTF-8') as f:
        #     f.write("Train graph contains " + str(len(self.graph_data)) + " triples\n")
        return np.array(self.sub_graph_data, dtype=np.int64)

    def get_reason_train_data(self):
        return OpendialKGData(self.option.rl_train_data_path)

    def get_reason_valid_data(self):
        return OpendialKGData(self.option.rl_valid_data_path)
    
    def get_reason_test_data(self):
        return OpendialKGData(self.option.rl_test_data_path)


class ParserException(Exception):
    def __init__(self, msg):
        '''
        :param msg: 异常信息
        '''
        self.msg = msg