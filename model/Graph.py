import torch
import numpy as np
from collections import defaultdict
import copy
from tqdm import tqdm
import torch.nn as nn
import random

class KnowledgeGraph:
    def __init__(self, option, data_loader):
        self.option = option
        self.data_loader = data_loader
        self.sub_graph_data = data_loader.get_sub_graph_data()
        self.graph_data = data_loader.get_graph_data()
        self.out_array = None
        self.all_correct = None
        self.construct_graph()

    def construct_graph(self):
        print("constructing graph...")
        all_out_dict = defaultdict(list)
        for head, relation, tail in tqdm(self.sub_graph_data):
            out = (relation, tail)
            if out not in all_out_dict[head]:
                all_out_dict[head].append(out)
            all_out_dict[head].append((relation, tail))
        # if self.option.fine_tune:
        for head, relation, tail in tqdm(self.graph_data):
            out = (relation, tail)
            if out not in all_out_dict[head]:
                all_out_dict[head].append(out)
        
        if self.option.out_path_shuffle:
            print("shuffle the out paths of graph")
            np.random.seed(self.option.seed)
            for head in tqdm(all_out_dict):
                if len(all_out_dict[head]) <= self.option.max_out - 1:
                    np.random.shuffle(all_out_dict[head])
                else:
                    all_out_dict[head] = all_out_dict[head][:self.option.max_out]
                    np.random.shuffle(all_out_dict[head])

        out_array = np.ones((self.option.num_entity, self.option.max_out, 2), dtype=np.int64)
        out_array[:, :, 0] *= self.data_loader.relation2num["Pad"]
        out_array[:, :, 1] *= self.data_loader.entity2num["Pad"]

        more_out_count = 0
        for head in tqdm(all_out_dict):
            # if self.option.dataset == "OpenDialKG":
            out_array[head, 0, 0] = self.data_loader.relation2num["Equal"]
            out_array[head, 0, 1] = head
            num_out = 1
            # else:
                # num_out = 0
            for relation, tail in all_out_dict[head]:
                if num_out == self.option.max_out:
                    more_out_count += 1
                    break
                out_array[head, num_out, 0] = relation
                out_array[head, num_out, 1] = tail
                num_out += 1
                # all_correct[(head, relation)].add(tail)
                
        # print("shuffling out paths...")
        # for head in tqdm(all_out_dict):
        #     random.shuffle(all_out_dict[head])

        self.out_array = torch.from_numpy(out_array)
        # self.all_correct = all_correct
        print("more_out_count", more_out_count)
        # if self.option.use_cuda:
        #     self.out_array = self.out_array.cuda()

    def get_out(self, current_entities):
        # ret = copy.deepcopy(self.out_array[current_entities, :, :])
        return self.out_array[current_entities, :, :]
 
    def get_next(self, current_entity, out_id):
        next_relation = self.out_array[current_entity, out_id, 0]
        next_entity = self.out_array[current_entity, out_id, 1]
        return next_relation, next_entity

    def get_nexts(self, current_entities, out_ids):
        next_relations = []
        next_entities = []
        for i,j in zip(current_entities, out_ids):
            next_relation, next_entity = self.get_next(i,j)
            next_relations.append(next_relation)
            next_entities.append(next_entity)
        # next_out = self.out_array[current_entities, :, :]
        # next_out_list = list()
        # for i in range(out_ids.shape[0]):
        #     next_out_list.append(next_out[i, out_ids[i]])
        # next_out = torch.stack(next_out_list)
        # next_relations = next_out[:, 0]
        # next_entities = next_out[:, 1]
        return torch.stack(next_relations), torch.stack(next_entities)

    def get_action(self, current_entity, out_id):
        next_action = self.out_array[current_entity, out_id, 0]
        return next_action
