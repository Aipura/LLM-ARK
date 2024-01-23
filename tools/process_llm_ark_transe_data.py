import json
import os

from tqdm import tqdm
from collections import defaultdict

def check_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        
with open("datasets/OpenDialKG/Graph/entities.txt", "r", encoding="utf8") as f:
    entities = []
    for l in f.readlines():
        entities.append(l.strip())

with open("datasets/OpenDialKG/Graph/relations.txt", "r", encoding="utf8") as f:
    relations = []
    for l in f.readlines():
        relations.append(l.strip())

triples_len = 0

with open("datasets/OpenDialKG/Graph/sub_triples.txt", "r", encoding="utf8") as f:
    data = [l.strip().split("\t") for l in f.readlines()]
    triplets = defaultdict(list)
    for item in data:
        head = item[0]
        tail = item[2]
        relation = item[1]
        out = (relation, tail)
        if out not in triplets[head]:
            triplets[head].append(out)
            triples_len+=1

with open("datasets/OpenDialKG/Graph/triples.txt", "r", encoding="utf8") as f:
    data = [l.strip().split("\t") for l in f.readlines()]
    for item in tqdm(data):
        head = item[0]
        tail = item[2]
        relation = item[1]
        out = (relation, tail)
        if out not in triplets[head]:
            triplets[head].append(out)
            triples_len+=1

full_triples = []

for k, v in triplets.items():
    full_triples.append(f"{k}\tEqual\t{k}")
    for r, t in v:
        full_triples.append(f"{k}\t{r}\t{t}")
    full_triples.append(f"{k}\tPad\tPad")

entities.append("Equal")
entities.append("Pad")
relations.append("Equal")
relations.append("Pad")
full_triples.append(f"Pad\tEqual\tPad")
full_triples.append(f"Pad\tPad\tPad")

full_triples_len = len(full_triples)
print(len(full_triples))

# with open("datasets/OpenDialKG/Graph/full_triples.txt", "w", encoding="utf8") as f:
#     for i in full_triples:
#         f.write(i)
#         f.write("\n")

# for e in entities:
#     for t in ["Equal", "Pad"]:
#         head = e
#         tail = t
#         relation = t
#         triplets[head].append((relation, tail))

entity2num = {}
num2entity = {}
relation2num = {}
num2relation = {}

for i,v in enumerate(entities):
    entity2num[v] = i
    num2entity[i] = v

for i,v in enumerate(relations):
    relation2num[v] = i
    num2relation[i] = v

transe_folder = "datasets/OpenDialKG/TransE"
check_dir(transe_folder)

with open(os.path.join(transe_folder, "entity2id.txt"), "w", encoding="utf8") as f:
    f.write(str(len(entities)))
    f.write("\n")
    for i in entities:
        f.write(f"{i}\t{str(entity2num[i])}")
        f.write("\n")

with open(os.path.join(transe_folder, "relation2id.txt"), "w", encoding="utf8") as f:
    f.write(str(len(relations)))
    f.write("\n")
    for i in relations:
        f.write(f"{i}\t{str(relation2num[i])}")
        f.write("\n")

with open(os.path.join(transe_folder, "train2id.txt"), "w", encoding="utf8") as f:
    f.write(str(full_triples_len))
    f.write("\n")
    for triple in full_triples:
        h, r, t = triple.split("\t")
        f.write(f"{str(entity2num[h])} {str(entity2num[t])} {str(relation2num[r])}")
        f.write("\n")
