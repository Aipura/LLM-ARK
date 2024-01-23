from tqdm import tqdm
import os
import numpy as np
import json

entities = []
relations = []
conv_graph_folder = "datasets/OpenDialKG/Conversation"
opendialkg_folder = "opendialkg/data"
all_graph_folder = "datasets/OpenDialKG/Graph"

def check_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        
check_dir(all_graph_folder)

with open(os.path.join(conv_graph_folder, "sub_relations.txt"),"r") as f:
    for line in tqdm(f.readlines()):
        relations.append(line.strip())

with open(os.path.join(conv_graph_folder, "sub_entities.txt"),"r") as f:
    for line in tqdm(f.readlines()):
        entities.append(line.strip())

with open(os.path.join(opendialkg_folder, "opendialkg_relations.txt"), 'r') as f:
    for line in tqdm(f.readlines()):
        relation = line.strip()
        if relation not in relations:
            relations.append(relation)
            
with open(os.path.join(opendialkg_folder, "opendialkg_entities.txt"), 'r') as f:
    for line in tqdm(f.readlines()):
        entity = line.strip()
        if entity not in entities:
            entities.append(entity)
            
triples = []
with open(os.path.join(opendialkg_folder, "opendialkg_triples.txt"), 'r') as f:
        for line in tqdm(f.readlines()):
            triple = line.strip().split("\t")
            if len(triple) != 3:
                continue
            h = triple[0]
            t = triple[2]
            r = triple[1]

            if h.strip() not in entities:
                entities.append(h.strip())

            if t.strip() not in entities:
                entities.append(t.strip())
            
            if r.strip() not in relations:
                relations.append(r.strip())
                
            item = f"{h}\t{r}\t{t}"
            triples.append(item)

print(len(entities))

print(len(relations))

print(len(triples))

meta_info = {
    "num_entities": len(entities),
    "num_relations": len(relations),
    "num_triples": len(triples)
}

with open(os.path.join(all_graph_folder, "meta_data.json"), "w", encoding="utf8") as f:
    json.dump(meta_info, f, indent=4, ensure_ascii=False)

with open(os.path.join(all_graph_folder, "entities.txt",),"w") as f:
    for i in entities:
        f.write(i.strip())
        f.write("\n") 

with open(os.path.join(all_graph_folder, "relations.txt",),"w") as f:
    for i in relations:
        f.write(i.strip())
        f.write("\n")

with open(os.path.join(all_graph_folder, "triples.txt",),"w") as f:
    for i in triples:
        f.write(i.strip())
        f.write("\n")

