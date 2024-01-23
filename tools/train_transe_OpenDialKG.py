import openke
import os
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

def check_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        
# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "../datasets/OpenDialKG/TransE/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# dataloader for test
# test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200, 
	p_norm = 1, 
	norm_flag = True)


# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = 5.0),
	batch_size = train_dataloader.get_batch_size()
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 1.0, use_gpu = True)
trainer.run()
transe_dir = "../checkpoint/OpenDialKG/TransE"
check_dir(transe_dir)
transe.save_checkpoint(os.path.join(transe_folder, "transe.ckpt"))

# get embedding

import torch

cpkt = torch.load("./checkpoint/transe.ckpt")

entity_weight = cpkt["ent_embeddings.weight"]

relation_weight = cpkt["rel_embeddings.weight"]

torch.save(entity_weight, os.path.join(transe_folder, "entity.pth"))
torch.save(relation_weight, os.path.join(transe_folder, "relation.pth"))

# test the model
# transe.load_checkpoint('./checkpoint/transe.ckpt')
# tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
# tester.run_link_prediction(type_constrain = False)