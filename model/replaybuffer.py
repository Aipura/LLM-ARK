import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, option, state_dim):
        self.option = option
        self.s = []
        self.a = []
        self.a_logprob = []
        self.r = []
        self.ce = []
        self.s_ = []
        self.dw = []
        self.done = []
        self.count = 0


    def store(self, s, a, a_logprob, r, s_, dw, done, ce):
        self.s.append(s.tolist())
        self.a.append(a)
        self.a_logprob.append(a_logprob.tolist())
        self.r.append(r)
        self.ce.append(ce)
        self.s_.append(s_.tolist())
        self.dw.append(dw)
        self.done.append(done)
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype = self.option.compute_dtype)
        a = torch.tensor(self.a, dtype=torch.long)  # In discrete action space, 'a' needs to be torch.long
        a_logprob = torch.tensor(self.a_logprob, dtype = self.option.compute_dtype)
        r = torch.tensor(self.r, dtype = self.option.compute_dtype)
        ce = torch.tensor(self.ce, dtype=torch.long) # In discrete action space, 'a' needs to be torch.long
        s_ = torch.tensor(self.s_, dtype = self.option.compute_dtype)
        dw = torch.tensor(self.dw, dtype = self.option.compute_dtype)
        done = torch.tensor(self.done, dtype = self.option.compute_dtype)

        return s, a, a_logprob, r, s_, dw, done, ce

    def re_init(self):
        self.s = []
        self.a = []
        self.a_logprob = []
        self.r = []
        self.ce = []
        self.s_ = []
        self.dw = []
        self.done = []
        self.count = 0
        
    def get_size(self):
        return len(self.s)
