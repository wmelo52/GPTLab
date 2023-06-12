# -*- coding: utf-8 -*-

import torch

cnt = [0, 0, 0, 0]

for _ in range(5000):
  sampled_Y = torch.multinomial(torch.tensor([0.1, 0.1, 0.3, 0.5]), 1)
  cnt[sampled_Y[0]] += 1

print(cnt)