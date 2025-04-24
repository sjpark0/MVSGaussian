import tqdm
import torch

from lib.utils import net_utils
import time

import os
import numpy as np
from lib.networks.mvsgs.network import Network
from lib.datasets.colmap.mvsgs import Dataset
from lib.datasets.samplers import mvsgsBatchSampler
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader
from types import SimpleNamespace

from lib.config import cfg
from lib.networks import make_network
from lib.evaluators import make_evaluator

def worker_init_fn(worker_id):
    np.random.seed(worker_id + (int(round(time.time() * 1000) % (2**16))))

def make_data_loader():
    dataset_dict = {}
    dataset_dict['data_root'] = "examples"
    dataset_dict['split'] = "test"
    dataset_dict['input_h_w'] = [640, 960]
    dataset_dict['scene'] = "Sample1"
    dataset = Dataset(**dataset_dict)
    sampler = torch.utils.data.sampler.SequentialSampler(dataset)

    sampler_meta = SimpleNamespace()
    sampler_meta.input_views_num = [3]
    sampler_meta.input_views_prob = [1.]

    batch_sampler = mvsgsBatchSampler(sampler, 1, False, sampler_meta)
    num_workers = 4

    
    collator = default_collate
    data_loader = DataLoader(dataset,
                            batch_sampler=batch_sampler,
                            num_workers=num_workers,
                            collate_fn=collator,
                            worker_init_fn=worker_init_fn)

    return data_loader


model_dir = "trained_model/mvsgs/Sample1"
network = make_network(cfg).cuda()
net_utils.load_network(network, model_dir)
network.eval()


data_loader = make_data_loader()
evaluator = make_evaluator(cfg)
net_time = []
scenes = []
for batch in tqdm.tqdm(data_loader):
    for k in batch:
        if k != 'meta':
            if 'novel_view' in k:
                for v in batch[k]:
                    batch[k][v] = batch[k][v].cuda()
            elif k == 'rendering_video_meta':
                for i in range(len(batch[k])):
                    for v in batch[k][i]:
                        batch[k][i][v] = batch[k][i][v].cuda()
            else:
                batch[k] = batch[k].cuda()
    with torch.no_grad():
        output = network(batch)
    evaluator.evaluate(output, batch)
    
    scenes.append(batch['meta']['scene'][0])
        
