import multiprocessing as mp

import gym
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MP_BatchSampler(object):
    def __init__(self, args,risk_learner_trainer, gamma_0, gamma_1):
        self.risklearner_trainer = risk_learner_trainer
        self.args = args
        self.gamma_0 = gamma_0
        self.gamma_1 = gamma_1
        self.current_epoch = 0

    def get_acquisition_score(self, tasks):
        acquisition_score, acquisition_mean, acquisition_std = self.risklearner_trainer.acquisition_function(tasks, self.gamma_0, self.gamma_1)
        return acquisition_score, acquisition_mean, acquisition_std

    def sample_tasks(self, shape, multiplier, init_dist='Uniform', test=False):
        candidate_tasks = torch.rand(int(multiplier*shape[0]*shape[1]),shape[2])
        # candidate_tasks = np.random.uniform(0.0, 1.0, size=(int(multiplier*shape[0]*shape[1]),shape[2]))
        acquisition_score, acquisition_mean, acquisition_std = self.get_acquisition_score(candidate_tasks) # candidate tasks 15 * loss 1
        acquisition_score = acquisition_score.squeeze(1) # candidate tasks 15
        if not self.args.no_add_random:
            selected_values, selected_index = torch.topk(acquisition_score, k=shape[0]*shape[1]//2)
        else:
            selected_values, selected_index = torch.topk(acquisition_score, k=shape[0]*shape[1])
        mask = ~torch.isin(torch.arange(0, int(multiplier*shape[0]*shape[1])), selected_index.cpu())
        unselected_index = torch.arange(0, int(multiplier*shape[0]*shape[1]))[mask]
        index=torch.cat((selected_index.cpu(),unselected_index),dim=0)[:shape[0]*shape[1]][torch.randperm(shape[0]*shape[1])] # num_tasks 10
        index = index.cpu()
        tasks = candidate_tasks[index]
        tasks = tasks.view(shape[0],shape[1],shape[2]).numpy()

        return tasks
    
    def train(self, tasks, y):
        loss, recon_loss, kl_loss = self.risklearner_trainer.train(tasks, y)
        return loss, recon_loss, kl_loss
    
class Diverse_MP_BatchSampler(MP_BatchSampler):
    def __init__(self, args,risk_learner_trainer, gamma_0, gamma_1, gamma_2):
        self.gamma_2 = gamma_2
        super(Diverse_MP_BatchSampler, self).__init__(args,risk_learner_trainer, gamma_0, gamma_1)

    def get_acquisition_score(self, tasks, real_batch_size=None, diversified=False):
        if real_batch_size is None:
            real_batch_size = int(tasks.shape[0])
        if diversified:
            best_batch_id, diversified_score, combine_local_diverse_score, combine_local_acquisition_score, acquisition_score = self.risklearner_trainer.acquisition_function(tasks,  self.gamma_0, self.gamma_1, self.gamma_2, real_batch_size=real_batch_size)
            return best_batch_id, diversified_score, combine_local_diverse_score, combine_local_acquisition_score, acquisition_score
        else:
            acquisition_score, acquisition_mean, acquisition_std = self.risklearner_trainer.acquisition_function(tasks, self.gamma_0, self.gamma_1, self.gamma_2, pure_acquisition=True, real_batch_size=real_batch_size)
            return acquisition_score, acquisition_mean, acquisition_std


    def sample_tasks(self, shape, multiplier, init_dist='Uniform', test=False):

        candidate_tasks = torch.rand(int(multiplier*shape[0]*shape[1]),shape[2])
        # candidate_tasks = np.random.uniform(0.0, 1.0, size=(int(multiplier*shape[0]*shape[1]),shape[2]))
        best_batch_id, diversified_score, combine_local_diverse_score, combine_local_acquisition_score, acquisition_score = self.get_acquisition_score(candidate_tasks, real_batch_size=int(shape[0]*shape[1]), diversified=True) # candidate tasks 15 * loss 1
        index = best_batch_id
        tasks = candidate_tasks[index]
        tasks = tasks.view(shape[0],shape[1],shape[2]).numpy()

        return tasks, diversified_score, combine_local_diverse_score, combine_local_acquisition_score
