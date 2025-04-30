import torch
import torch.nn.functional as F
import pdb
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal
import itertools
import numpy as np

class RiskLearnerTrainer():
    """
    Class to handle training of RiskLearner for functions.
    """

    def __init__(self, device, risklearner, optimizer, 
                 output_type="deterministic", 
                 kl_weight=0.0001, 
                 diversity_type=None, # "msd" or "rs"
                 posterior_sampling=False,
                 ):
        
        self.diversity_type = diversity_type
        self.posterior_sampling = posterior_sampling
        self.device = device
        self.risklearner = risklearner
        self.optimizer = optimizer

        # ++++++Prediction distribution p(l|tau)++++++++++++++++++++++++++++
        self.output_type = output_type
        self.kl_weight = kl_weight

        # ++++++initialize the p(z_0)++++++++++++++++++++++++++++
        r_dim = self.risklearner.r_dim
        prior_init_mu = torch.zeros([1, r_dim]).to(self.device)
        prior_init_sigma = torch.ones([1, r_dim]).to(self.device)
        self.z_prior = Normal(prior_init_mu, prior_init_sigma)
        self.last_risk_x = None
        self.last_risk_y = None

        # ++++++Acquisition functions++++++++++++++++++++++++++++
        self.acquisition_type = "lower_confidence_bound"
        if not self.posterior_sampling:
            self.num_samples = 50
        else:
            self.num_samples = 1

    def train(self, Risk_X, Risk_Y):
        Risk_X, Risk_Y = Risk_X.unsqueeze(0), Risk_Y.unsqueeze(0).unsqueeze(-1)
        # shape: batch_size, num_points, dim

        self.optimizer.zero_grad()
        p_y_pred, z_variational_posterior = self.risklearner(Risk_X, Risk_Y, self.output_type)
        z_prior = self.z_prior

        loss, recon_loss, kl_loss = self._loss(p_y_pred, Risk_Y, z_variational_posterior, z_prior)
        loss.backward()
        self.optimizer.step()

        # updated z_prior
        self.z_prior = Normal(z_variational_posterior.loc.detach(), z_variational_posterior.scale.detach())
        self.last_risk_x = Risk_X
        self.last_risk_y = Risk_Y

        return loss, recon_loss, kl_loss
    def _loss(self, p_y_pred, y_target, posterior, prior):

        negative_log_likelihood = F.mse_loss(p_y_pred, y_target, reduction="mean")
        # KL has shape (batch_size, r_dim). Take mean over batch and sum over r_dim (since r_dim is dimension of normal distribution)
        kl = kl_divergence(posterior, prior).mean(dim=0).sum()

        return negative_log_likelihood + kl * self.kl_weight, negative_log_likelihood, kl

    def msd_diversified_score(self, Risk_X_candidate, acquisition_score, gamma_2, real_batch_size):
        with torch.no_grad():
            x = Risk_X_candidate.squeeze(0)  # bs, dim
            acquisition_score = acquisition_score.squeeze().detach()
            num_candidates = len(x)

            S = []
            while len(S) < real_batch_size:
                phi_us = torch.full((num_candidates,), -float('inf'), device=x.device)
                fus = acquisition_score / 2

                if len(S) > 0:
                    x_S = x[S]
                    S_dus = torch.norm(x_S[:, None, :] - x_S[None, :, :], dim=-1).min()
                    dus = (torch.norm(x[:, None, :] - x_S, dim=-1).min(dim=1)[0]) * gamma_2
                else:
                    dus = torch.zeros(num_candidates, device=x.device)

                phi_us = fus + dus

                phi_us[S] = -float('inf')
                assert torch.argmax(phi_us).item() not in S
                S.append(torch.argmax(phi_us).item())

            S = np.array(S)
            x_expanded = x[S]  # (real_batch_size, dim)
            x_diff = x_expanded[:, None, :] - x_expanded[None, :, :]  # (real_batch_size, real_batch_size, dim)
            local_diverse_score = torch.norm(x_diff, dim=-1).sum() / ((real_batch_size) * (real_batch_size - 1))

        return S, acquisition_score[S].sum().item() + local_diverse_score.item(), local_diverse_score.item(), acquisition_score[S].sum().item()

    def acquisition_function(self, Risk_X_candidate, gamma_0=1.0, gamma_1=1.0, gamma_2=0.0, pure_acquisition=False, real_batch_size=None):

        Risk_X_candidate = Risk_X_candidate.to(self.device)
        x = Risk_X_candidate.unsqueeze(0)
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        # Shape: 1 * 100 * 2

        if self.last_risk_x is None:
            z_sample = self.z_prior.rsample([self.num_samples])
        else:
            _, z_variational_posterior = self.risklearner(self.last_risk_x, self.last_risk_y, self.output_type)
            z_posterior = Normal(z_variational_posterior.loc.detach(), z_variational_posterior.scale.detach())
            z_sample = z_posterior.rsample([self.num_samples]) 
        # Shape: num_samples * 1 * 10

        p_y_pred = self.risklearner.xz_to_y(x, z_sample, self.output_type)
        # Shape: num_samples * batch_size * 1

        output_mu = torch.mean(p_y_pred, dim=0)#bs, 1
        output_sigma = torch.std(p_y_pred, dim=0)#bs, 1

        if self.posterior_sampling:
            acquisition_score = output_mu
        else:
            acquisition_score = gamma_0 * output_mu + gamma_1 * output_sigma

        if pure_acquisition or self.diversity_type is None:
            return acquisition_score, output_mu, output_sigma

        best_batch_id, diversified_score, combine_local_diverse_score, combine_local_acquisition_score = self.msd_diversified_score(x, acquisition_score, gamma_2, real_batch_size)

        return best_batch_id, diversified_score, combine_local_diverse_score, combine_local_acquisition_score, acquisition_score


        