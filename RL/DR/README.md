# PDTS-DR

This repository implements the PDTS and baselines proposed in the paper **"Fast and Robust: Task Sampling with Posterior and Diversity Synergies for Adaptive Decision-Makers in Randomized Environments"**, focusing on physical robotics domain randomization scenarios. The code is adapted from https://github.com/montrealrobotics/active-domainrand .

## 🚀 Quick Start

### Installation
```bash
conda create -n pdts_dr python=3.7 -y
conda activate pdts_dr

# Install dependencies (mujoco210 required)
pip install -e .
# Install gym-ergojr
pip install git+https://github.com/fgolemo/gym-ergojr.git
# Install openai baselines
git clone https://github.com/openai/baselines.git
cd baselines
pip install tensorflow-gpu==1.14
pip install -e .
```

## 🔧 Benchmarking All Methods
### Command Line Examples
|Scenario| Method | Command |
|--------|--------|---------|
|Pusher|**PDTS** (Ours)| `python -m experiments.domainrand.experiment_driver pusher --continuous-svpg --svpg-rollout-length 10 --uniform_sample_steps 0.1 --sampler_multiplier 64 --sampling_gamma_2 0.05 --algo pdts --seed=10`|
||MPTS|`python -m experiments.domainrand.experiment_driver pusher --continuous-svpg --svpg-rollout-length 10 --uniform_sample_steps 0.1 --sampler_multiplier 5 --algo mpts --seed=10`|
||ERM|`python -m experiments.domainrand.experiment_driver pusher --continuous-svpg --svpg-rollout-length 1 --uniform_sample_steps 1.0 --sampler_multiplier 1 --algo erm --seed=10`|
||DRM|`python -m experiments.domainrand.experiment_driver pusher --continuous-svpg --svpg-rollout-length 1 --uniform_sample_steps 1.0 --sampler_multiplier 1 --cvar 0.5 --algo drm --seed=10`|
||GDRM|`python -m experiments.domainrand.experiment_driver pusher --continuous-svpg --svpg-rollout-length 1 --uniform_sample_steps 1.0 --sampler_multiplier 1 --gdroweight 0.01 --algo gdrm --seed=10`|
|LunarLander|**PDTS** (Ours)| `python -m experiments.domainrand.experiment_driver lunar --continuous-svpg --svpg-rollout-length 10 --uniform_sample_steps 0.1 --sampler_multiplier 64 --sampling_gamma_2 8 --algo pdts --seed=10`|
||MPTS|`python -m experiments.domainrand.experiment_driver lunar --continuous-svpg --svpg-rollout-length 10 --uniform_sample_steps 0.1 --sampler_multiplier 2.5 --algo mpts --seed=10`|
||ERM|`python -m experiments.domainrand.experiment_driver lunar --continuous-svpg --svpg-rollout-length 1 --uniform_sample_steps 1.0 --sampler_multiplier 1 --algo erm --seed=10`|
||DRM|`python -m experiments.domainrand.experiment_driver lunar --continuous-svpg --svpg-rollout-length 1 --uniform_sample_steps 1.0 --sampler_multiplier 1 --cvar 0.5 --algo drm --seed=10`|
||GDRM|`python -m experiments.domainrand.experiment_driver lunar --continuous-svpg --svpg-rollout-length 1 --uniform_sample_steps 1.0 --sampler_multiplier 1 --gdroweight 0.01 --algo gdrm --seed=10`|
|ErgoReacher|**PDTS** (Ours)|`python -m experiments.domainrand.experiment_driver ergo --continuous-svpg --svpg-rollout-length 10 --uniform_sample_steps 0.1 --sampler_multiplier 64 --sampling_gamma_2 1 --algo pdts --seed=10`|
||MPTS|`python -m experiments.domainrand.experiment_driver ergo --continuous-svpg --svpg-rollout-length 10 --uniform_sample_steps 0.1 --sampler_multiplier 25 --algo mpts --seed=10`|
||ERM|`python -m experiments.domainrand.experiment_driver ergo --continuous-svpg --svpg-rollout-length 1 --uniform_sample_steps 1.0 --sampler_multiplier 1 --algo erm --seed=10`|
||DRM|`python -m experiments.domainrand.experiment_driver ergo --continuous-svpg --svpg-rollout-length 1 --uniform_sample_steps 1.0 --sampler_multiplier 1 --cvar 0.5 --algo drm --seed=10`|
||GDRM|`python -m experiments.domainrand.experiment_driver ergo --continuous-svpg --svpg-rollout-length 1 --uniform_sample_steps 1.0 --sampler_multiplier 1 --gdroweight 0.01 --algo gdrm --seed=10`|
