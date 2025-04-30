# PDTS-Sinusoid

This repository implements the PDTS and baselines proposed in the paper **"Fast and Robust: Task Sampling with Posterior and Diversity Synergies for Adaptive Decision-Makers in Randomized Environments"**, focusing on sinusoid regression scenario. The code is adapted from https://github.com/thu-rllab/MPTS .


## 🚀 Quick Start

### Installation
```bash
conda create -n pdts_sinu python=3.7 -y
conda activate pdts_sinu

# Install dependencies
pip install -r requirements.txt
```

## 🔧 Benchmarking All Methods
### Command Line Examples
| Method | Command |
|--------|---------|
| **PDTS** (Ours) | `python main.py --gpu_id 0 --log_name logs/pdts --sampling_strategy pdts --num_candidates 512 --global_seed 1` |
| MPTS   | `python main.py --gpu_id 0 --log_name logs/mpts --sampling_strategy mpts --num_candidates 32 --global_seed 1` |
| ERM    | `python main.py --gpu_id 0 --log_name logs/erm --sampling_strategy erm --num_candidates 16 --global_seed 1` |
| DRM    | `python main.py --gpu_id 0 --log_name logs/drm --sampling_strategy drm --num_candidates 32 --global_seed 1` |
| GDRM   | `python main.py --gpu_id 0 --log_name logs/gdrm --sampling_strategy gdrm --num_candidates 16 --global_seed 1` |

