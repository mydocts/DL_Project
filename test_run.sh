#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate DL_Project
export HF_ENDPOINT=https://hf-mirror.com
python evaluate.py
