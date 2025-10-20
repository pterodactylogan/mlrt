#!/bin/bash
#
#SBATCH --job-name=neural-test
#SBATCH --output=test.txt
#SBATCH --nodes=1
#SBATCH --time=30:00:00
#SBATCH -p long-28core
#SBATCH --mail-type=END
#SBATCH --mail-user=logan.swanson@stonybrook.edu

python3 ../subregular-learning/src/neural_net/tensorflow/main.py \
--model-file ../subregular-learning/test_spec.txt \
