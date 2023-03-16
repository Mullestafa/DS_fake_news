#!/bin/bash
#SBATCH --job-name=fn-preprocessing_job
#SBATCH --output=fn-preprocessing_job.out
#SBATCH --error=fn-preprocessing_job.err
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=64

# activate conda environment
source activate python3

# run Python script
python preproccess_a1.py --input data/news_cleaned_2018_02_13.csv --output fake_news_output.csv