#!/bin/bash
#SBATCH --job-name=nltk
#SBATCH --output=/home/%u/out/nltk.%j.out
#SBATCH --error=/home/%u/err/nltk.%j.err
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=32GB
ml python
pip3 install --user nltk
pip3 install --user --upgrade certifi
python3 -m nltk.downloader all
python3 nltkNER.py
