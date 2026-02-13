#!/bin/bash

echo "loading modules"
module load slurm
module load anaconda 
conda activate /gpfs/projects/HeinzGroup/caffeine_underground/data_ff

for file in $(ls ../small-grid-ini/0.0.1.0.0.0.searchdeep.0.ini); do
  echo "Processing file: $file"
  # Perform actions on $file
  python3 ffgen.py --ini $file \
	--modeldir ../FlexFringe/models-reg-large/${file##*/} \
	--datadir /gpfs/projects/HeinzGroup/asoubki/FlexFringe/data/MLRegTest/Large \
	--data-type Train.txt \
	--email sarah.payne@stonybrook.edu \
	--partitions extended-28core long-28core #medium-28core long-28core extended-28core large-28core 
done
