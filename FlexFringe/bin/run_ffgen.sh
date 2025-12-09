#!/bin/bash

echo "loading modules"
module load slurm
module load anaconda 
conda activate /gpfs/projects/HeinzGroup/caffeine_underground/ffgen

for file in $(ls ../grid-ini/0.0.1.*.ini); do
  echo "Processing file: $file"
  # Perform actions on $file
  python3 ffgen.py --ini $file \
	--modeldir ../FlexFringe/models-ps-small/${file##*/} \
	--datadir /gpfs/projects/HeinzGroup/caffeine_underground/mlrt/data/PlusShort/Small \
	--data-type TrainPS.ff \
	--email logan.swanson@stonybrook.edu \
	--partitions medium-28core #medium-28core long-28core extended-28core large-28core 
done
