#!/bin/bash

echo "loading modules"
module load slurm
module load anaconda 
conda activate /gpfs/projects/HeinzGroup/caffeine_underground/ffgen

for file in $(ls ../grid-ini/0.0.0.1.1.0*.ini); do
  echo "Processing file: $file"
  # Perform actions on $file
  python3 ffgen.py --ini $file \
	--modeldir ../FlexFringe/models-ps-small-grid/${file##*/} \
	--datadir /gpfs/projects/HeinzGroup/caffeine_underground/mlrt/data/PlusShort/ \
	--data-size Small \
	--data-type TrainPS \
	--partitions medium-28core #medium-28core long-28core extended-28core large-28core 
done
