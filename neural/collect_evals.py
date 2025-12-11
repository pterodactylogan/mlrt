import os

from glob import glob

import pandas as pd

model_dir = "./models/Standard"

gstr = os.path.join(model_dir, f"**/*_eval.txt")

for path in glob(gstr, recursive=True):
    print(path)
