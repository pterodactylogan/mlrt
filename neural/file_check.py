import json
import os

spec_folder = "./spec_files_50_PS"
na_file = open("NAs.csv")
names = []

for line in na_file.readlines():
    _, alph, t, lang, w, thresh, i, *_ = line.split(",")
    lang = lang.strip('\"')
    if lang == "language_class":
        continue

    if len(alph) == 1:
        alph = "0"+alph
    if len(t) == 1:
        t= "0"+t
    names.append("Small/" + ".".join([alph, t, lang, w, thresh, i]) + "_TrainPS")

for f in os.listdir(spec_folder):
    specs = open(spec_folder + "/" + f).read()
    for name in names:
        if name in specs:
            print(name, f)
    


##all_specs = json.loads(open("./spec_files_50_PS/specs_7.txt").read())
##
##for spec in all_specs:
##    train_data = spec["train-data"]
##    with open(train_data, 'r') as f:
##        for line in f.readlines():
##            try:
##                x, y = line.strip().split('\t')
##            except:
##                print("file:", train_data)
##                print("line:", line)
