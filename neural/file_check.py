import json

all_specs = json.loads(open("./spec_files_50_PS/specs_7.txt").read())

for spec in all_specs:
    train_data = spec["train-data"]
    with open(train_data, 'r') as f:
        for line in f.readlines():
            try:
                x, y = line.strip().split('\t')
            except:
                print("file:", train_data)
                print("line:", line)
