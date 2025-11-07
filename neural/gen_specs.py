from itertools import product
import json

specs = []

name_files = ["onlyshort_names.txt",
              "both_small_names.txt",
              "both_mid_names.txt",
              "both_large_names.txt",
              "long_small_names.txt",
              "long_mid_names.txt",
              "long_large_names.txt"]
network_types = ["simple", "gru", "lstm", "transformer"]

num_models = 0
for f in name_files:
    for line in open("../data_filenames/" + f).readlines():
        if "Train" not in line or line.strip()[-3:] == ".ff":
            continue
        if "both" in f and ".mlrt" not in line:
            continue
        for net_type in network_types:
            if "onlyshort" in f:
                path = "../data/OnlyShort/"
            elif "both" in f:
                path = "../data/PlusShort/"
            elif "long" in f:
                path = "../../../subregular-learning/data_gen/"
            

            test_path = "../../../subregular-learning/data_gen/"
            val_path = "../../../subregular-learning/data_gen/"
            model_dir = "./models/"
            if "onlyshort" in f:
                model_dir += "OnlyShort/"
            elif "both" in f:
                model_dir += "PlusShort/"
            elif "long" in f:
                model_dir += "Standard/"
            model_dir += net_type + "_" + "".join(line.split(".")[:-1])

            if "small" in f:
                path += "Small/"
                val_path += "Small/"
                test_path += "Large/"
                model_dir += "_Small"
            elif "mid" in f:
                path += "Mid/"
                val_path += "Mid/"
                test_path += "Large/"
                model_dir += "_Mid"
            elif "large" in f:
                path += "Large/"
                val_path += "Large/"
                test_path += "Large/"
                model_dir += "_Large"
            elif "onlyshort" in f:
                test_path += "Large/"
                val_path += "Small/"


            specs.append({
                "model-type": net_type,
                "train-data": path + line.strip(),
                "val-data": val_path + line.strip().split("Train")[0] + "Dev.txt",
                "eval-data": test_path + line.strip().split("Train")[0] + "Test",
                "model-dir": model_dir
            })

            num_models += 1
            if num_models % 50 == 0:
                with open(f"./spec_files_50/specs_{num_models // 50}.txt", "w") as out_file:
                    json.dump(specs, out_file)
                specs = []
