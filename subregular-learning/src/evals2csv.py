import csv
from enum import Enum

class StrLen(Enum):
    OS = 1 # only short
    PS = 2 # plus short
    STD = 3 # standard

lines = open("all_evals.txt", "r").readlines()
tests = {}
for line in lines:
    test = line.strip().split(":")[0].replace("models/", "")
    test = test.replace("OnlyShort/", "")
    test = test.replace("PlusShort/", "")
    test = test.replace("Standard/", "")
    
    if test not in tests:
        tests[test] = {}
    metric = line.strip().split(":")[1]
    value = line.strip().split()[1]
    tests[test][metric] = value

csv_fname = "all_evals.csv"
with open(csv_fname, "w", newline="\n") as f:
    writer = csv.writer(f)
    writer.writerows([[
        "alph",
        "tier",
        "class",
        "k",
        "j",
        "i",
        "direction",
        "network_type",
        "drop",
        "train_set_size",
        "test_type",
        "tp",
        "fp",
        "tn",
        "fn",
        "tpr",
        "fpr",
        "precision",
        "fscore",
        "accuracy",
        "auc",
        "brier"
    ]])

for test in tests:
    model = test.split("/")[0]
    network_type = model.split("_")[0]
    lang = model.split("_")[1]

    if "OS" not in model:
        train_set_size = model.split("_")[3]
    else:
        train_set_size = "Small"

    alph = lang[0:2]
    tier = lang[2:4]
    lang_class = lang[4:-3]
    k = lang[-3]
    j = lang[-2]
    lang_i = lang[-1]

    test_type = test.split("/")[1].split("_")[0].replace("Test", "")

    with open(csv_fname, "a", newline="\n") as f:
        writer = csv.writer(f)
        try:
            writer.writerows([[
                alph,
                tier,
                lang_class,
                k,
                j,
                lang_i,
                network_type,
                train_set_size,
                test_type,
                tests[test]["TP"],
                tests[test]["FP"],
                tests[test]["TN"],
                tests[test]["FN"],
                tests[test]["TPR"],
                tests[test]["FPR"],
                tests[test]["Precision"],
                tests[test]["F-score"],
                tests[test]["Accuracy"],
                tests[test]["AUC"],
                tests[test]["Brier"]
            ]])
        except KeyError:
            pass
