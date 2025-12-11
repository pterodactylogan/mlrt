import os

from glob import glob

outfile = open("./standard_evals.csv", "w")
model_dir = "./models/Standard"

gstr = os.path.join(model_dir, f"**/*_eval.txt")

outfile.write("alph,tier,class,k,j,i,network_type,train_set_size,test_type,tp,fp,tn,fn,tpr,fpr,precision,fscore,accuracy,auc,brier")
outfile.write("\n")

for path in glob(gstr, recursive=True):
    specs = path.split("/")[-2]
    network_type = specs.split("_")[0]
    alph = specs.split("_")[1][0:2]
    tier = specs.split("_")[1][2:4]
    lang_class = specs.split("_")[1][4:-3]
    k = specs.split("_")[1][-3]
    j = specs.split("_")[1][-2]
    i = specs.split("_")[1][-1]
    train_set_size = specs.split("_")[-1]

    test_type = path.split("/")[-1][4:6]

    with open(path) as efile:
        metrics = {}
        for line in efile.readlines():
            met = (line.split(":")[0]).strip()
            val = (line.split(":")[1]).strip()
            metrics[met] = val

    outfile.write(",".join([
        alph, tier, lang_class, k, j, i, network_type,
        train_set_size, test_type, metrics["TP"], metrics["FP"],
        metrics["TN"], metrics["FN"], metrics["TPR"],
        metrics["FPR"], metrics["Precision"], metrics["F-score"],
        metrics["Accuracy"], metrics["AUC"], metrics["Brier"]
        ]))
    outfile.write("\n")


outfile.close()
