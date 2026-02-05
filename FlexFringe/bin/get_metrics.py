import pandas as pd

##rpni_evals = pd.read_csv("../FlexFringe/models-os/0.0.1.0.0.0.none.0.ini/eval_combined.csv")
##edsm_evals = pd.read_csv("../FlexFringe/models-os/1.0.1.0.0.0.none.0.ini/eval_combined.csv")
##
##print(edsm_evals.columns.tolist())
##
##edsm_evals = edsm_evals[edsm_evals["split"] != "Dev"]
##rpni_evals = rpni_evals[rpni_evals["split"] != "Dev"]
##print(len(edsm_evals))
##
##edsm_evals = edsm_evals[edsm_evals["alphabet_size"] != 64]
##print(len(edsm_evals))
##
##print("EDSM OS accuracy", edsm_evals["accuracy"].mean())
##print("EDSM OS f1", edsm_evals["f1"].mean())
##print("RPNI OS accuracy", rpni_evals["accuracy"].mean())
##print("RPNI OS f1", rpni_evals["f1"].mean())

# for FF models:
# remove NaNs?
# check for duplicated "model name"
# remove anything that isn't Dev
# remove anything where the corresponding Train accuracy isn't 1
# find avg precision and accuracy
# track best scores for RPNI and EDSM

prefix = "../FlexFringe/models-ps-small/"
suffix = ".ini/eval_combined.csv"


file_paths = []
# 0: RPNI, 1: EDSM
for heuristic in [0,1]:
    for reversetraces in [0]:
        for extend in [0,1]:
            for shallowfirst in [0,1]:
                for search in ["searchdeep", "searchlocal", "searchglobal", "searchpartial", "none"]:
                    for sinkson in [0]:
                        file_path = prefix + f"{heuristic}.{reversetraces}.{extend}.{shallowfirst}.0.0.{search}.{sinkson}"
                        if sinkson:
                            for sinkcount in [1,10,25]:
                                for mergesinkscore in [0,1]:
                                    new_file_path = file_path + f".{sinkcount}.{mergesinkscore}"
                                    file_paths.append(new_file_path + suffix)
                        else:
                            file_paths.append(file_path + suffix)

best_acc = 0
best_acc_cell = ""
best_f1 = 0
best_f1_cell = ""

best_acc_edsm = 0
best_acc_edsm_cell = ""
best_f1_edsm = 0
best_f1_edsm_cell = ""

best_acc_rpni = 0
best_acc_rpni_cell = ""
best_f1_rpni = 0
best_f1_rpni_cell = ""

for p in file_paths:
    #print("processing", p)

    try:
        evals = pd.read_csv(p)
    except:
        print("no eval found for", p)
        continue
    
    orig_len = len(evals)
    cols = evals.columns.tolist()
    cols.remove("last_modified")
    evals = evals.drop_duplicates(subset=cols)
    dedup_len = len(evals)
    if orig_len != dedup_len:
        print(f"removed {orig_len - dedup_len} duplicate rows")

    evals = evals.dropna()
    dropna_len = len(evals)
    if dedup_len != dropna_len:
        print(f"removed {dedup_len - dropna_len} nan rows")

##    failed = evals[evals["accuracy"] < 1]
##    failed = failed[failed["split"] == "Train"]

    dev_evals = evals[evals["split"] == "Dev"]
    acc = dev_evals["accuracy"].mean()
    f1 = dev_evals["f1"].mean()

    cell = p.replace(prefix, "").replace(suffix, "")
    if acc > best_acc:
        best_acc = acc
        best_acc_cell = cell

    if f1 > best_f1:
        best_f1 = f1
        best_f1_cell = cell

    if cell[0] == "0":
        if acc > best_acc_rpni:
            best_acc_rpni = acc
            best_acc_rpni_cell = cell
        if f1 > best_f1_rpni:
            best_f1_rpni = f1
            best_f1_rpni_cell = cell
    elif cell[0] == "1":
        if acc > best_acc_edsm:
            best_acc_edsm = acc
            best_acc_edsm_cell = cell
        if f1 > best_f1_edsm:
            best_f1_edsm = f1
            best_f1_edsm_cell = cell
    else:
        print("bad eval path:", p)

    # # find avg f1 and accuracy on Dev data
    # track best scores for RPNI and EDSM

print("best acc", best_acc, best_acc_cell)
print("best f1", best_f1, best_f1_cell)

print("best acc RPNI", best_acc_rpni, best_acc_rpni_cell)
print("best f1 RPNI", best_f1_rpni, best_f1_rpni_cell)

print("best acc EDSM", best_acc_edsm, best_acc_edsm_cell)
print("best f1 EDSM", best_f1_edsm, best_f1_edsm_cell)
