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

# RPNI
file_paths = []
for heuristic in [0,1]:
    for reversetraces in [0,1]:
        for extend in [0,1]:
            for shallowfirst in [0,1]:
                for search in ["searchdeep", "searchlocal", "searchglobal", "searchpartial", "none"]:
                    for sinkson in [0,1]:
                        file_path = prefix + f"{heuristic}.{reversetraces}.{extend}.{shallowfirst}.0.0.{search}.{sinkson}"
                        if sinkson:
                            for sinkcount in [1,10,25]:
                                for mergesinkscore in [0,1]:
                                    new_file_path = file_path + f".{sinkcount}.{mergesinkscore}"
                                    file_paths.append(new_file_path + suffix)
                        else:
                            file_paths.append(file_path + suffix)

successfull = []
for p in file_paths:
    print("processing", p)
    
    evals = pd.read_csv(p)
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

    failed = evals[evals["accuracy"] < 1]
    failed = failed[failed["split"] == "Train"]
    if len(failed) == 0:
        successfull.append(p)
        
##    print(failed[["split", "alphabet_size", "tier_size", "language_class",
##                  "factor_width", "threshold", "index", "accuracy"]])

print("\n".join(successfull))
