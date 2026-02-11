import pandas as pd


prefix = "../FlexFringe/models-ps-small/"
suffix = ".ini/eval_combined.csv"

cell = "0.0.1.0.0.0.searchdeep.0"

test_types = ["TestSR", "TestLR", "TestSA", "TestLA"]

evals = pd.read_csv(prefix+cell+suffix)

##orig_len = len(evals)
##cols = evals.columns.tolist()
##cols.remove("last_modified")
##evals = evals.drop_duplicates(subset=cols)
##dedup_len = len(evals)
##if orig_len != dedup_len:
##    print(f"removed {orig_len - dedup_len} duplicate rows")
##
##evals = evals.dropna()
##dropna_len = len(evals)
##if dedup_len != dropna_len:
##    print(f"removed {dedup_len - dropna_len} nan rows")

SR_evals = evals[evals["split"] == "TestSR"]
print("SR:", SR_evals["f1"].mean())
LR_evals = evals[evals["split"] == "TestLR"]
print("LR:", LR_evals["f1"].mean())
SA_evals = evals[evals["split"] == "TestSA"]
print("SA:", SA_evals["f1"].mean())
LA_evals = evals[evals["split"] == "TestLA"]
print("LA:", LA_evals["f1"].mean())

