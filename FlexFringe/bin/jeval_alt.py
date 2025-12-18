#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate eval job script and send to slurm.

Example Usage:
    $ jeval.py --ini edsm -o ../FlexFringe/evals-short -m ../FlexFringe/models-short
"""
import os
import sys
import time
from glob import glob
from contextlib import suppress

import pandas as pd

from src.bin.flexfringe import FF_DIR
from src.core.context import Context
from src.core.app import harness
from src.core import slurm


EVAL_BIN = os.path.abspath(os.path.join(os.path.dirname(__file__), "eval.py"))
TMP_DIR = "/gpfs/projects/HeinzGroup/tmp"


def get_missing_models(modeldir: str) -> set[str]:
    gstr = os.path.join(os.path.realpath(modeldir), f"*.final.json")
    all_models = {p.replace(".final.json", "") for p in glob(gstr) if "_64" not in p}

    combined = pd.DataFrame()
    with suppress(FileNotFoundError):
        combined = pd.read_csv("eval_combined.csv").dropna(subset = ["model_path"])

    if not combined.empty:
        all_evals = {p.replace(".final.json", "") for p in combined["model_path"]}

    estr = os.path.join(os.path.realpath(modeldir), f"*_eval.csv")
    all_evals += {p.replace("_eval.csv", "") for p in glob(estr)}
    
    if not len(all_evals) == 0:
        return all_models - all_evals
    return all_models


# TODO: Maybe this should just merge with eval.py?
def main(ctx: Context) -> None:
    partitions = slurm.sinfo().PARTITION.unique()
    #ctx.parser.add_argument("-o", "--outdir", default=os.path.join(FF_DIR, "evals"))
    ctx.parser.add_argument("-m", "--modeldir", default=os.path.join(FF_DIR, "models"))
    ctx.parser.add_argument(
        "-p", "--partition", choices=partitions, default="short-28core"
    )
    ctx.parser.add_argument(
        "-y", "--dryrun", action="store_true", help="don't send to slurm"
    )
    ctx.parser.add_argument("-e", "--email", default="logan.swanson@stonybrook.edu")

    ctx.parser.set_defaults(
        modules=["shared", "gnu-parallel/6.0", "anaconda/3", "gcc/12.1.0"]
    )
    args = ctx.parser.parse_args()
    # Generate file for gnu-parallel.
    scriptname = os.path.basename(sys.argv[0]).replace(".py", "")
    cmdpath = time.strftime(os.path.join(TMP_DIR, f"{scriptname}.%Y%m%d.%H%M%S.txt"))
    cmds = []
    for path in get_missing_models(args.modeldir):
        outpath = path + "_eval.csv"
        cmds.append(f"{EVAL_BIN} {path}.final.json -o {outpath}")
    ctx.log.info("writing: %s", cmdpath)
    with open(cmdpath, "w") as fd:
        fd.write("\n".join(cmds))
    # Submit job to slurm.
    
    # XXX: This logic is also in bin/ffgen.py. Should be pulled into the slurm module.
    cores = 28 if "28" in args.partition else 24
    nodes = 8 if "medium" in args.partition else 1
    nodes = 24 if "large" in args.partition else nodes

    name = os.path.basename(args.modeldir)
    slurm.sbatch(
        # f"cat {cmdpath} | parallel --tmpdir={TMP_DIR} -l1 srun -N1 -n1 sh -c '$@' --",
        f"cat {cmdpath} | parallel --tmpdir={TMP_DIR} -P {cores}",
        flags={
            "ntasks-per-node": cores,
            "nodes": nodes,
            "time": slurm.timelimit(args.partition),
            "partition": args.partition,
            "output": f"outfiles/{name}_evals.log",
            "job-name": f"{name}-evals",
            "mail-type": "BEGIN,END",
            "mail-user": args.email,
        },
        modules=args.modules,
        dryrun=args.dryrun,
    )


if __name__ == "__main__":
    harness(main)
