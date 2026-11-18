#!/usr/bin/env python3
from __future__ import annotations
"""
Generate CPR config JSON files and run sweeps with OPM flow.

Covers the full 3×2×2×2 action space:
  weight_type  : quasiimpes | trueimpes | trueimpesanalytic
  precond type : cpr | cprw
  fine smoother: paroverilu0 | dilu
  coarse smoother (AMG): ilu0 | dilu

Usage
-----
python run_sweep.py \\
    --flow   /path/to/flow \\
    --decks  /path/to/SPE9.DATA /path/to/NORNE_ATW2013.DATA \\
    --out    /tmp/sweep2

Outputs one CSV per (deck, config) in --out, named
  <deck_stem>_<label>.csv

Reuse-interval sweep
--------------------
To sweep reuse intervals for one fixed CPR configuration, use for example:
  python run_sweep.py --flow /path/to/flow \
      --decks /path/to/SPE9.DATA --out /tmp/reuse_spe9 \
      --reuse-config-label 12 --reuse-intervals 2 4 8 16 30 60

This writes <deck>_reuse_<interval>.csv files and forces
--cpr-reuse-setup=4 --cpr-reuse-interval=<interval>.  Use label 12 for decks
with wells and label 8 for decks without wells.
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Legacy 24-config space.  The deployable joint sweep below uses the reduced
# 12-config space because CPR and CPRW are equivalent for the chosen deck mode.
# ---------------------------------------------------------------------------

WEIGHT_TYPES   = ["quasiimpes", "trueimpes", "trueimpesanalytic"]
PRECOND_TYPES  = ["cpr", "cprw"]
FINE_SMOOTHERS = ["paroverilu0", "dilu"]
COARSE_SMOOTHERS = ["ilu0", "dilu"]

ALL_CONFIGS = []
for dec, weight in enumerate(WEIGHT_TYPES):
    for cprw_idx, ptype in enumerate(PRECOND_TYPES):
        for fine_idx, fine in enumerate(FINE_SMOOTHERS):
            for coarse_idx, coarse in enumerate(COARSE_SMOOTHERS):
                label = dec * 8 + cprw_idx * 4 + fine_idx * 2 + coarse_idx
                ALL_CONFIGS.append({
                    "label":  label,
                    "name":   f"{ptype}:{weight}:{fine}:{coarse}",
                    "ptype":  ptype,
                    "weight": weight,
                    "fine":   fine,
                    "coarse": coarse,
                })

assert len(ALL_CONFIGS) == 24
assert len({c["label"] for c in ALL_CONFIGS}) == 24


def reduced_configs(ptype: str) -> list[dict]:
    """The 12 non-redundant CPR actions for a fixed CPR/CPRW mode."""
    configs = []
    for dec, weight in enumerate(WEIGHT_TYPES):
        for fine_idx, fine in enumerate(FINE_SMOOTHERS):
            for coarse_idx, coarse in enumerate(COARSE_SMOOTHERS):
                label = dec * 4 + fine_idx * 2 + coarse_idx
                configs.append({
                    "label": label,
                    "name": f"{ptype}:{weight}:{fine}:{coarse}",
                    "ptype": ptype,
                    "weight": weight,
                    "fine": fine,
                    "coarse": coarse,
                })
    return configs


# ---------------------------------------------------------------------------
# JSON config generation
# ---------------------------------------------------------------------------

AMG_DEFAULTS = {
    "type":              "amg",
    "alpha":             0.333333333333,
    "relaxation":        1.0,
    "iterations":        1,
    "coarsenTarget":     1200,
    "pre_smooth":        1,
    "post_smooth":       1,
    "beta":              0.0,
    "verbosity":         0,
    "maxlevel":          15,
    "skip_isolated":     0,
    "accumulate":        1,
    "prolongationdamping": 1.0,
    "maxdistance":       2,
    "maxconnectivity":   15,
    "maxaggsize":        6,
    "minaggsize":        4,
}


def make_json(cfg: dict, tol: float = 0.005, maxiter: int = 20) -> dict:
    # tol/maxiter are flow's CPR defaults (cprDefaultReduction/MaxIter in
    # setupPropertyTree.cpp) and the cprw well settings match setupCPRW(), so
    # label 12 reproduces the default solver exactly.
    amg = dict(AMG_DEFAULTS)
    amg["smoother"] = cfg["coarse"]
    if "amg_damping" in cfg:
        amg["prolongationdamping"] = cfg["amg_damping"]
    if "amg_target" in cfg:
        amg["coarsenTarget"] = cfg["amg_target"]

    prec: dict = {
        "type":        cfg["ptype"],
        "weight_type": cfg["weight"],
        "pre_smooth":  0,
        "post_smooth": 1,
        "finesmoother": {
            "type":       cfg["fine"],
            "relaxation": 1.0,
        },
        "verbosity": 0,
        "coarsesolver": {
            "maxiter":       1,
            "tol":           0.1,
            "solver":        "loopsolver",
            "verbosity":     0,
            "preconditioner": amg,
        },
    }
    if cfg["ptype"] == "cprw":
        prec["use_well_weights"] = "false"
        prec["add_wells"]        = "true"

    return {
        "maxiter":       maxiter,
        "tol":           tol,
        "verbosity":     0,
        "solver":        "bicgstab",
        "preconditioner": prec,
    }


def write_configs(config_dir: Path, configs: list[dict]) -> dict[int, Path]:
    config_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for cfg in configs:
        p = config_dir / f"label_{cfg['label']:02d}.json"
        p.write_text(json.dumps(make_json(cfg), indent=2))
        paths[cfg["label"]] = p
    return paths


def write_joint_configs(config_dir: Path, configs: list[dict], dampings: list[float],
                        targets: list[int]) -> dict[tuple[int, float, int], Path]:
    config_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for cfg in configs:
        for damping in dampings:
            for target in targets:
                action = dict(cfg, amg_damping=damping, amg_target=target)
                tag = f"pd{damping:g}_ct{target}"
                path = config_dir / f"label_{cfg['label']:02d}_{tag}.json"
                path.write_text(json.dumps(make_json(action), indent=2))
                paths[(cfg["label"], damping, target)] = path
    return paths


# ---------------------------------------------------------------------------
# Running flow
# ---------------------------------------------------------------------------

def run_one(flow_cmd: list[str], deck: str, cfg: dict, json_path: Path,
            out_dir: Path, csv_path: Path,
            reuse_interval: int | None = None) -> tuple[int, bool, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "flow.log"
    env = os.environ.copy()
    env["OPM_CPR_SWEEP_LOG"] = str(csv_path)

    cmd = [*flow_cmd, f"--linear-solver={json_path}"]
    if reuse_interval is not None:
        cmd += ["--cpr-reuse-setup=4", f"--cpr-reuse-interval={reuse_interval}"]
    # Default CPR setup reuse, unless an explicit interval was requested; the
    # CSV's setup_rebuilt column marks the solves that rebuilt the setup.
    cmd += [f"--output-dir={out_dir}", deck]
    with open(log, "w") as fh:
        result = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return cfg["label"], result.returncode == 0, cfg["name"]


def run_reuse_one(flow_cmd: list[str], deck: str, cfg: dict, json_path: Path,
                  interval: int, out_dir: Path, csv_path: Path
                  ) -> tuple[int, bool, str]:
    """Run one fixed CPR configuration with one explicit reuse interval."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "flow.log"
    env = os.environ.copy()
    env["OPM_CPR_SWEEP_LOG"] = str(csv_path)

    cmd = [
        *flow_cmd,
        f"--linear-solver={json_path}",
        "--cpr-reuse-setup=4",
        f"--cpr-reuse-interval={interval}",
        f"--output-dir={out_dir}",
        deck,
    ]
    with open(log, "w") as fh:
        result = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return interval, result.returncode == 0, cfg["name"]


def run_joint_one(flow_cmd: list[str], deck: str, cfg: dict, json_path: Path,
                  damping: float, target: int, reuse_interval: int | None,
                  out_dir: Path, csv_path: Path) -> tuple[int, float, int, int | None,
                                                          bool, str]:
    """Run one reduced CPR × AMG configuration, optionally with reuse."""
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OPM_CPR_SWEEP_LOG"] = str(csv_path)
    cmd = [*flow_cmd, f"--linear-solver={json_path}",
           *([] if reuse_interval is None else ["--cpr-reuse-setup=4",
                                                 f"--cpr-reuse-interval={reuse_interval}" ]),
           f"--output-dir={out_dir}", deck]
    with open(out_dir / "flow.log", "w") as fh:
        result = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return (cfg["label"], damping, target, reuse_interval,
            result.returncode == 0, cfg["name"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flow",    required=True, help="Path to flow executable")
    parser.add_argument("--nprocs",  type=int, default=1,
                        help="MPI ranks per flow run via mpirun (default: 1 = serial)")
    parser.add_argument("--decks",   nargs="+", required=True,
                        help="Reservoir deck files to sweep (.DATA)")
    parser.add_argument("--out",     default="/tmp/sweep2",
                        help="Output directory (default: /tmp/sweep2)")
    parser.add_argument("--jobs",    type=int, default=None,
                        help="Max parallel flow processes (default: nCPU / nprocs)")
    parser.add_argument("--labels",  nargs="*", type=int,
                        help="Subset of labels (0-23 legacy, 0-11 joint)")
    parser.add_argument("--reuse-config-label", type=int, default=None,
                        help="Fixed CPR label for --reuse-intervals (default: 12; "
                             "use 8 for decks without wells)")
    parser.add_argument("--reuse-config-labels", nargs="+", type=int,
                        help="Multiple fixed CPR labels for --reuse-intervals")
    parser.add_argument("--reuse-intervals", nargs="+", type=int,
                        help="Run interval sweep for one fixed CPR label, e.g. "
                             "2 4 8 16 30 60")
    parser.add_argument("--reuse-interval", type=int,
                        help="Use one explicit interval for the normal 24-config "
                             "sweep, e.g. 32")
    parser.add_argument("--joint-cpr-amg", action="store_true",
                        help="Sweep 12 CPR actions × AMG damping/target combinations")
    parser.add_argument("--cpr-type", choices=["cpr", "cprw"], default="cprw",
                        help="CPR mode for --joint-cpr-amg (default: cprw)")
    parser.add_argument("--amg-dampings", nargs="+", type=float,
                        default=[1.0, 1.3, 1.6],
                        help="AMG prolongation damping values")
    parser.add_argument("--amg-targets", nargs="+", type=int,
                        default=[1200, 600, 300],
                        help="AMG coarsening targets")
    args = parser.parse_args()

    if args.reuse_interval is not None and args.reuse_intervals is not None:
        parser.error("use either --reuse-interval or --reuse-intervals, not both")
    if args.reuse_interval is not None and args.reuse_interval <= 0:
        parser.error("--reuse-interval must be positive")
    if args.reuse_config_labels is not None and args.reuse_config_label is not None:
        parser.error("use either --reuse-config-label or --reuse-config-labels, not both")

    out_dir    = Path(args.out)
    config_dir = out_dir / "configs"

    # Build the flow command: prepend mpirun when --nprocs > 1.
    if args.nprocs > 1:
        flow_cmd = ["mpirun", "-np", str(args.nprocs),
                    "--bind-to", "none", "--mca", "btl", "self,sm",
                    args.flow]
    else:
        flow_cmd = [args.flow]

    # Default parallelism: don't over-subscribe — each MPI job uses nprocs cores.
    max_jobs = args.jobs or max(1, (os.cpu_count() or 4) // args.nprocs)

    configs = reduced_configs(args.cpr_type) if args.joint_cpr_amg else ALL_CONFIGS
    if args.labels:
        configs = [c for c in configs if c["label"] in args.labels]

    if args.joint_cpr_amg:
        if not args.amg_dampings or not args.amg_targets:
            parser.error("joint sweep needs at least one damping and target")
        print(f"Writing {len(configs) * len(args.amg_dampings) * len(args.amg_targets)} "
              f"joint JSON configs → {config_dir}")
        json_paths = write_joint_configs(config_dir, configs,
                                         list(dict.fromkeys(args.amg_dampings)),
                                         list(dict.fromkeys(args.amg_targets)))
    else:
        print(f"Writing {len(configs)} JSON configs → {config_dir}")
        json_paths = write_configs(config_dir, configs)

    if args.reuse_intervals is not None and not args.joint_cpr_amg:
        if any(interval <= 0 for interval in args.reuse_intervals):
            parser.error("--reuse-intervals values must be positive")
        intervals = list(dict.fromkeys(args.reuse_intervals))
        labels = (args.reuse_config_labels if args.reuse_config_labels is not None
                  else [12 if args.reuse_config_label is None else args.reuse_config_label])
        if any(not 0 <= label < len(ALL_CONFIGS) for label in labels):
            parser.error("reuse configuration labels must be between 0 and 23")
        labels = list(dict.fromkeys(labels))
        tasks = []
        for deck in args.decks:
            deck_stem = Path(deck).stem
            for label in labels:
                cfg = ALL_CONFIGS[label]
                for interval in intervals:
                    run_out = out_dir / deck_stem / f"label_{label:02d}_reuse_{interval:02d}"
                    csv_path = out_dir / f"{deck_stem}_label_{label:02d}_reuse_{interval:02d}.csv"
                    tasks.append((deck, cfg, json_paths[label], interval,
                                  run_out, csv_path))

        print(f"Reuse-interval sweep: labels={','.join(f'{label:02d}' for label in labels)}")
        print(f"Intervals: {', '.join(map(str, intervals))}")
        print(f"Launching {len(tasks)} runs across {len(args.decks)} deck(s), "
              f"≤{max_jobs} in parallel (nprocs={args.nprocs}) ...")

        ok = 0
        fail = 0
        with ThreadPoolExecutor(max_workers=max_jobs) as pool:
            futures = {
                pool.submit(run_reuse_one, flow_cmd, deck, cfg, jp,
                            interval, rod, csv): (deck, interval)
                for deck, cfg, jp, interval, rod, csv in tasks
            }
            for fut in as_completed(futures):
                deck, interval = futures[fut]
                got_interval, success, name = fut.result()
                deck_stem = Path(deck).stem
                status = "OK " if success else "FAIL"
                print(f"  [{status}] {deck_stem}  interval={got_interval:02d}  {name}")
                if success:
                    ok += 1
                else:
                    fail += 1

        print(f"\nDone: {ok} succeeded, {fail} failed.")
        if fail:
            print("Check flow.log files in the run subdirectories for errors.")
        print(f"\nCSV output: {out_dir}/*_reuse_<interval>.csv")
        return

    if args.joint_cpr_amg:
        dampings = list(dict.fromkeys(args.amg_dampings))
        targets = list(dict.fromkeys(args.amg_targets))
        intervals = ([None] if args.reuse_intervals is None else
                     list(dict.fromkeys(args.reuse_intervals)))
        if any(interval is not None and interval <= 0 for interval in intervals):
            parser.error("--reuse-intervals values must be positive")
        tasks = []
        for deck in args.decks:
            deck_stem = Path(deck).stem
            for cfg in configs:
                for damping in dampings:
                    for target in targets:
                        for interval in intervals:
                            tag = f"pd{damping:g}_ct{target}"
                            if interval is not None:
                                tag += f"_ri{interval}"
                            run_out = out_dir / deck_stem / f"label_{cfg['label']:02d}_{tag}"
                            csv_path = out_dir / f"{deck_stem}_label_{cfg['label']:02d}_{tag}.csv"
                            tasks.append((deck, cfg, json_paths[(cfg["label"], damping, target)],
                                          damping, target, interval, run_out, csv_path))
        print(f"Joint CPR×AMG sweep: {len(configs)} CPR × {len(dampings)} damping × "
              f"{len(targets)} targets × {len(intervals)} reuse interval(s)")
        print(f"Launching {len(tasks)} runs across {len(args.decks)} deck(s), "
              f"≤{max_jobs} in parallel (nprocs={args.nprocs}) ...")
        ok = fail = 0
        with ThreadPoolExecutor(max_workers=max_jobs) as pool:
            futures = {
                pool.submit(run_joint_one, flow_cmd, deck, cfg, jp, damping,
                            target, interval, run_out, csv):
                    (deck, cfg, damping, target, interval)
                for deck, cfg, jp, damping, target, interval, run_out, csv in tasks
            }
            for fut in as_completed(futures):
                deck, cfg, damping, target, interval = futures[fut]
                label, damping, target, interval, success, name = fut.result()
                status = "OK " if success else "FAIL"
                interval_text = "" if interval is None else f" interval={interval}"
                print(f"  [{status}] {Path(deck).stem} label={label:02d} "
                      f"damping={damping:g} target={target}{interval_text} {name}")
                ok += int(success); fail += int(not success)
        print(f"\nDone: {ok} succeeded, {fail} failed.")
        if fail:
            print("Check flow.log files in the run subdirectories for errors.")
        print(f"\nCSV output: {out_dir}/*_label_<label>_pd<damping>_ct<target>[_ri<interval>].csv")
        return

    tasks = []
    for deck in args.decks:
        deck_stem = Path(deck).stem
        for cfg in configs:
            label     = cfg["label"]
            run_out   = out_dir / deck_stem / f"label_{label:02d}"
            csv_path  = out_dir / f"{deck_stem}_{label:02d}.csv"
            tasks.append((deck, cfg, json_paths[label], run_out, csv_path))

    print(f"Launching {len(tasks)} runs across {len(args.decks)} deck(s), "
          f"≤{max_jobs} in parallel (nprocs={args.nprocs}) ...")

    ok = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=max_jobs) as pool:
        futures = {
            pool.submit(run_one, flow_cmd, deck, cfg, jp, rod, csv,
                        args.reuse_interval): (deck, cfg)
            for deck, cfg, jp, rod, csv in tasks
        }
        for fut in as_completed(futures):
            deck, cfg = futures[fut]
            label, success, name = fut.result()
            deck_stem = Path(deck).stem
            status = "OK " if success else "FAIL"
            print(f"  [{status}] {deck_stem}  label={label:02d}  {name}")
            if success:
                ok += 1
            else:
                fail += 1

    print(f"\nDone: {ok} succeeded, {fail} failed.")
    if fail:
        print("Check flow.log files in the run subdirectories for errors.")

    print(f"\nCSV output: {out_dir}/*.csv")


if __name__ == "__main__":
    main()
