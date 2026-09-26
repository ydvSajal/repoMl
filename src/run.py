"""Pipeline entry point (owner: Sajal, tasks S2/S5). Placeholder from S0.

python -m src.run --split {val,test} [--stage {normalize,block,features,train,decide,all}] [--baseline] [--force]
"""
import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--stage", choices=["normalize", "block", "features", "train", "decide", "all"], default="all")
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--force", action="store_true")
    p.parse_args()
    raise NotImplementedError("S2/S5 (Sajal)")


if __name__ == "__main__":
    main()
