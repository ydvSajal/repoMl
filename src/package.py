"""Submission zip builder (owner: Sajal, task S7). Placeholder from S0.

python -m src.package --team <team_name>
"""
import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--team", required=True)
    p.parse_args()
    raise NotImplementedError("S7 (Sajal)")


if __name__ == "__main__":
    main()
