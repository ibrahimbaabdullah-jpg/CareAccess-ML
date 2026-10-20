"""Reproduce the frozen chronological benchmark. Run from the repository root."""

import argparse
import json

from careaccess.training import train


def main(raw):
    default, report = train(raw)
    summary = {"default_model": default, "metrics": report["metrics"], "seconds": report["training_seconds"]}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    main(parser.parse_args().data)
