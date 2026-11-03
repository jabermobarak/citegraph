import argparse
from pathlib import Path

from .data import prepare
from .evaluate import train_release


def main():
    parser = argparse.ArgumentParser(prog="citegraph")
    parser.add_argument("command", choices=["prepare", "train", "verify"])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.data, args.archive))
    elif args.command == "train":
        train_release(args.data, args.artifacts, epochs=args.epochs)
    else:
        from .service import Workspace

        workspace = Workspace(args.data, args.artifacts)
        print({"ready": True, "nodes": len(workspace.graph.ids)})


if __name__ == "__main__":
    main()
