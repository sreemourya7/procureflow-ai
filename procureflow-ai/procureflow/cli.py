from __future__ import annotations
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="procureflow")
    parser.add_argument(
        "--root", type=Path, default=Path(os.getenv("PROCUREFLOW_ROOT", Path.cwd()))
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("generate")
    sub.add_parser("train")
    sub.add_parser("evaluate")
    live = sub.add_parser("evaluate-live")
    live.add_argument("--cases", type=int, default=14)
    demo = sub.add_parser("demo")
    demo.add_argument("--mode", choices=["offline", "live"], default="offline")
    demo.add_argument("--key", default="demo-v1")
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "generate":
        from .synthetic import write_data

        write_data(root)
        print("Generated seeded synthetic datasets and demo submission.")
    elif args.command == "train":
        from .ml import train

        result = train(root)
        print(json.dumps(result, indent=2))
    elif args.command == "evaluate":
        from .evaluation import evaluate

        print(json.dumps(evaluate(root), indent=2))
    elif args.command == "evaluate-live":
        from .live_evaluation import evaluate_live

        print(json.dumps(evaluate_live(root, args.cases), indent=2))
    elif args.command == "demo":
        from .engine import Engine

        engine = Engine(root, mode=args.mode)
        submission = json.loads((root / "data/examples/submission.json").read_text())
        cid, _ = engine.create(submission, args.key)
        print(json.dumps(engine.run(cid), indent=2))
    elif args.command == "serve":
        import uvicorn
        from .api import create_app

        uvicorn.run(create_app(root), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
