# -*- coding: utf-8 -*-
"""Command-line entry point: `dualct {train,cv,predict,eval,predict-dir}`.

Override precedence (highest to lowest):
dedicated CLI flags > --set key.path=value > YAML file > dataclass defaults.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from .config import Config, load_config


def _build_common_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=str, default="configs/iodine.yaml", help="Path to a YAML config file")
    p.add_argument(
        "--set",
        dest="set_overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted-key overrides, e.g. --set loss.region=full train.num_epochs=10",
    )
    p.add_argument("--fold", type=int, default=None, help="Override cv.fold (0..n_folds-1)")
    p.add_argument("--epochs", type=int, default=None, help="Override train.num_epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Override train.batch_size")
    p.add_argument("--loss-region", type=str, default=None, choices=["mask", "full"], help="Override loss.region")
    p.add_argument("--loss-aux", type=str, default=None, choices=["none", "cdf"], help="Override loss.aux")
    p.add_argument(
        "--residual", dest="residual", action="store_true", default=None, help="Enable residual connection (model.residual=true)"
    )
    p.add_argument(
        "--no-residual", dest="residual", action="store_false", help="Disable residual connection (model.residual=false)"
    )
    p.add_argument("--data-root", type=str, default=None, help="Override data.root")
    p.add_argument("--ckpt", type=str, default=None, help="Path to a checkpoint (.safetensors or .pth); required by predict/eval/predict-dir")
    p.add_argument("--input-dir", type=str, default=None, help="[predict-dir] input folder (a series, or a parent of per-patient folders)")
    p.add_argument("--out-dir", type=str, default=None, help="[predict-dir] output folder for the synthesized DICOM series")
    p.add_argument("--mask-dir", type=str, default=None, help="[predict-dir] optional external mask folder (iodine task)")
    return p


def _dedicated_overrides(args: argparse.Namespace) -> List[str]:
    overrides = []
    if args.fold is not None:
        overrides.append(f"cv.fold={args.fold}")
    if args.epochs is not None:
        overrides.append(f"train.num_epochs={args.epochs}")
    if args.batch_size is not None:
        overrides.append(f"train.batch_size={args.batch_size}")
    if args.loss_region is not None:
        overrides.append(f"loss.region={args.loss_region}")
    if args.loss_aux is not None:
        overrides.append(f"loss.aux={args.loss_aux}")
    if args.residual is not None:
        overrides.append(f"model.residual={'true' if args.residual else 'false'}")
    if args.data_root is not None:
        overrides.append(f"data.root={args.data_root}")
    return overrides


def _resolve_config(args: argparse.Namespace) -> Config:
    # --set entries are applied first, dedicated flags second, so dedicated
    # flags win when both target the same key (dedicated flags = highest precedence).
    all_overrides = list(args.set_overrides) + _dedicated_overrides(args)
    return load_config(args.config, all_overrides)


def _save_resolved_config(cfg: Config, command: str) -> Path:
    out_dir = Path(cfg.output.metrics_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"resolved_config_{command}_fold{cfg.cv.fold}.yaml"
    cfg.to_yaml(out_path)
    print(f"[cli] resolved config saved to {out_path}")
    return out_path


def cmd_train(args: argparse.Namespace) -> None:
    from .engine import train_one_fold

    cfg = _resolve_config(args)
    _save_resolved_config(cfg, "train")
    metrics = train_one_fold(cfg, cfg.cv.fold)
    print("[cli train] final metrics:", metrics)


def cmd_cv(args: argparse.Namespace) -> None:
    from .engine import run_cv

    cfg = _resolve_config(args)
    _save_resolved_config(cfg, "cv")
    run_cv(cfg)


def cmd_predict(args: argparse.Namespace) -> None:
    from .inference import predict_fold

    cfg = _resolve_config(args)
    _save_resolved_config(cfg, "predict")
    if not args.ckpt:
        raise SystemExit("`dualct predict` requires --ckpt <path-to-checkpoint>")
    metrics = predict_fold(cfg, cfg.cv.fold, args.ckpt, save_dicom=True)
    print("[cli predict] metrics:", metrics)


def cmd_eval(args: argparse.Namespace) -> None:
    from .inference import eval_fold

    cfg = _resolve_config(args)
    _save_resolved_config(cfg, "eval")
    if not args.ckpt:
        raise SystemExit("`dualct eval` requires --ckpt <path-to-checkpoint>")
    metrics = eval_fold(cfg, cfg.cv.fold, args.ckpt)
    print("[cli eval] metrics:", metrics)


def cmd_predict_dir(args: argparse.Namespace) -> None:
    from .inference import predict_directory

    cfg = _resolve_config(args)
    _save_resolved_config(cfg, "predict_dir")
    if not args.ckpt:
        raise SystemExit("`dualct predict-dir` requires --ckpt <path-to-checkpoint>")
    if not args.input_dir:
        raise SystemExit("`dualct predict-dir` requires --input-dir <folder>")
    out_dir = args.out_dir or str(Path(cfg.output.pred_dir) / "standalone")
    saved = predict_directory(cfg, args.input_dir, args.ckpt, out_dir, mask_dir=args.mask_dir)
    print(f"[cli predict-dir] saved {len(saved)} series under {out_dir}")


def main(argv: Optional[List[str]] = None) -> None:
    common = _build_common_parser()
    parser = argparse.ArgumentParser(
        prog="dualct",
        description="120 kVp dual-energy CT -> Iodine (Lung PBV) map or kVp-level (80/140 kVp) translation pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", parents=[common], help="Train a single CV fold")
    p_train.set_defaults(func=cmd_train)

    p_cv = sub.add_parser("cv", parents=[common], help="Run all folds of cross-validation and summarize")
    p_cv.set_defaults(func=cmd_cv)

    p_predict = sub.add_parser("predict", parents=[common], help="Predict + save DICOM series for a fold")
    p_predict.set_defaults(func=cmd_predict)

    p_predict_dir = sub.add_parser(
        "predict-dir", parents=[common], help="Standalone inference on an arbitrary input folder"
    )
    p_predict_dir.set_defaults(func=cmd_predict_dir)

    p_eval = sub.add_parser("eval", parents=[common], help="Evaluate a checkpoint on a fold's held-out test set")
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
