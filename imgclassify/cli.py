"""Command-line entry point.

Examples:
    python -m imgclassify classify --image data/images/dog.jpg
    python -m imgclassify classify --sample cat dog --top-k 3
    python -m imgclassify compare --model resnet18 resnet50 resnet101
    python -m imgclassify benchmark --n 10
    python -m imgclassify all
"""

from __future__ import annotations

import argparse
import json
import os

from .benchmark import benchmark_batch_vs_individual
from .compare import compare_models
from .data import get_sample_images
from .predict import MODEL_REGISTRY, classify_image, load_labels, load_model
from .visualize import plot_batch_vs_individual, plot_model_comparison, plot_topk_predictions

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def _resolve_images(args: argparse.Namespace) -> dict[str, str]:
    if args.image:
        return {os.path.splitext(os.path.basename(p))[0]: p for p in args.image}
    return get_sample_images(args.sample)


def cmd_classify(args: argparse.Namespace) -> None:
    images = _resolve_images(args)
    labels = load_labels()
    model = load_model(args.model)

    predictions = {}
    for name, path in images.items():
        preds = classify_image(path, model, labels, top_k=args.top_k)
        predictions[name] = preds
        print(f"\n[{name}]")
        for i, p in enumerate(preds, 1):
            print(f"  {i}. {p['class']:35s} {p['probability']:6.2f}%")

    if args.plot:
        out = plot_topk_predictions(images, predictions)
        print(f"\nFigure saved -> {out}")

    if args.save_json:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_json = os.path.join(RESULTS_DIR, "classify_results.json")
        with open(out_json, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"Results saved -> {out_json}")


def cmd_compare(args: argparse.Namespace) -> None:
    images = get_sample_images(args.sample)
    comparison = compare_models(images, args.model)

    for name, stats in comparison.items():
        print(f"\n{name}")
        print(f"  Parameters:    {stats['params_M']} M")
        print(f"  Avg inference: {stats['avg_inference_ms']} ms")
        print(f"  Published top-1 acc: {stats['published_top1_acc']}%")

    out = plot_model_comparison(comparison)
    print(f"\nFigure saved -> {out}")

    if args.save_json:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_json = os.path.join(RESULTS_DIR, "compare_results.json")
        with open(out_json, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"Results saved -> {out_json}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    images = get_sample_images()
    paths = (list(images.values()) * ((args.n // len(images)) + 1))[: args.n]

    result = benchmark_batch_vs_individual(paths, args.model)
    print(f"Individual - total: {result['individual_total_ms']} ms | "
          f"per image: {result['individual_per_img_ms']} ms")
    print(f"Batch      - total: {result['batch_total_ms']} ms | "
          f"per image: {result['batch_per_img_ms']} ms")
    print(f"Speedup: {result['speedup']}x")

    out = plot_batch_vs_individual(
        result["individual_times_ms"], result["batch_total_ms"], result["speedup"]
    )
    print(f"Figure saved -> {out}")

    if args.save_json:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_json = os.path.join(RESULTS_DIR, "benchmark_results.json")
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved -> {out_json}")


def cmd_all(args: argparse.Namespace) -> None:
    """Reproduce the full original lab pipeline: classify, compare, benchmark."""
    args.image = None
    args.sample = None
    args.top_k = 5
    args.model = "resnet50"
    args.plot = True
    args.save_json = True
    cmd_classify(args)

    args.model = list(MODEL_REGISTRY)
    cmd_compare(args)

    args.model = "resnet50"
    args.n = 10
    cmd_benchmark(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="imgclassify", description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_classify = sub.add_parser("classify", help="Top-k predictions for one or more images")
    p_classify.add_argument("--image", nargs="+", help="Path(s) to your own image(s)")
    p_classify.add_argument("--sample", nargs="+", help="Built-in sample names (default: all)")
    p_classify.add_argument("--model", default="resnet50", choices=list(MODEL_REGISTRY))
    p_classify.add_argument("--top-k", type=int, default=5)
    p_classify.add_argument("--plot", action="store_true", default=True)
    p_classify.add_argument("--save-json", action="store_true", default=True)
    p_classify.set_defaults(func=cmd_classify)

    p_compare = sub.add_parser("compare", help="Compare ResNet variants on size/speed/accuracy")
    p_compare.add_argument("--sample", nargs="+", help="Built-in sample names (default: all)")
    p_compare.add_argument("--model", nargs="+", default=list(MODEL_REGISTRY),
                            choices=list(MODEL_REGISTRY))
    p_compare.add_argument("--save-json", action="store_true", default=True)
    p_compare.set_defaults(func=cmd_compare)

    p_bench = sub.add_parser("benchmark", help="Batch vs. individual inference timing")
    p_bench.add_argument("--n", type=int, default=10, help="Total images to process")
    p_bench.add_argument("--model", default="resnet50", choices=list(MODEL_REGISTRY))
    p_bench.add_argument("--save-json", action="store_true", default=True)
    p_bench.set_defaults(func=cmd_benchmark)

    p_all = sub.add_parser("all", help="Run classify + compare + benchmark end to end")
    p_all.set_defaults(func=cmd_all)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
