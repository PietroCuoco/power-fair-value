"""End-to-end pipeline entrypoint. Stages are wired in over Days 1-5."""
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Power fair-value pipeline")
    parser.add_argument(
        "--stage",
        choices=["ingest", "qa", "features", "model", "trade", "all"],
        default="all",
    )
    args = parser.parse_args()
    print(f"[run_pipeline] stage={args.stage} (not yet wired)")


if __name__ == "__main__":
    main()