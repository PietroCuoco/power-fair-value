.PHONY: install run test lint figures submission clean

## install dependencies into the active environment
install:
	pip install -r requirements.txt
	pip install -e .

## full reproducible pipeline: ingest -> QA -> features -> models -> trade -> llm -> submission -> figures
run:
	python scripts/run_pipeline.py --stage all

## run the test suite
test:
	pytest -q

## lint
lint:
	ruff check .

## regenerate just the figures
figures:
	python scripts/run_pipeline.py --stage figures

## regenerate just submission.csv
submission:
	python scripts/run_pipeline.py --stage submission

## remove generated artifacts (keeps raw data)
clean:
	rm -f data/processed/*.parquet reports/figures/*.png submission.csv
