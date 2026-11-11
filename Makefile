.PHONY: install dataset train-detector train-ppo train-slm dry-run full-run stats plots literature all clean

PYTHON := .venv/bin/python
SEED    := 42

install:
	/root/.local/bin/uv pip install \
		networkx pyyaml pydantic pytest psutil matplotlib \
		scikit-learn xgboost drain3 pandas scipy statsmodels \
		scikit-posthocs pingouin lifelines tabulate \
		gymnasium stable-baselines3 sentence-transformers \
		groq openai torch torchvision

dataset:
	$(PYTHON) data/gen_intent_dataset.py --total 8000 --seed $(SEED) --out-dir data

train-detector:
	$(PYTHON) train/train_detector.py --samples 2000 --seed $(SEED)

train-ppo:
	$(PYTHON) train/train_ppo.py --steps 50000 --seed $(SEED) --resume

train-slm:
	$(PYTHON) train/train_slm.py --mode eval --test-jsonl data/intent_dataset_test.jsonl \
		--seed $(SEED) --n-samples 50

dry-run:
	$(PYTHON) src/benchmark_runner.py \
		--topology T1_tree \
		--scenarios S1_link_failure S2_ddos \
		--configs B0 B2 FULL \
		--n 2 --seed $(SEED) --dry-run

full-run:
	$(PYTHON) src/benchmark_runner.py \
		--topology T1_tree T2_abilene T3_fattree \
		--scenarios S1_link_failure S2_ddos S3_misconfig S4_cascading S5_combined \
		--configs B0 B1 B2 B3 FULL A1 A2 A3 A4 A5 A6 \
		--n 30 --seed $(SEED)

stats:
	$(PYTHON) evaluation/stats.py \
		--input results/raw/runs.csv \
		--output results/tables/
	$(PYTHON) evaluation/failure_taxonomy.py \
		--audit-log results/logs/audit.jsonl \
		--output-dir results/tables/

literature:
	$(PYTHON) evaluation/compare_literature.py \
		--input results/raw/runs.csv \
		--output results/comparison_existing.csv \
		--output-dir results/tables/

plots:
	$(PYTHON) plots/make_plots.py \
		--input results/raw/runs.csv \
		--output results/figures/

test:
	$(PYTHON) -m pytest tests/ -v -k "not needs_root" --tb=short

test-root:
	$(PYTHON) -m pytest tests/ -v --tb=short

all: dataset train-detector train-ppo dry-run stats literature plots

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	mn --clean 2>/dev/null; true

