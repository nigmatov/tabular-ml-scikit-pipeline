.PHONY: setup train eval serve
setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

train:
	python src/train.py --data data/sample.csv --target target

eval:
	python src/evaluate.py --ckpt models/best.joblib --data data/sample.csv --target target

serve:
	uvicorn src.serve:app --port 8001 --reload
