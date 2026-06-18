from src.pipeline import run_pipeline
import sys

try:
    run_pipeline()
except Exception as e:
    print(e, file=sys.stderr)
    sys.exit(1)
