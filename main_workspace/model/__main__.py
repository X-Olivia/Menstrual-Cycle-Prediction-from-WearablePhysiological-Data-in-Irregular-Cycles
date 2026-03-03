"""Run experiment entry when executing python -m model; avoids run_experiment being preloaded by package __init__ and triggering warnings."""
from .run_experiment import run_experiment

if __name__ == "__main__":
    result = run_experiment()
    print("Val:", result["val"])
    print("Test:", result["test"])
