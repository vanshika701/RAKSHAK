"""
run_training.py

Thin entry point for train_model.py's pipeline - run this file, not
train_model.py directly.

train_model.py defines custom classes (LabelDecodingClassifier,
SoftVotingEnsemble, ThresholdedEnsemble) that get saved via joblib.
Executing train_model.py directly as a script (`python src/train_model.py`)
would tag those classes as belonging to the module "__main__", which only
unpickles correctly from another direct run of that exact file - not from a
normal import (`from src.train_model import ...`), which is how the saved
models need to be loadable everywhere else (notebooks, detector.py, one-off
diagnostic scripts). Importing train_model here instead means its classes
get their one real, consistently-importable module path from the start.
"""

import sys
from pathlib import Path

# Running this file directly (`python src/run_training.py`) puts src/
# itself on sys.path, not the project root - "src" isn't importable as a
# package from inside src/. Adding the parent directory explicitly makes
# this work the same simple way as every other `python src/*.py` command,
# regardless of where it's invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train_model import main  # noqa: E402

if __name__ == "__main__":
    main()
