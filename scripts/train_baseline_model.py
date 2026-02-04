from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ml.train_baseline_model import (  # noqa: E402
    load_env,
    report_training_data_state,
    main,
)

__all__ = ["load_env", "report_training_data_state", "main"]


if __name__ == "__main__":
    main()

