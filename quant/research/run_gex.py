"""Unattended gamma run: python -m quant.research.run_gex -> Telegram summary + full report (or a warning)."""
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from quant.run_step1 import tg_file, tg_text  # noqa: E402

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        from quant.research.gex import run
        S, L = run()
        text = "\n".join(["🧪 dhan_quant · Research #5: gamma exposure (dealer positioning)",
                          "From option open interest + IV · pick ≤2024 · check 2025–26", ""] + S)
        (ROOT / "reports" / "gex_phone.txt").write_text(text, encoding="utf-8")
        print(text)
        tg_text(text[:3900])
        tg_file(ROOT / "reports" / "gex.txt", "Gamma exposure research: full results")
    except Exception as e:                                   # noqa: BLE001
        traceback.print_exc()
        tg_text(f"⚠️ dhan_quant gamma exposure research failed: {type(e).__name__}: {e}")
        raise
