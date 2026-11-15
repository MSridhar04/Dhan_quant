"""Unattended VRP run: python -m quant.research.run_vrp -> Telegram summary + full report (or a warning)."""
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from quant.run_step1 import tg_file, tg_text  # noqa: E402

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        from quant.research.vrp import run
        S, L = run()
        text = "\n".join(["🧪 dhan_quant · Research #1: volatility risk premium",
                          "Sell hedged ATM straddle before weekly expiry, hold to expiry",
                          "Chosen ≤2024 · judged 2025–Oct 2026 (unseen) · 1 lot, real costs", ""] + S)
        (ROOT / "reports" / "vrp_phone.txt").write_text(text, encoding="utf-8")
        print(text)
        tg_text(text[:3900])
        tg_file(ROOT / "reports" / "vrp.txt", "VRP research: full results")
    except Exception as e:                                   # noqa: BLE001
        traceback.print_exc()
        tg_text(f"⚠️ dhan_quant VRP research failed: {type(e).__name__}: {e}")
        raise
