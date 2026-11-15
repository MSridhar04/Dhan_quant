"""Unattended regime run: python -m quant.research.run_regime -> Telegram summary + full report (or a warning)."""
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from quant.run_step1 import tg_file, tg_text  # noqa: E402

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        from quant.research.regime import run
        S, L, best = run()
        text = "\n".join(["🧪 dhan_quant · Research #2: regime detection",
                          "8 paper strategies + weekly iron fly · skip a strategy in regimes it lost in EARLIER years",
                          "Walk-forward: learn on past years, test the next (2023, 2024, 2025, 2026)",
                          "Totals = all trades → with regime filter", ""] + S)
        (ROOT / "reports" / "regime_phone.txt").write_text(text, encoding="utf-8")
        print(text)
        tg_text(text[:3900])
        tg_file(ROOT / "reports" / "regime.txt", "Regime research: full results")
    except Exception as e:                                   # noqa: BLE001
        traceback.print_exc()
        tg_text(f"⚠️ dhan_quant regime research failed: {type(e).__name__}: {e}")
        raise
