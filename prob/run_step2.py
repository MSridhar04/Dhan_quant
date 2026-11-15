"""Step 2 rerun (2026-11-15): same tests as Step 1 + quiet-day, now with India VIX and synthetic-futures features.
Sends a before/after comparison to Telegram.   python -m prob.run_step2
"""
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"
sys.path.insert(0, str(ROOT))
from prob.run_step1 import tg_file, tg_text  # noqa: E402


def keep_old(name: str) -> list[str]:
    src = REP / name
    dst = REP / name.replace(".txt", "_v1.txt")
    if src.exists() and not dst.exists():
        shutil.copy(src, dst)
    return dst.read_text(encoding="utf-8").splitlines() if dst.exists() else []


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        old1 = keep_old("step1_phone.txt")[3:]
        keep_old("step1.txt"); keep_old("quietday.txt")
        old_q = keep_old("quietday_phone.txt")[3:]
        from prob import quietday, step1
        S1, L1 = step1.run()
        shutil.copy(REP / "step1.txt", REP / "step2_tests.txt")
        S2, L2 = quietday.run()
        shutil.copy(REP / "quietday.txt", REP / "step2_quietday.txt")
        out = ["🧪 dhan_prob Step 2 · with India VIX + synthetic futures", "Train ≤2025 · test 2026 · before → after", ""]
        for new, old in zip(S1, old1):
            if new.strip() and new != old and not new.startswith(("📈", "🧰")):
                out += [f"before: {old}", f"after:  {new}"]
            else:
                out.append(new)
        out += ["", "Quiet-day iron fly (after):"] + S2
        text = "\n".join(out)
        (REP / "step2_phone.txt").write_text(text, encoding="utf-8")
        print(text)
        tg_text(text[:3900])
        for f, cap in (("step2_tests.txt", "Step 2: full test results"), ("step2_quietday.txt", "Step 2: quiet-day details")):
            tg_file(REP / f, cap)
    except Exception as e:                                   # noqa: BLE001
        traceback.print_exc()
        tg_text(f"⚠️ dhan_prob Step 2 failed: {type(e).__name__}: {e}")
        raise
