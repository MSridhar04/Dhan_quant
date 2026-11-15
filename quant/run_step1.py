"""Unattended Step 1 run: python -m quant.run_step1 -> Telegram (summary + full report), or a warning if it fails."""
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALGO = ROOT.parent / "dhan_algo"
APY = ALGO / ".venv" / "Scripts" / "python.exe"


def tg_text(text: str) -> None:
    code = ("import sys; sys.path.insert(0,'scripts'); from weekly_summary import send; "
            "send(open(sys.argv[1], encoding='utf-8').read())")
    tmp = ROOT / "reports" / "_tg.txt"
    tmp.write_text(text, encoding="utf-8")
    subprocess.run([str(APY), "-c", code, str(tmp)], cwd=ALGO, stdin=subprocess.DEVNULL)


def tg_file(path: Path, caption: str) -> None:
    subprocess.run([str(APY), "scripts/send_telegram_files.py", "--me", str(path), "--caption", caption], cwd=ALGO,
                   stdin=subprocess.DEVNULL)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        from quant.step1 import run
        S, L = run()
        head = ["🧪 dhan_quant Step 1 · test 2026 (trained ≤2025)", "AUC: 0.50 = coin toss, 0.60+ = useful", ""]
        text = "\n".join(head + S)
        (ROOT / "reports" / "step1_phone.txt").write_text(text, encoding="utf-8")
        print(text)
        tg_text(text)
        tg_file(ROOT / "reports" / "step1.txt", "dhan_quant Step 1: full results")
    except Exception as e:                                   # noqa: BLE001
        traceback.print_exc()
        tg_text(f"⚠️ dhan_quant Step 1 failed: {type(e).__name__}: {e}")
        raise
