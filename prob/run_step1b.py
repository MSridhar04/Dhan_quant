"""Unattended Step 1b run: python -m prob.run_step1b -> Telegram (summary + full report), or a warning if it fails."""
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
        from prob.step1b import run
        S, L = run()
        head = ["🧪 dhan_prob Step 1b · big-move model uses · test 2026", "Settings picked on 2025, tested on 2026", ""]
        text = "\n".join(head + S)
        (ROOT / "reports" / "step1b_phone.txt").write_text(text, encoding="utf-8")
        print(text)
        tg_text(text)
        tg_file(ROOT / "reports" / "step1b.txt", "dhan_prob Step 1b: full results")
    except Exception as e:                                   # noqa: BLE001
        traceback.print_exc()
        tg_text(f"⚠️ dhan_prob Step 1b failed: {type(e).__name__}: {e}")
        raise
