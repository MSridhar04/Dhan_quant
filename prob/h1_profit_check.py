import sys
sys.path.insert(0, r"C:\Users\Sridhar M\Desktop\Main_projects\dhan_prob")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from prob import model as M
df, feats, tr, v, te, p, p_val, mdl = M.fit("SENSEX", 1)
y = te.y1.to_numpy().astype(int)
for hold in (1, 2, 3):
    for thr in (0.55, 0.6, 0.65):
        r = M.profit_check("SENSEX", te, p, thr, hold=hold)
        print(f"hold {hold}m >= {int(thr*100)}%: {r['trades']} trades, gross {r['gross']:,.0f}, net {r['net']:,.0f}, win {r['win_%']:.0f}%")
imp = pd.Series(mdl.feature_importance("gain"), index=feats).sort_values(ascending=False)
print("top:", ", ".join(imp.index[:8]))
m = p.max(1) >= 0.6
print("strong 1-min readings: median |fwd1| bp", round(te.fwd1[m].abs().median(), 2), "vs all", round(te.fwd1.abs().median(), 2))
