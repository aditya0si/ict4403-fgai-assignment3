"""Create comparison plots from measured metrics, without hard-coded scores."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

root = Path("results")
names = ["scratch_transformer", "mt5_small", "tinyllama", "gemma_270m"]
rows = []
for name in names:
    path = root / f"{name}_metrics.json"
    if path.exists(): rows.append((name, json.loads(path.read_text(encoding="utf-8"))))
if not rows: raise SystemExit("No measured result files found")

labels = [name.replace("_", " ").title() for name, _ in rows]
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, metric, title, color in zip(
    axes,
    ["bleu", "chrf", "seconds_per_sentence"],
    ["Corpus BLEU", "Corpus chrF", "Inference seconds per sentence"],
    ["#3366cc", "#178b78", "#b66326"],
):
    values = [record[metric] for _, record in rows]
    bars = ax.bar(labels, values, color=color)
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=35)
    ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("English to French translation on the same 100 sentence test set")
fig.tight_layout()
fig.savefig(root / "comparison.png", dpi=180, bbox_inches="tight")
print(root / "comparison.png")
