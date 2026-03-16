import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

_BASE = os.path.join(os.path.dirname(__file__), "..", "..")
_DATA_RAW = os.path.join(_BASE, "data", "raw")
_DATA_OUTPUT = os.path.join(_BASE, "data", "output")
_FIGURES = os.path.join(_BASE, "outputs", "figures")

actual_df = pd.read_csv(os.path.join(_DATA_RAW, "stock_2330.TW_2026-03-04_to_2026-03-07.csv"), parse_dates=["Date"])
pred_df = pd.read_csv(os.path.join(_DATA_OUTPUT, "predicted_prices.csv"), parse_dates=["Date"])

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(actual_df["Date"], actual_df["Adj Close"],
        marker="o", color="steelblue", linewidth=2, label="Actual Adj Close")

for _, row in actual_df.iterrows():
    ax.annotate(f'{row["Adj Close"]:.0f}',
                xy=(row["Date"], row["Adj Close"]),
                xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=9, color="steelblue")

ax.plot(pred_df["Date"], pred_df["Predicted_Price"],
        marker="s", linestyle="--", color="tomato", linewidth=2, label="Predicted Price")

for _, row in pred_df.iterrows():
    ax.annotate(f'{row["Predicted_Price"]:.0f}',
                xy=(row["Date"], row["Predicted_Price"]),
                xytext=(0, -16), textcoords="offset points",
                ha="center", fontsize=9, color="tomato")

ax.set_title("TSMC (2330.TW) — Actual vs Predicted Price", fontsize=13)
ax.set_xlabel("Date")
ax.set_ylabel("Price (TWD)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
ax.xaxis.set_major_locator(mdates.DayLocator())
plt.xticks(rotation=45)
ax.set_ylim(bottom=1100)
ax.yaxis.set_major_locator(plt.MultipleLocator(100))
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(_FIGURES, "price_comparison.png"), dpi=150)
plt.show()
print("Chart saved to price_comparison.png")
