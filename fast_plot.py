import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

for model in ["HH", "HR", "IZH"]:
    files = sorted(glob.glob(f"{model}_dt_*.csv"))

    plt.figure(figsize=(10, 5))

    for f in files:
        df = pd.read_csv(f)
        dt = os.path.basename(f).replace(f"{model}_dt_", "").replace(".csv", "")
        dt = dt.replace("_", ".")

        plt.plot(df["time_ms"], df["value"], label=f"dt={dt} ms")

    plt.title(model)
    plt.xlabel("Time (ms)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
