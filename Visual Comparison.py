import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

manual = pd.read_csv("Dashed Black Line Manual Extraction.csv")
x_m = manual.iloc[:, 0].values
y_m = manual.iloc[:, 1].values

data = pd.read_csv("Dashed Black Line GPT5.csv")
x = data["t_min"].values
y = data["G_mg_dl"].values

x_m = x_m.astype(int)
x = x.astype(int)

even_manual = (x_m % 2 == 0)
even_est = (x % 2 == 0)

x_m = x_m[even_manual]
y_m = y_m[even_manual]

x = x[even_est]
y = y[even_est]

idx_m = np.argsort(x_m)
x_m, y_m = x_m[idx_m], y_m[idx_m]

idx = np.argsort(x)
x, y = x[idx], y[idx]

spline = CubicSpline(x, y)
x_dense = np.linspace(0, 360, 2000)
y_est = spline(x_dense)

plt.figure(figsize=(8.5,5.5))

plt.plot(
    x_m, y_m,
    linestyle="--",
    linewidth=1,
    marker="o",
    markersize=1.5,
    color="black",
    alpha=0.6,
    label="Manual extraction"
)

plt.plot(
    x_dense, y_est,
    linewidth=1.5,
    color="blue",
    label="ChatGPT 5 interpolation"
)

plt.xlabel("t (min)")
plt.ylabel("G(t) (mg/dl)")
plt.xlim(0, 360)
plt.ylim(0, 600)
plt.grid(True, linestyle=":", alpha=0.7)
plt.legend(frameon=False)

plt.tight_layout()
plt.show()
