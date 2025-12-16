import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline

manual = pd.read_csv("Dashed Black Line Manual Extraction.csv")
est = pd.read_csv("Dashed Black Line GPT5.csv")

manual.columns = ["t_min", "G_manual"]

t_est = est["t_min"].values
G_est = est["G_mg_dl"].values

idx = np.argsort(t_est)
t_est = t_est[idx]
G_est = G_est[idx]

spline = CubicSpline(t_est, G_est)

t_manual = manual["t_min"].values
G_manual = manual["G_manual"].values

G_est_interp = spline(t_manual)

mse = np.mean((G_manual - G_est_interp) ** 2)

print(f"Mean Squared Error (MSE): {mse:.3f}")

print(f"Number of manual points used: {len(G_manual)}")
