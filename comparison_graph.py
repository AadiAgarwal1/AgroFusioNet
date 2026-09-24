import os
import matplotlib.pyplot as plt

# -----------------------------
# Helper function to add labels
# -----------------------------
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f"{height:.3f}",
            ha='center',
            va='bottom'
        )

# -----------------------------
# Create figures folder
# -----------------------------
os.makedirs("figures", exist_ok=True)

# -----------------------------
# YIELD (Regression Metrics)
# -----------------------------
models = ["Linear", "RF", "XGB", "AgroFusionNet"]

rmse = [0.716456849, 0.51590737, 0.442134224, 0.316797285]
r2 = [0.672807487, 0.830344969, 0.875396125, 0.936028552]

# RMSE Plot
plt.figure()
bars = plt.bar(models, rmse)
plt.title("RMSE Comparison (Yield Prediction)")
plt.xlabel("Models")
plt.ylabel("RMSE")
plt.xticks(rotation=30)
add_labels(bars)
plt.tight_layout()
plt.savefig("figures/rmse_comparison.png")
plt.close()

# R2 Plot
plt.figure()
bars = plt.bar(models, r2)
plt.title("R² Comparison (Yield Prediction)")
plt.xlabel("Models")
plt.ylabel("R² Score")
plt.xticks(rotation=30)
add_labels(bars)
plt.tight_layout()
plt.savefig("figures/r2_comparison.png")
plt.close()

# -----------------------------
# STRESS (Classification Metrics)
# -----------------------------
models_stress = ["Logistic", "RF", "XGB", "AgroFusionNet"]

accuracy = [0.816647926, 0.8269438, 0.849397943, 0.876174245]
f1 = [0.823422833, 0.835773547, 0.856720125, 0.881575077]

# Accuracy Plot
plt.figure()
bars = plt.bar(models_stress, accuracy)
plt.title("Accuracy Comparison (Stress Classification)")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
add_labels(bars)
plt.tight_layout()
plt.savefig("figures/accuracy_comparison.png")
plt.close()

# F1 Plot
plt.figure()
bars = plt.bar(models_stress, f1)
plt.title("F1 Score Comparison (Stress Classification)")
plt.ylabel("F1 Score")
plt.xticks(rotation=30)
add_labels(bars)
plt.tight_layout()
plt.savefig("figures/f1_comparison.png")
plt.close()

print("✅ Graphs with values saved in 'figures/' folder!")