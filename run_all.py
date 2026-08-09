import subprocess, sys, re, datetime
from pathlib import Path

MODELS = ["TabPFN.py", "SVR.py", "MLP.py", "XGBoost.py", "ExtraTrees.py", "RandomForest.py", "HGB.py"]

ANALYSIS = [
    "analysis/feature_dependence.py",          # correlations, VIF, trait clusters
    "analysis/model_comparison_seeded.py",     # model comparison + statistical tests
    "analysis/SHAP_analysis.py",               # SHAP values and plots
    "analysis/permutation_importance.py",      # permutation importance
    "analysis/loco_analysis.py",               # LOCO feature contribution
    "analysis/repeated_split_evaluation.py",   # ablation + repeated-split robustness
]

results = []

for script in MODELS:
    print(f"\n{'='*55}")
    print(f"  Running: {script}")
    print(f"{'='*55}")
    proc = subprocess.run(
        [sys.executable, Path(__file__).parent / script],
        capture_output=True, text=True
    )
    out = proc.stdout
    print(out)
    if proc.stderr:
        print("STDERR:", proc.stderr[-500:])

    cv   = re.search(r"CV R²\s*\(Mean.*?\)\s*:\s*([\d.]+)", out)
    test = re.search(r"Test R²\s*:\s*([\d.]+)", out)
    rmse = re.search(r"Test RMSE\s*:\s*([\d.]+)", out)

    results.append({
        "model": script.replace(".py", ""),
        "cv":   float(cv.group(1))   if cv   else None,
        "test": float(test.group(1)) if test else None,
        "rmse": float(rmse.group(1)) if rmse else None,
    })

results.sort(key=lambda x: x["test"] if x["test"] else 0, reverse=True)

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

print(f"\n{'='*55}")
print(f"  All models completed!")
print(f"  Date: {now}")
print(f"{'='*55}")
print(f"\n  SUMMARY TABLE")
print(f"  {'Model':<20} {'CV R²':>10} {'Test R²':>10} {'RMSE':>10}")
print("  " + "-" * 53)
medals = ["1.", "2.", "3."]
for i, r in enumerate(results):
    rank = medals[i] if i < 3 else f"{i+1}."
    cv   = f"{r['cv']:.4f}"   if r["cv"]   else "  —  "
    test = f"{r['test']:.4f}" if r["test"] else "  —  "
    rmse = f"{r['rmse']:.4f}" if r["rmse"] else "  —  "
    print(f"  {rank} {r['model']:<18} {cv:>10} {test:>10} {rmse:>10}")

# Run analysis scripts
print(f"\n{'='*55}")
print("  Running analysis scripts ...")
print(f"{'='*55}")
for script in ANALYSIS:
    print(f"\n  > {script}")
    proc = subprocess.run(
        [sys.executable, Path(__file__).parent / script],
        capture_output=True, text=True
    )
    print(proc.stdout)
    if proc.stderr:
        print("STDERR:", proc.stderr[-500:])

print(f"\n{'='*55}")
print("  All analyses completed!")
print(f"{'='*55}")
