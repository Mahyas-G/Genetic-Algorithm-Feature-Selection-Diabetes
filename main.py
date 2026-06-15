import pandas as pd
import numpy as np
import random
import warnings
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)

warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)

df = pd.read_csv("early_stage_diabetes.csv")
df.columns = df.columns.str.strip()

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

X = df.drop("Class", axis=1).values
y = df["Class"].values
FEATURE_NAMES = df.drop("Class", axis=1).columns.tolist()
N = X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Dataset: {df.shape}  |  Features: {N}")
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")
print(f"Features: {FEATURE_NAMES}\n")

POP_SIZE            = 30
GENERATIONS         = 40
ELITE_SIZE          = 2
MUTATION_RATE       = 0.05
CX_PROB             = 0.85
LAMBDA_PENALTY      = 0.05
STAGNATION_LIMIT    = 8
DIVERSITY_TOLERANCE = 0.05
STABILITY_RUNS      = 20
EVAL_SEEDS          = [42, 52, 62, 72, 82]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def eval_fitness(chrom: np.ndarray, cache: dict) -> float:
    key = tuple(chrom)
    if key in cache:
        return cache[key]

    num_selected = np.sum(chrom)
    if num_selected == 0:
        cache[key] = 0.0
        return 0.0

    idx = np.where(chrom == 1)[0]
    
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=42,
        ))
    ])

    base_score = cross_val_score(
        model, X_train[:, idx], y_train,
        cv=cv, scoring="accuracy"
    ).mean()

    penalty = LAMBDA_PENALTY * (num_selected / N)
    final_score = base_score - penalty

    cache[key] = float(final_score)
    return float(final_score)

def init_pop(n_pop: int, n_genes: int) -> np.ndarray:
    pop = []
    while len(pop) < n_pop:
        c = np.random.randint(0, 2, n_genes)
        if c.sum() >= 1:
            pop.append(c)
    return np.array(pop)

def sel_tournament(pop: np.ndarray, fit: np.ndarray, k: int = 3) -> np.ndarray:
    idx = np.random.choice(len(pop), k, replace=False)
    return pop[idx[np.argmax(fit[idx])]].copy()

def sel_rank(pop: np.ndarray, fit: np.ndarray) -> np.ndarray:
    ranks  = np.argsort(np.argsort(fit)) + 1
    probs  = (ranks ** 2) / (ranks ** 2).sum()
    return pop[np.random.choice(len(pop), p=probs)].copy()

def sel_roulette(pop: np.ndarray, fit: np.ndarray) -> np.ndarray:
    f = np.array(fit, dtype=float)
    f -= f.min()
    s  = f.sum()
    p  = f / s if s > 0 else np.ones(len(f)) / len(f)
    return pop[np.random.choice(len(pop), p=p)].copy()

SELECTION_METHODS = {
    "Tournament":    sel_tournament,
    "Rank-based":    sel_rank,
    "Roulette Wheel": sel_roulette,
}

def cx_two_point(p1: np.ndarray, p2: np.ndarray) -> tuple:
    c1, c2 = p1.copy(), p2.copy()
    a, b   = sorted(np.random.choice(N, 2, replace=False))
    c1[a:b], c2[a:b] = p2[a:b].copy(), p1[a:b].copy()
    return c1, c2

def cx_uniform(p1: np.ndarray, p2: np.ndarray) -> tuple:
    mask = np.random.rand(N) < 0.5
    c1   = np.where(mask, p2, p1)
    c2   = np.where(mask, p1, p2)
    return c1.copy(), c2.copy()

CROSSOVER_METHODS = {
    "Two-point": cx_two_point,
    "Uniform":   cx_uniform,
}

def _ensure_valid(c: np.ndarray) -> np.ndarray:
    if c.sum() == 0:
        c[np.random.randint(N)] = 1
    return c

def mut_bitflip(c: np.ndarray) -> np.ndarray:
    m = c.copy()
    flip = np.random.rand(N) < MUTATION_RATE
    m[flip] = 1 - m[flip]
    return _ensure_valid(m)

def mut_random_reset(c: np.ndarray) -> np.ndarray:
    m    = c.copy()
    n_reset = np.random.randint(1, max(2, N // 4))
    idx  = np.random.choice(N, n_reset, replace=False)
    m[idx] = np.random.randint(0, 2, n_reset)
    return _ensure_valid(m)

MUTATION_METHODS = {
    "Bit-flip":      mut_bitflip,
    "Random-reset":  mut_random_reset,
}

def run_ga(
    selection: str = "Tournament",
    crossover: str = "Two-point",
    mutation:  str = "Bit-flip",
    pop_size:  int = POP_SIZE,
    n_gen:     int = GENERATIONS,
    seed:      int = 42,
    verbose:   bool = True,
) -> dict:
    np.random.seed(seed)
    random.seed(seed)

    _sel = SELECTION_METHODS[selection]
    _cx  = CROSSOVER_METHODS[crossover]
    _mut = MUTATION_METHODS[mutation]

    pop = init_pop(pop_size, N)
    g_best_chrom = None
    g_best_fit   = -1.0
    h_best, h_mean = [], []
    
    local_cache = {}
    stagnation_counter = 0

    if verbose:
        print(f"\n{'━'*70}")
        print(f"  sel={selection} | cx={crossover} | mut={mutation}")
        print(f"  pop={pop_size}, gen={n_gen}")
        print(f"{'━'*70}")
        print(f"{'Gen':>4} │ {'Best Fit':>9} │ {'Mean Fit':>9} │ {'#Feat':>6}")
        print("─"*70)

    t0 = time.time()

    for gen in range(n_gen):
        fit = np.array([eval_fitness(ind, local_cache) for ind in pop])

        bi = int(np.argmax(fit))
        if fit[bi] > g_best_fit:
            g_best_fit   = fit[bi]
            g_best_chrom = pop[bi].copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        h_best.append(g_best_fit)
        h_mean.append(float(fit.mean()))

        if verbose and (gen % 5 == 0 or gen == n_gen - 1 or stagnation_counter >= STAGNATION_LIMIT):
            print(f"{gen+1:>4} │ {g_best_fit:9.4f} │ {fit.mean():9.4f} │ {int(pop[bi].sum()):>6}")

        if stagnation_counter >= STAGNATION_LIMIT:
            if verbose:
                print(f"  [!] Early stopping triggered at generation {gen+1} (Stagnation)")
            break

        elite_idx = np.argsort(fit)[::-1][:ELITE_SIZE]
        new_pop   = [pop[i].copy() for i in elite_idx]

        while len(new_pop) < pop_size:
            p1 = _sel(pop, fit) if selection != "Tournament" else _sel(pop, fit, k=3)
            p2 = _sel(pop, fit) if selection != "Tournament" else _sel(pop, fit, k=3)

            if np.random.rand() < CX_PROB:
                c1, c2 = _cx(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            new_pop.append(_mut(c1))
            if len(new_pop) < pop_size:
                new_pop.append(_mut(c2))

        pop = np.array(new_pop)
        
        div = np.mean(np.std(pop, axis=0))
        if div < DIVERSITY_TOLERANCE:
            worst_idx = np.argsort(fit)[:3]
            for w in worst_idx:
                pop[w] = init_pop(1, N)[0]

    elapsed = time.time() - t0
    sel_features = [FEATURE_NAMES[i] for i, g in enumerate(g_best_chrom) if g]

    if verbose:
        print(f"\n  Done in {elapsed:.1f}s  |  Best Penalized Fit = {g_best_fit:.4f}")
        print(f"  Selected ({len(sel_features)}/{N}): {sel_features}")

    return dict(
        best_chrom = g_best_chrom,
        best_fit   = g_best_fit,
        selected   = sel_features,
        h_best     = h_best,
        h_mean     = h_mean,
        elapsed    = elapsed,
        selection  = selection,
        crossover  = crossover,
        mutation   = mutation,
    )

COMP_GEN = 15

print("="*70 + "\n  EXPERIMENT A — Default GA (Penalized Analysis)\n" + "="*70)
R_main = run_ga(
    selection="Tournament", crossover="Two-point", mutation="Bit-flip",
    pop_size=POP_SIZE, n_gen=GENERATIONS, seed=42, verbose=True,
)

print("\n" + "="*70 + "\n  EXPERIMENT B — Selection Methods\n" + "="*70)
R_sel = {}
for name in SELECTION_METHODS:
    R_sel[name] = run_ga(
        selection=name, crossover="Two-point", mutation="Bit-flip",
        pop_size=POP_SIZE, n_gen=COMP_GEN, seed=42, verbose=False,
    )
    r = R_sel[name]
    print(f"  {name:20s}  Fit={r['best_fit']:.4f}  ({len(r['selected'])} feat): {r['selected']}")

print("\n" + "="*70 + "\n  EXPERIMENT C — Crossover Methods\n" + "="*70)
R_cx = {}
for name in CROSSOVER_METHODS:
    R_cx[name] = run_ga(
        selection="Tournament", crossover=name, mutation="Bit-flip",
        pop_size=POP_SIZE, n_gen=COMP_GEN, seed=42, verbose=False,
    )
    r = R_cx[name]
    print(f"  {name:20s}  Fit={r['best_fit']:.4f}  ({len(r['selected'])} feat): {r['selected']}")

print("\n" + "="*70 + "\n  EXPERIMENT D — Mutation Methods\n" + "="*70)
R_mut = {}
for name in MUTATION_METHODS:
    R_mut[name] = run_ga(
        selection="Tournament", crossover="Two-point", mutation=name,
        pop_size=POP_SIZE, n_gen=COMP_GEN, seed=42, verbose=False,
    )
    r = R_mut[name]
    print(f"  {name:20s}  Fit={r['best_fit']:.4f}  ({len(r['selected'])} feat): {r['selected']}")

print("\n" + "="*70 + f"\n  EXPERIMENT E — Stability Selection ({STABILITY_RUNS} Independent Runs)\n" + "="*70)
sel_freq = np.zeros(N)
for s_run in range(STABILITY_RUNS):
    r_stab = run_ga(
        selection="Tournament", crossover="Two-point", mutation="Bit-flip",
        pop_size=POP_SIZE, n_gen=GENERATIONS, seed=100+s_run, verbose=False,
    )
    sel_freq += r_stab["best_chrom"]
sel_freq_pct = sel_freq / STABILITY_RUNS * 100

stable_idx = np.where(sel_freq_pct >= 50.0)[0]
if len(stable_idx) == 0:
    stable_idx = np.where(R_main["best_chrom"] == 1)[0]
stable_names = [FEATURE_NAMES[i] for i in stable_idx]

print(f"\n{'='*70}")
print(f"  FINAL STABILITY SELECTION: {len(stable_names)}/{N} features selected")
print(f"  {stable_names}")
print(f"{'='*70}")

def train_and_eval_robust(feat_idx, tag):
    accs, aucs = [], []
    cm_best = None
    X_sub = X[:, feat_idx] if len(feat_idx) > 0 else X
    
    for s in EVAL_SEEDS:
        Xt, Xv, yt, yv = train_test_split(X_sub, y, test_size=0.2, stratify=y, random_state=s)
        sc = StandardScaler()
        Xt = sc.fit_transform(Xt)
        Xv = sc.transform(Xv)
        
        m = MLPClassifier(
            hidden_layer_sizes=(16, 8), activation="relu", solver="adam",
            max_iter=500, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=15, random_state=s,
        )
        m.fit(Xt, yt)
        p = m.predict(Xv)
        pr = m.predict_proba(Xv)[:, 1]
        
        accs.append(accuracy_score(yv, p))
        aucs.append(roc_auc_score(yv, pr))
        if s == 42:
            cm_best = confusion_matrix(yv, p)
            
    m_acc, s_acc = np.mean(accs), np.std(accs)
    m_auc, s_auc = np.mean(aucs), np.std(aucs)
    
    print(f"\n  [{tag}]")
    print(f"  Test Acc = {m_acc:.4f} ± {s_acc:.4f}  (Mean over {len(EVAL_SEEDS)} splits)")
    print(f"  Test AUC = {m_auc:.4f} ± {s_auc:.4f}  (Mean over {len(EVAL_SEEDS)} splits)")
    return m_acc, m_auc, cm_best

print("\nBaseline — All Features:")
acc_base, auc_base, cm_base = train_and_eval_robust(np.arange(N), f"Baseline ({N} features)")

print("\nGA Stability-selected Features:")
acc_ga, auc_ga, cm_ga = train_and_eval_robust(stable_idx, f"GA Stable ({len(stable_names)} features)")

C_GRID = "#cccccc"
C_BEST = "#d62728"
C_MEAN = "#1f77b4"
P_SEL  = ["#1f77b4", "#ff7f0e", "#2ca02c"]
P_CX   = ["#9467bd", "#8c564b"]
P_MUT  = ["#17becf", "#bcbd22"]

def style_ax(ax):
    ax.grid(True, color=C_GRID, linewidth=0.7, zorder=0)
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
        sp.set_color("#aaaaaa")

def draw_cm(ax, cm, title):
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Neg", "Pos"], fontsize=9)
    ax.set_yticklabels(["Neg", "Pos"], fontsize=9)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    ax.set_title(title, fontsize=10)
    thresh = cm.max() / 2.0
    for r_ in range(2):
        for c_ in range(2):
            ax.text(c_, r_, str(cm[r_, c_]),
                    ha="center", va="center", fontsize=16, fontweight="bold",
                    color="white" if cm[r_, c_] > thresh else "#333333")

fig1 = plt.figure(figsize=(18, 14))
gs1  = gridspec.GridSpec(3, 2, figure=fig1, hspace=0.50, wspace=0.32, height_ratios=[1.2, 1, 1])

ax0 = fig1.add_subplot(gs1[0, :])
gens_main = list(range(1, len(R_main["h_best"]) + 1))
ax0.plot(gens_main, R_main["h_best"], color=C_BEST, linewidth=2.0, marker="o", markersize=4, label="Best Fitness", zorder=3)
ax0.plot(gens_main, R_main["h_mean"], color=C_MEAN, linewidth=1.5, linestyle="--", marker="s", markersize=3, label="Mean Fitness", zorder=3)
ax0.fill_between(gens_main, R_main["h_mean"], R_main["h_best"], color=C_BEST, alpha=0.07, zorder=2)
ax0.axhline(R_main["best_fit"], color=C_BEST, linewidth=0.8, linestyle=":", alpha=0.5)
ax0.annotate(
    f"Best Fit = {R_main['best_fit']:.4f}\n{len(R_main['selected'])} features",
    xy=(gens_main[-1], R_main["h_best"][-1]),
    xytext=(int(len(gens_main) * 0.65), R_main["best_fit"] - 0.05),
    fontsize=9, color=C_BEST,
    arrowprops=dict(arrowstyle="->", color=C_BEST, lw=1.2),
)
ax0.set_xlabel("Generation", fontsize=10)
ax0.set_ylabel("Penalized Fitness Score", fontsize=10)
ax0.set_title(f"Default GA Convergence  ·  sel=Tournament  |  cx=Two-point  |  mut=Bit-flip", fontsize=11)
ax0.legend(fontsize=9, framealpha=0.9, loc="lower right")
style_ax(ax0)

ax1 = fig1.add_subplot(gs1[1, 0])
for (name, r), col in zip(R_sel.items(), P_SEL):
    g = list(range(1, len(r["h_best"]) + 1))
    ax1.plot(g, r["h_best"], color=col, linewidth=1.8, marker="o", markersize=4, label=f"{name} ({r['best_fit']:.4f})", zorder=3)
ax1.set_xlabel("Generation", fontsize=9)
ax1.set_ylabel("Best Fitness", fontsize=9)
ax1.set_title("Selection Methods Comparison", fontsize=10)
ax1.legend(fontsize=8.5, framealpha=0.9)
style_ax(ax1)

ax2 = fig1.add_subplot(gs1[1, 1])
for (name, r), col in zip(R_cx.items(), P_CX):
    g = list(range(1, len(r["h_best"]) + 1))
    ax2.plot(g, r["h_best"], color=col, linewidth=1.8, marker="s", markersize=4, label=f"{name} ({r['best_fit']:.4f})", zorder=3)
ax2.set_xlabel("Generation", fontsize=9)
ax2.set_ylabel("Best Fitness", fontsize=9)
ax2.set_title("Crossover Operators Comparison", fontsize=10)
ax2.legend(fontsize=8.5, framealpha=0.9)
style_ax(ax2)

ax3 = fig1.add_subplot(gs1[2, 0])
for (name, r), col in zip(R_mut.items(), P_MUT):
    g = list(range(1, len(r["h_best"]) + 1))
    ax3.plot(g, r["h_best"], color=col, linewidth=1.8, marker="^", markersize=4, label=f"{name} ({r['best_fit']:.4f})", zorder=3)
ax3.set_xlabel("Generation", fontsize=9)
ax3.set_ylabel("Best Fitness", fontsize=9)
ax3.set_title("Mutation Operators Comparison", fontsize=10)
ax3.legend(fontsize=8.5, framealpha=0.9)
style_ax(ax3)

ax4 = fig1.add_subplot(gs1[2, 1])
configs = (
    [(f"Sel: {n}",  r["best_fit"], P_SEL[i]) for i, (n, r) in enumerate(R_sel.items())] +
    [(f"Cx:  {n}",  r["best_fit"], P_CX[i])  for i, (n, r) in enumerate(R_cx.items())]  +
    [(f"Mut: {n}",  r["best_fit"], P_MUT[i]) for i, (n, r) in enumerate(R_mut.items())]
)
bar_labels = [c[0] for c in configs]
bar_vals   = [c[1] for c in configs]
bar_colors = [c[2] for c in configs]
bars = ax4.barh(bar_labels, bar_vals, color=bar_colors, height=0.55, zorder=3)
for b, v in zip(bars, bar_vals):
    ax4.text(v + 0.001, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center", fontsize=8.5)
ax4.set_xlabel("Best Fitness Score", fontsize=9)
ax4.set_title("Operator Comparison Summary", fontsize=10)
lo = min(bar_vals) * 0.99
hi = max(bar_vals) * 1.015
ax4.set_xlim(lo, hi)
style_ax(ax4)

fig1.suptitle(f"Genetic Algorithm — Feature Selection  ·  Early Stage Diabetes\nBest: {len(R_main['selected'])}/{N} features  ·  Penalized Fit = {R_main['best_fit']:.4f}", fontsize=12, fontweight="bold", y=1.01)
plt.savefig("ga_convergence.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.show()

fig2 = plt.figure(figsize=(18, 7))
gs2  = gridspec.GridSpec(1, 3, figure=fig2, wspace=0.35)

ax_f = fig2.add_subplot(gs2[0, 0])
sort_idx     = np.argsort(sel_freq_pct)[::-1]
feat_sorted  = [FEATURE_NAMES[i] for i in sort_idx]
freq_sorted  = [sel_freq_pct[i] for i in sort_idx]
stable_selected = [bool(i in stable_idx) for i in sort_idx]
bar_c = ["#2ca02c" if s else "#cccccc" for s in stable_selected]
ax_f.barh(feat_sorted, freq_sorted, color=bar_c, height=0.6, zorder=3)
ax_f.axvline(50, color=C_BEST, linewidth=1.0, linestyle="--", alpha=0.7, label="50% threshold")
for i, (pct, sel) in enumerate(zip(freq_sorted, stable_selected)):
    ax_f.text(pct + 2, i, f"{pct:.0f}%", va="center", fontsize=8, color="#2ca02c" if sel else "#888888")
ax_f.set_xlabel("Selection Frequency (%)", fontsize=9)
ax_f.set_xlim(0, 120)
ax_f.set_title(f"Feature Stability Importance\n(frequency across {STABILITY_RUNS} GA runs)\n■ green = stable features (>= 50%)", fontsize=10)
ax_f.legend(fontsize=8, loc="lower right")
style_ax(ax_f)

ax_b = fig2.add_subplot(gs2[0, 1])
draw_cm(ax_b, cm_base, f"Baseline — All {N} Features\nAcc={acc_base:.4f}  AUC={auc_base:.4f}")

ax_g = fig2.add_subplot(gs2[0, 2])
draw_cm(ax_g, cm_ga, f"GA Stability Selected — {len(stable_names)} Features\nAcc={acc_ga:.4f}  AUC={auc_ga:.4f}")

fig2.suptitle("Model Evaluation: Baseline vs GA Stability Selection  ·  MLP(16, 8)  ·  Early Stage Diabetes", fontsize=12, fontweight="bold")
plt.savefig("ga_results.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.show()

print(f"\n{'='*70}")
print(f"  {'Metric':<25} {'Baseline':>12} {'GA-Stable':>12}")
print(f"  {'─'*49}")
print(f"  {'# Features':<25} {N:>12} {len(stable_names):>12}")
print(f"  {'Test Accuracy (Mean)':<25} {acc_base:>12.4f} {acc_ga:>12.4f}")
print(f"  {'Test AUC-ROC (Mean)':<25} {auc_base:>12.4f} {auc_ga:>12.4f}")
print(f"  {'Features Removed':<25} {'—':>12} {N - len(stable_names):>12}")
print(f"{'='*70}")
print(f"  Selected: {stable_names}")
print(f"{'='*70}")
