import pandas as pd
import numpy as np
import random
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

warnings.filterwarnings("ignore")

np.random.seed(42)
random.seed(42)

df = pd.read_csv("early_stage_diabetes.csv")
df.columns = df.columns.str.strip()

for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

X = df.drop("Class", axis=1).values
y = df["Class"].values
feature_names = df.drop("Class", axis=1).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

n_features = X.shape[1]

POP_SIZE = 20
GENERATIONS = 15
ELITE_SIZE = 2
MUTATION_RATE = 0.05
LAMBDA_PENALTY = 0.1

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fitness_cache = {}

def fitness(chrom):
    key = tuple(chrom)
    if key in fitness_cache:
        return fitness_cache[key]

    num_selected = np.sum(chrom)
    if num_selected == 0:
        fitness_cache[key] = 0.0
        return 0.0

    idx = np.where(chrom == 1)[0]
    X_sub = X_train[:, idx]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(8,),
            max_iter=300,
            early_stopping=True,
            random_state=42
        ))
    ])

    base_score = cross_val_score(model, X_sub, y_train, cv=cv, scoring="accuracy").mean()
    
    penalty = LAMBDA_PENALTY * (num_selected / n_features)
    final_score = base_score - penalty

    fitness_cache[key] = final_score
    return final_score

pop = np.random.randint(0, 2, (POP_SIZE, n_features))
for i in range(len(pop)):
    if pop[i].sum() == 0:
        pop[i, np.random.randint(n_features)] = 1

def tournament(pop, fit, k=3):
    idx = np.random.choice(len(pop), k, replace=False)
    best = idx[np.argmax(fit[idx])]
    return pop[best].copy()

def rank(pop, fit):
    ranks = np.argsort(np.argsort(fit)) + 1
    squared_ranks = ranks ** 2
    probs = squared_ranks / squared_ranks.sum()
    return pop[np.random.choice(len(pop), p=probs)].copy()

def uniform(p1, p2):
    mask = np.random.rand(n_features) < 0.5
    return np.where(mask, p1, p2), np.where(mask, p2, p1)

def two_point(p1, p2):
    a, b = sorted(np.random.choice(n_features, 2, replace=False))
    c1, c2 = p1.copy(), p2.copy()
    c1[a:b], c2[a:b] = p2[a:b].copy(), p1[a:b].copy()
    return c1, c2

def bit_flip_mutate(x):
    for i in range(n_features):
        if np.random.rand() < MUTATION_RATE:
            x[i] = 1 - x[i]
    return x

def random_reset_mutate(x):
    num_resets = np.random.randint(2, 5)
    if n_features >= num_resets:
        indices = np.random.choice(n_features, num_resets, replace=False)
        x[indices] = np.random.randint(2, size=num_resets)
    return x

def ensure_valid(x):
    if np.sum(x) == 0:
        x[np.random.randint(n_features)] = 1
    return x

best, best_fit = None, -1
best_history = []
avg_history = []

print(f"{'Generation':<12} | {'Best Fitness':<15} | {'Avg Fitness':<15} | {'Selected Features'}")
print("-" * 65)

for gen in range(GENERATIONS):
    fit = np.array([fitness(ind) for ind in pop])
    
    gen_best_idx = np.argmax(fit)
    gen_best_fit = fit[gen_best_idx]
    gen_avg_fit = np.mean(fit)
    
    best_history.append(gen_best_fit)
    avg_history.append(gen_avg_fit)

    if gen_best_fit > best_fit:
        best_fit = gen_best_fit
        best = pop[gen_best_idx].copy()

    num_features_gen = np.sum(pop[gen_best_idx])
    print(f"{gen+1:<12} | {gen_best_fit:<15.4f} | {gen_avg_fit:<15.4f} | {num_features_gen}")

    ranked_indices = np.argsort(fit)[::-1]
    ranked_pop = pop[ranked_indices]

    new_pop = list(ranked_pop[:ELITE_SIZE])

    while len(new_pop) < POP_SIZE:
        p1 = tournament(pop, fit) if random.random() < 0.5 else rank(pop, fit)
        p2 = tournament(pop, fit) if random.random() < 0.5 else rank(pop, fit)

        if random.random() < 0.5:
            c1, c2 = uniform(p1, p2)
        else:
            c1, c2 = two_point(p1, p2)

        c1 = bit_flip_mutate(c1) if random.random() < 0.5 else random_reset_mutate(c1)
        c1 = ensure_valid(c1)
        new_pop.append(c1)
        
        if len(new_pop) < POP_SIZE:
            c2 = bit_flip_mutate(c2) if random.random() < 0.5 else random_reset_mutate(c2)
            c2 = ensure_valid(c2)
            new_pop.append(c2)

    pop = np.array(new_pop)

selected = np.where(best == 1)[0]
final_features = [feature_names[i] for i in selected]

Xtr, Xte = X_train[:, selected], X_test[:, selected]

final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(hidden_layer_sizes=(8,), max_iter=500, random_state=42))
])

final_model.fit(Xtr, y_train)
pred = final_model.predict(Xte)
test_proba = final_model.predict_proba(Xte)[:, 1]

final_acc = accuracy_score(y_test, pred)
final_auc = roc_auc_score(y_test, test_proba)
cm = confusion_matrix(y_test, pred)

print("\n" + "=" * 55)
print("  FINAL EVALUATION ON HOLDOUT TEST SET")
print("=" * 55)
print(f"  FEATURES SELECTED : {len(selected)} out of {n_features}")
print(f"  TEST ACCURACY     : {final_acc:.4f} ({final_acc*100:.2f}%)")
print(f"  TEST AUC-ROC      : {final_auc:.4f}")
print("=" * 55)
print("\nSelected Features List:")
print(final_features)
print("\nClassification Report:\n")
print(classification_report(y_test, pred, target_names=["Negative", "Positive"]))

fig = plt.figure(figsize=(15, 6))
gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.5, 1], wspace=0.3)

C_BEST = '#2ca02c'
C_AVG = '#1f77b4'
C_GRID = '#cccccc'

ax1 = fig.add_subplot(gs[0, 0])
ax1.grid(True, color=C_GRID, linewidth=0.7, zorder=0)
ax1.set_facecolor('white')
for spine in ax1.spines.values():
    spine.set_linewidth(0.6)
    spine.set_color('#aaaaaa')

ax1.plot(range(1, len(best_history) + 1), best_history, marker='o', color=C_BEST, linewidth=2, markersize=6, label='Best Fitness', zorder=3)
ax1.plot(range(1, len(avg_history) + 1), avg_history, marker='s', color=C_AVG, linewidth=1.5, markersize=5, linestyle='--', label='Average Fitness', zorder=3)
ax1.set_title("GA Convergence (Penalized CV Accuracy)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Generations", fontsize=10)
ax1.set_ylabel("Fitness Score", fontsize=10)
ax1.set_xticks(range(1, len(best_history) + 1))
ax1.legend(loc='lower right')

ax2 = fig.add_subplot(gs[0, 1])
im = ax2.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im, ax=ax2, fraction=0.04, pad=0.04)
classes = ['Negative', 'Positive']
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(classes, fontsize=9)
ax2.set_yticklabels(classes, fontsize=9)
ax2.set_xlabel("Predicted", fontsize=9)
ax2.set_ylabel("Actual", fontsize=9)
ax2.set_title("Final Model Confusion Matrix", fontsize=12, fontweight='bold')

thresh = cm.max() / 2.0
for i in range(2):
    for j in range(2):
        ax2.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14, fontweight='bold',
                 color='white' if cm[i, j] > thresh else '#333333')

fig.suptitle(
    f"\n"
    f"Test Acc: {final_acc:.4f} | Test AUC: {final_auc:.4f} | Features: {len(selected)}",
    fontsize=13, fontweight='bold', y=1.02
)

plt.savefig("result.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
