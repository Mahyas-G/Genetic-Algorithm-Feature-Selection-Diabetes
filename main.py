import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

df = pd.read_csv("early_stage_diabetes.csv")
df.columns = df.columns.str.strip()

print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['Class'].value_counts()}\n")

encoders = {}
for col in df.select_dtypes(include=['object', 'string']).columns:
    enc = LabelEncoder()
    df[col] = enc.fit_transform(df[col].astype(str))
    encoders[col] = enc

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

architectures = [
    (8,), (16,), (32,), (64,),
    (8, 8), (16, 16), (32, 16), (32, 32), (64, 32),
    (16, 8, 4), (32, 16, 8), (64, 32, 16)
]

best_architecture = None
best_cv_auc_score = -1
best_cv_acc_score = -1

train_accs   = []
test_accs    = []
test_aucs    = []
cv_acc_means = []
cv_acc_stds  = []
labels       = []

print(f"{'Architecture':<18} | {'Train Acc':>9} | {'Test Acc':>8} | {'Test AUC':>8} | {'CV Acc':>7} ±std | {'CV AUC':>6}")
print("-" * 85)

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for arch in architectures:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=arch,
            activation='relu',
            solver='adam',
            alpha=0.0005,
            learning_rate_init=0.001,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            max_iter=2000,
            random_state=42,
        ))
    ])
    
    cv_results = cross_validate(
        pipeline, X_train, y_train, 
        cv=cv_strategy, 
        scoring=['accuracy', 'roc_auc']
    )
    
    cv_acc_mean = cv_results['test_accuracy'].mean()
    cv_acc_std  = cv_results['test_accuracy'].std()
    cv_auc_mean = cv_results['test_roc_auc'].mean()

    model = MLPClassifier(
        hidden_layer_sizes=arch,
        activation='relu',
        solver='adam',
        alpha=0.0005,
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        max_iter=2000,
        random_state=42,
    )
    model.fit(X_train_s, y_train)

    train_pred = model.predict(X_train_s)
    test_pred  = model.predict(X_test_s)
    test_proba = model.predict_proba(X_test_s)[:, 1]

    train_acc = accuracy_score(y_train, train_pred)
    test_acc  = accuracy_score(y_test,  test_pred)
    test_auc  = roc_auc_score(y_test, test_proba)

    train_accs.append(train_acc)
    test_accs.append(test_acc)
    test_aucs.append(test_auc)
    cv_acc_means.append(cv_acc_mean)
    cv_acc_stds.append(cv_acc_std)
    labels.append(str(arch))

    print(f"{str(arch):<18} | {train_acc:9.4f} | {test_acc:8.4f} | {test_auc:8.4f} | {cv_acc_mean:7.4f} ±{cv_acc_std:.3f} | {cv_auc_mean:6.4f}")

    if (cv_auc_mean > best_cv_auc_score) or (np.isclose(cv_auc_mean, best_cv_auc_score) and cv_acc_mean > best_cv_acc_score):
        best_cv_auc_score = cv_auc_mean
        best_cv_acc_score = cv_acc_mean
        best_architecture = arch

best_model = MLPClassifier(
    hidden_layer_sizes=best_architecture,
    activation='relu',
    solver='adam',
    alpha=0.0005,
    learning_rate_init=0.001,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    max_iter=2000,
    random_state=42,
)
best_model.fit(X_train_s, y_train)

predictions = best_model.predict(X_test_s)
test_proba  = best_model.predict_proba(X_test_s)[:, 1]
final_test_accuracy = accuracy_score(y_test, predictions)
final_test_auc = roc_auc_score(y_test, test_proba)

print("\n" + "=" * 55)
print(f"  BEST ARCHITECTURE (BY CV AUC/ACC): {best_architecture}")
print(f"  BEST CV AUC SCORE                : {best_cv_auc_score:.4f}")
print(f"  FINAL TEST ACCURACY              : {final_test_accuracy:.4f} ({final_test_accuracy*100:.2f}%)")
print(f"  FINAL TEST AUC                   : {final_test_auc:.4f}")
print("=" * 55)

print("\nClassification Report\n")
print(classification_report(y_test, predictions, target_names=["Negative", "Positive"]))

print("\nConfusion Matrix")
cm = confusion_matrix(y_test, predictions)
print(cm)

x_idx  = list(range(len(labels)))
best_i = labels.index(str(best_architecture))

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.25, height_ratios=[1.3, 1])

C_TRAIN = '#1f77b4'
C_TEST  = '#ff7f0e'
C_AUC   = '#2ca02c'
C_CV    = '#9467bd'
C_BEST  = '#d62728'
C_GRID  = '#cccccc'

def style_ax(ax):
    ax.grid(True, color=C_GRID, linewidth=0.7, zorder=0)
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color('#aaaaaa')

ax1 = fig.add_subplot(gs[0, :])
ax1.plot(x_idx, train_accs, marker='o', color=C_TRAIN, linewidth=1.8, markersize=5, label='Train Accuracy', zorder=3)
ax1.plot(x_idx, test_accs,  marker='s', color=C_TEST,  linewidth=1.8, markersize=5, label='Holdout Test Accuracy',  zorder=3)
ax1.plot(x_idx, cv_acc_means, marker='^', color=C_CV,    linewidth=1.4, markersize=5, linestyle='--', label='CV Accuracy (5-fold)', zorder=3)

ax1.fill_between(
    x_idx,
    [m - s for m, s in zip(cv_acc_means, cv_acc_stds)],
    [m + s for m, s in zip(cv_acc_means, cv_acc_stds)],
    color=C_CV, alpha=0.12, label='CV ± std', zorder=2
)

ax1.axvline(best_i, color=C_BEST, linewidth=1.2, linestyle=':', alpha=0.8, zorder=2)
ax1.annotate(
    f'Best by AUC/Acc: {best_architecture}\nTest Acc={final_test_accuracy:.4f}',
    xy=(best_i, test_accs[best_i]),
    xytext=(best_i + 0.4, test_accs[best_i] - 0.05),
    fontsize=8,
    color=C_BEST,
    arrowprops=dict(arrowstyle='->', color=C_BEST, lw=1.0),
)

ax1.set_xticks(x_idx)
ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax1.set_ylabel("Accuracy", fontsize=10)
ax1.set_ylim(0.65, 1.02)
ax1.set_title("MLP Architecture Comparison — Train / Holdout Test / CV Accuracy", fontsize=11)
ax1.legend(loc='lower right', fontsize=8, framealpha=0.9)
style_ax(ax1)

ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(x_idx, test_aucs, marker='D', color=C_AUC, linewidth=1.8, markersize=5, label='Holdout Test AUC-ROC', zorder=3)
ax2.axvline(best_i, color=C_BEST, linewidth=1.2, linestyle=':', alpha=0.8)

for i, v in enumerate(test_aucs):
    ax2.annotate(f'{v:.3f}', xy=(i, v), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=6.5, color='#444444')

ax2.set_xticks(x_idx)
ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax2.set_ylabel("AUC-ROC", fontsize=10)
ax2.set_ylim(0.92, 1.01)
ax2.set_title("Holdout Test AUC-ROC per Architecture", fontsize=11)
ax2.legend(fontsize=8, framealpha=0.9)
style_ax(ax2)

ax3 = fig.add_subplot(gs[1, 1])
im = ax3.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im, ax=ax3, fraction=0.04, pad=0.03)
classes = ['Negative', 'Positive']
tick_marks = [0, 1]
ax3.set_xticks(tick_marks)
ax3.set_yticks(tick_marks)
ax3.set_xticklabels(classes, fontsize=9)
ax3.set_yticklabels(classes, fontsize=9)
ax3.set_xlabel("Predicted", fontsize=9)
ax3.set_ylabel("Actual", fontsize=9)
ax3.set_title("Confusion Matrix (Best Model)", fontsize=11)

thresh = cm.max() / 2.0
for i in range(2):
    for j in range(2):
        ax3.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14, fontweight='bold',
                 color='white' if cm[i, j] > thresh else '#333333')

fig.suptitle(
    f"Early Stage Diabetes — MLP Neural Network  |  "
    f"Best CV Arch: {best_architecture}  |  Test Acc: {final_test_accuracy*100:.2f}%  |  "
    f"Test AUC: {final_test_auc:.4f}",
    fontsize=11, fontweight='bold', y=0.995
)

plt.savefig("result.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("\nPlot saved to result.png")
