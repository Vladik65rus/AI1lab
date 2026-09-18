# %% [markdown]
# # Лабораторная работа 1. Линейная регрессия и факторный анализ
# Датасет: California Housing Prices (Kaggle: camnugent/california-housing-prices)
# Целевая переменная: median_house_value — медианная стоимость дома в квартале.

# %% Импорт библиотек
import os
import sys
import urllib.request
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.family"] = "DejaVu Sans"

# Рабочая папка = папка со скриптом (чтобы пути не зависели от способа запуска)
try:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass  # запуск в Jupyter — рабочая папка уже верная

FIG = "figs/"
os.makedirs(FIG, exist_ok=True)
RES = {}

# %% 1. Загрузка данных
CSV = "housing.csv"
URL = ("https://raw.githubusercontent.com/ageron/handson-ml2/"
       "master/datasets/housing/housing.csv")
if not os.path.exists(CSV):
    print("Файл не найден, скачиваем датасет...")
    try:
        urllib.request.urlretrieve(URL, CSV)
    except Exception as e:
        sys.exit(f"Не удалось скачать датасет ({e}). "
                 f"Скачайте вручную по ссылке {URL} и положите файл рядом со скриптом.")

df = pd.read_csv(CSV)
print("Размер датасета:", df.shape)
print(df.head())
print(df.info())
print(df.describe().T)
RES["shape_raw"] = list(df.shape)
RES["describe"] = df.describe().T.round(2).to_dict()

# %% 2. Первичный анализ: пропуски
na = df.isna().sum()
print("Пропуски:\n", na[na > 0])
RES["na"] = na[na > 0].to_dict()
print("Дубликаты:", df.duplicated().sum())
RES["dupl"] = int(df.duplicated().sum())
RES["cat_counts"] = df["ocean_proximity"].value_counts().to_dict()

# %% 2.1 Распределения признаков
num_cols = df.select_dtypes(include=np.number).columns.tolist()
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
for ax, c in zip(axes.ravel(), num_cols):
    ax.hist(df[c].dropna(), bins=50, color="#4C72B0", edgecolor="white", linewidth=0.3)
    ax.set_title(c, fontsize=10)
fig.suptitle("Распределение числовых признаков", fontsize=13)
fig.tight_layout()
fig.savefig(FIG + "01_hist.png", bbox_inches="tight")
plt.close(fig)

# %% 2.2 Целевая переменная
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(df["median_house_value"], bins=50, color="#C44E52", edgecolor="white", linewidth=0.3)
axes[0].set_title("median_house_value (исходная)")
axes[1].boxplot(df["median_house_value"], vert=False)
axes[1].set_title("Ящик с усами")
fig.tight_layout()
fig.savefig(FIG + "02_target.png", bbox_inches="tight")
plt.close(fig)

# %% 2.3 Категориальный признак
fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(data=df, x="ocean_proximity", y="median_house_value", ax=ax, palette="Set2")
ax.set_title("Стоимость жилья по категории удалённости от океана")
ax.tick_params(axis="x", rotation=15)
fig.tight_layout()
fig.savefig(FIG + "03_cat.png", bbox_inches="tight")
plt.close(fig)

# %% 3. Предобработка
# 3.1 Удаление пропусков (207 строк в total_bedrooms — менее 1 % выборки)
df = df.dropna().reset_index(drop=True)

# 3.2 Удаление "цензурированных" наблюдений (значения обрезаны сверху на 500001)
cap = df["median_house_value"].max()
n_cap = int((df["median_house_value"] >= cap).sum())
RES["cap_value"], RES["n_cap"] = float(cap), n_cap
df = df[df["median_house_value"] < cap].reset_index(drop=True)

# 3.3 Производные признаки (нормировка на размер квартала)
df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
df["population_per_household"] = df["population"] / df["households"]

# 3.4 Кодирование категориальной переменной (one-hot, dummy-кодирование)
df = pd.get_dummies(df, columns=["ocean_proximity"], prefix="op", drop_first=True, dtype=float)

TARGET = "median_house_value"
X = df.drop(columns=[TARGET])
y = df[TARGET]
print("Размер после предобработки:", df.shape)
print("Признаки:", list(X.columns))
RES["shape_clean"] = list(df.shape)
RES["features"] = list(X.columns)

# %% 4. Корреляционный анализ
corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            annot_kws={"size": 7}, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("Матрица корреляций (Пирсон)")
fig.tight_layout()
fig.savefig(FIG + "04_corr.png", bbox_inches="tight")
plt.close(fig)

target_corr = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
print("Корреляция с целевой переменной:\n", target_corr.round(3))
RES["target_corr"] = target_corr.round(3).to_dict()

# %% 4.1 Диаграммы рассеяния с наиболее значимыми признаками
top3 = target_corr.abs().sort_values(ascending=False).index[:3]
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, c in zip(axes, top3):
    ax.scatter(df[c], y, s=4, alpha=0.2, color="#4C72B0")
    ax.set_xlabel(c); ax.set_ylabel(TARGET)
    ax.set_title(f"r = {corr.loc[c, TARGET]:.2f}", fontsize=10)
fig.suptitle("Зависимость целевой переменной от ключевых признаков")
fig.tight_layout()
fig.savefig(FIG + "05_scatter.png", bbox_inches="tight")
plt.close(fig)

# %% 4.2 Расчёт VIF (фактор инфляции дисперсии)
def vif_table(X_df):
    Xs = StandardScaler().fit_transform(X_df)
    Xs = np.column_stack([np.ones(len(Xs)), Xs])
    return pd.DataFrame({
        "Признак": X_df.columns,
        "VIF": [variance_inflation_factor(Xs, i + 1) for i in range(X_df.shape[1])]
    }).sort_values("VIF", ascending=False).reset_index(drop=True)

vif = vif_table(X)
print(vif.round(2))
RES["vif"] = vif.round(2).to_dict("records")

# %% 5. Разделение выборки и стандартизация
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE)
print("train:", X_train.shape, " test:", X_test.shape)
RES["split"] = {"train": list(X_train.shape), "test": list(X_test.shape)}

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# %% 5.1 Функция оценки качества
def evaluate(model, Xtr, ytr, Xte, yte, name):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_r2 = cross_val_score(model, Xtr, ytr, cv=cv, scoring="r2")
    cv_rmse = -cross_val_score(model, Xtr, ytr, cv=cv, scoring="neg_root_mean_squared_error")
    row = {
        "Модель": name,
        "RMSE": float(np.sqrt(mean_squared_error(yte, pred))),
        "R2": float(r2_score(yte, pred)),
        "MAPE, %": float(mean_absolute_percentage_error(yte, pred) * 100),
        "CV R2 (5-fold)": float(cv_r2.mean()),
        "CV RMSE (5-fold)": float(cv_rmse.mean()),
    }
    print(f"{name}: RMSE={row['RMSE']:.1f}  R2={row['R2']:.4f}  "
          f"MAPE={row['MAPE, %']:.2f}%  CV R2={row['CV R2 (5-fold)']:.4f}")
    return row, pred

results = []

# %% 5.2 Линейная регрессия на исходных признаках
lin = LinearRegression()
row_lin, pred_lin = evaluate(lin, X_train_s, y_train, X_test_s, y_test,
                             "Линейная регрессия (исходные признаки)")
results.append(row_lin)

coefs = pd.Series(lin.coef_, index=X.columns).sort_values(key=abs, ascending=False)
print("Коэффициенты (на стандартизованных данных):\n", coefs.round(1))
RES["coefs"] = coefs.round(1).to_dict()

# %% 5.3 Гребневая регрессия (подбор alpha по сетке с кросс-валидацией)
grid = GridSearchCV(Ridge(), {"alpha": np.logspace(-3, 4, 40)},
                    cv=KFold(5, shuffle=True, random_state=RANDOM_STATE), scoring="r2")
grid.fit(X_train_s, y_train)
best_alpha = grid.best_params_["alpha"]
print("Лучшее alpha:", round(best_alpha, 4))
RES["best_alpha"] = float(best_alpha)

row_ridge, pred_ridge = evaluate(Ridge(alpha=best_alpha), X_train_s, y_train,
                                 X_test_s, y_test, "Гребневая регрессия (исходные признаки)")
results.append(row_ridge)

# %% 5.4 Диагностика остатков
resid = y_test - pred_lin
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(pred_lin, resid, s=5, alpha=0.2, color="#4C72B0")
axes[0].axhline(0, color="red", lw=1)
axes[0].set_xlabel("Предсказание"); axes[0].set_ylabel("Остаток")
axes[0].set_title("Остатки против предсказаний")
axes[1].scatter(y_test, pred_lin, s=5, alpha=0.2, color="#55A868")
lims = [y_test.min(), y_test.max()]
axes[1].plot(lims, lims, color="red", lw=1)
axes[1].set_xlabel("Факт"); axes[1].set_ylabel("Прогноз")
axes[1].set_title("Факт против прогноза")
fig.tight_layout()
fig.savefig(FIG + "06_resid.png", bbox_inches="tight")
plt.close(fig)

# %% 6. Метод главных компонент (PCA)
pca_full = PCA().fit(X_train_s)
evr = pca_full.explained_variance_ratio_
cum = np.cumsum(evr)
n_comp = int(np.argmax(cum >= 0.95) + 1)
print("Доля объяснённой дисперсии:", np.round(evr, 3))
print("Накопленная:", np.round(cum, 3))
print("Компонент для 95 % дисперсии:", n_comp)
RES["evr"] = [float(v) for v in evr]
RES["cum"] = [float(v) for v in cum]
RES["n_comp"] = n_comp
RES["eigenvalues"] = [float(v) for v in pca_full.explained_variance_]

# %% 6.1 График каменистой осыпи (scree plot)
fig, ax = plt.subplots(figsize=(8, 4.5))
xs = np.arange(1, len(evr) + 1)
ax.bar(xs, evr, color="#4C72B0", label="Доля дисперсии компоненты")
ax.plot(xs, cum, "o-", color="#C44E52", label="Накопленная доля")
ax.axhline(0.95, ls="--", color="gray", lw=1)
ax.axvline(n_comp, ls="--", color="green", lw=1)
ax.set_xlabel("Номер главной компоненты"); ax.set_ylabel("Доля объяснённой дисперсии")
ax.set_title("График каменистой осыпи (scree plot)")
ax.set_xticks(xs); ax.legend()
fig.tight_layout()
fig.savefig(FIG + "07_scree.png", bbox_inches="tight")
plt.close(fig)

# %% 6.2 Матрица нагрузок первых компонент
pca = PCA(n_components=n_comp, random_state=RANDOM_STATE).fit(X_train_s)
X_train_p = pca.transform(X_train_s)
X_test_p = pca.transform(X_test_s)

loadings = pd.DataFrame(pca.components_.T,
                        index=X.columns,
                        columns=[f"PC{i+1}" for i in range(n_comp)])
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(loadings, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            annot_kws={"size": 7}, ax=ax)
ax.set_title("Нагрузки признаков на главные компоненты")
fig.tight_layout()
fig.savefig(FIG + "08_loadings.png", bbox_inches="tight")
plt.close(fig)
RES["loadings"] = loadings.round(2).to_dict()

# VIF после PCA (компоненты ортогональны -> VIF ~ 1)
vif_pca = vif_table(pd.DataFrame(X_train_p, columns=loadings.columns))
print(vif_pca.round(3))
RES["vif_pca"] = vif_pca.round(3).to_dict("records")

# %% 7. Регрессия на главных компонентах
row_lin_p, _ = evaluate(LinearRegression(), X_train_p, y_train, X_test_p, y_test,
                        f"Линейная регрессия (PCA, {n_comp} компонент)")
results.append(row_lin_p)

grid_p = GridSearchCV(Ridge(), {"alpha": np.logspace(-3, 4, 40)},
                      cv=KFold(5, shuffle=True, random_state=RANDOM_STATE), scoring="r2")
grid_p.fit(X_train_p, y_train)
alpha_p = grid_p.best_params_["alpha"]
RES["best_alpha_pca"] = float(alpha_p)
row_ridge_p, _ = evaluate(Ridge(alpha=alpha_p), X_train_p, y_train, X_test_p, y_test,
                          f"Гребневая регрессия (PCA, {n_comp} компонент)")
results.append(row_ridge_p)

# %% 7.1 Сводная таблица метрик
metrics = pd.DataFrame(results).round(4)
print(metrics.to_string(index=False))
metrics.to_csv("metrics.csv", index=False, encoding="utf-8-sig")
RES["metrics"] = metrics.to_dict("records")

# %% 7.2 Сравнение метрик графически
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
names = ["ЛР\n(исходные)", "Ridge\n(исходные)", f"ЛР\n(PCA-{n_comp})", f"Ridge\n(PCA-{n_comp})"]
for ax, col, ttl in zip(axes, ["RMSE", "R2", "MAPE, %"], ["RMSE (меньше — лучше)",
                                                          "R² (больше — лучше)",
                                                          "MAPE, % (меньше — лучше)"]):
    bars = ax.bar(names, metrics[col], color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    ax.set_title(ttl, fontsize=11)
    ax.bar_label(bars, fmt="%.3f" if col == "R2" else "%.2f", fontsize=9)
    ax.margins(y=0.15)
fig.suptitle("Сравнение качества моделей до и после PCA")
fig.tight_layout()
fig.savefig(FIG + "09_metrics.png", bbox_inches="tight")
plt.close(fig)

# %% 7.3 Зависимость качества от числа компонент
scores = []
for k in range(1, X_train_s.shape[1] + 1):
    p = PCA(n_components=k, random_state=RANDOM_STATE).fit(X_train_s)
    m = LinearRegression().fit(p.transform(X_train_s), y_train)
    scores.append(r2_score(y_test, m.predict(p.transform(X_test_s))))
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(scores) + 1), scores, "o-", color="#4C72B0")
ax.axhline(row_lin["R2"], ls="--", color="red", label="ЛР на исходных признаках")
ax.set_xlabel("Число главных компонент"); ax.set_ylabel("R² на тестовой выборке")
ax.set_title("Качество модели в зависимости от числа компонент")
ax.legend(); ax.set_xticks(range(1, len(scores) + 1))
fig.tight_layout()
fig.savefig(FIG + "10_r2_vs_k.png", bbox_inches="tight")
plt.close(fig)
RES["r2_vs_k"] = [float(s) for s in scores]

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(RES, f, ensure_ascii=False, indent=1, default=str)
print("Готово.")
