import os
import sys
import warnings

warnings.filterwarnings("ignore")

import imblearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "..", "Dataset", "IBM_HR_Attrition.xlsx")
OUTPUT_DIR = os.path.normpath(os.path.join(script_dir, ".."))


def out(nama_file):
    return os.path.join(OUTPUT_DIR, nama_file)


def judul(teks):
    print("\n" + "=" * 70)
    print(teks)
    print("=" * 70)


# ----------------------------------------------------------------------
# 3.1 PERSIAPAN LIBRARY
# ----------------------------------------------------------------------
judul("PROGRAM KLASIFIKASI IBM HR ATTRITION")
print("Python           :", sys.version.split()[0])
print("pandas           :", pd.__version__)
print("numpy            :", np.__version__)
print("scikit-learn     :", sklearn.__version__)
print("imbalanced-learn :", imblearn.__version__)
print("random_state     :", RANDOM_STATE)

# ----------------------------------------------------------------------
# 3.2 INPUT DATA
# ----------------------------------------------------------------------
judul("INPUT DATA")
df = pd.read_excel(dataset_path, sheet_name=0, engine="openpyxl")
print("Dataset berhasil dibaca.")
print("Jumlah baris dan kolom:", df.shape)
print("\n5 baris pertama:")
print(df.head())
print("\nInfo dataset:")
df.info()

# ----------------------------------------------------------------------
# 3.3 EKSPLORASI DISTRIBUSI KELAS
# ----------------------------------------------------------------------
judul("DISTRIBUSI KELAS ATTRITION")
distribusi = df["Attrition"].value_counts()
print("Jumlah:")
print(distribusi)
print("\nPersentase (%):")
print((df["Attrition"].value_counts(normalize=True) * 100).round(2))
print("\nRasio No : Yes = %.2f : 1" % (distribusi["No"] / distribusi["Yes"]))

fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
sns.countplot(x="Attrition", data=df, ax=axes[0], palette="coolwarm")
axes[0].set_title("Distribusi Attrition")
axes[0].set_ylabel("Jumlah")
axes[1].pie(
    distribusi,
    labels=distribusi.index,
    autopct="%1.1f%%",
    colors=["#a8c6f0", "#f0b7a4"],
    startangle=90,
)
axes[1].set_title("Persentase Attrition")
plt.tight_layout()
plt.savefig(out("01_distribusi_attrition.png"), dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 3.4 PREPROCESSING (SEBELUM SPLIT)
# ----------------------------------------------------------------------
judul("DETEKSI MISSING VALUE")
missing_per_kolom = df.isnull().sum()
print(missing_per_kolom.to_string())
total_missing = int(missing_per_kolom.sum())
print("\nTotal missing value:", total_missing)
if total_missing > 0:
    for kolom in df.columns[df.isnull().any()]:
        if df[kolom].dtype == "object":
            df[kolom] = df[kolom].fillna(df[kolom].mode()[0])
        else:
            df[kolom] = df[kolom].fillna(df[kolom].median())
    print("Missing value ditangani (median untuk numerik, modus untuk kategorikal).")
else:
    print("Tidak ada missing value, handling tidak diperlukan.")

judul("DETEKSI DUPLIKASI DATA")
duplikat = int(df.duplicated().sum())
duplikat_tanpa_id = int(df.drop(columns=["EmployeeNumber"]).duplicated().sum())
print("Jumlah baris duplikat                    :", duplikat)
print("Jumlah baris duplikat (tanpa EmployeeNumber):", duplikat_tanpa_id)
if duplikat > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print("Baris duplikat dihapus. Jumlah baris sekarang:", df.shape[0])
else:
    print("Tidak ada duplikasi, jumlah data tetap", df.shape[0])

judul("DETEKSI OUTLIER (METODE IQR)")
outlier_columns = [
    "MonthlyIncome",
    "YearsAtCompany",
    "YearsSinceLastPromotion",
    "TotalWorkingYears",
    "NumCompaniesWorked",
    "YearsInCurrentRole",
    "YearsWithCurrManager",
]
baris = []
for kolom in df.select_dtypes(exclude="object").columns:
    q1, q3 = df[kolom].quantile([0.25, 0.75])
    iqr = q3 - q1
    bawah, atas = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    jumlah = int(((df[kolom] < bawah) | (df[kolom] > atas)).sum())
    if jumlah > 0:
        baris.append(
            {
                "Kolom": kolom,
                "Jumlah Outlier": jumlah,
                "Batas Bawah": round(bawah, 2),
                "Batas Atas": round(atas, 2),
                "Ditangani": "Ya" if kolom in outlier_columns else "Tidak (diskret)",
            }
        )
ringkasan_outlier = pd.DataFrame(baris).sort_values("Jumlah Outlier", ascending=False)
print(ringkasan_outlier.to_string(index=False))

fig, axes = plt.subplots(2, 4, figsize=(10, 6))
axes = axes.flatten()
for index, kolom in enumerate(outlier_columns):
    sns.boxplot(y=df[kolom], ax=axes[index], color="#a8c6f0")
    axes[index].set_title(kolom, fontsize=9)
    axes[index].set_ylabel("")
axes[7].axis("off")
plt.suptitle("Boxplot Deteksi Outlier (Data Penuh)")
plt.tight_layout()
plt.savefig(out("02_boxplot_deteksi_outlier.png"), dpi=150)
plt.close()

judul("PENGHAPUSAN KOLOM TIDAK RELEVAN DAN ENCODING TARGET")
print("Shape sebelum drop:", df.shape)
columns_to_drop = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]
for kolom in columns_to_drop:
    print("  %-15s nilai unik: %d" % (kolom, df[kolom].nunique()))
df = df.drop(columns=columns_to_drop)
print("Shape setelah drop:", df.shape)
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
print("\nEncoding target (Yes = 1, No = 0):")
print(df["Attrition"].value_counts())

# ----------------------------------------------------------------------
# 3.5 SPLIT DATA 80:20
# ----------------------------------------------------------------------
judul("SPLIT DATA 80% TRAINING : 20% TESTING")
X = df.drop(columns=["Attrition"])
y = df["Attrition"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
X_train = X_train.copy()
X_test = X_test.copy()
print("Data training:", X_train.shape[0])
print("Data testing :", X_test.shape[0])
tabel_split = pd.DataFrame(
    {
        "Total": [len(y), len(y_train), len(y_test)],
        "No (0)": [(y == 0).sum(), (y_train == 0).sum(), (y_test == 0).sum()],
        "Yes (1)": [(y == 1).sum(), (y_train == 1).sum(), (y_test == 1).sum()],
    },
    index=["Data awal", "Data training", "Data testing"],
)
tabel_split["% Yes"] = (tabel_split["Yes (1)"] / tabel_split["Total"] * 100).round(2)
print("\nDistribusi kelas:")
print(tabel_split)

# ----------------------------------------------------------------------
# 3.6 HANDLING OUTLIER (BATAS IQR DARI DATA TRAINING)
# ----------------------------------------------------------------------
judul("HANDLING OUTLIER (CAPPING IQR)")
bounds = {}
for kolom in outlier_columns:
    q1 = X_train[kolom].quantile(0.25)
    q3 = X_train[kolom].quantile(0.75)
    iqr = q3 - q1
    bounds[kolom] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)


def hitung_outlier(data, kolom):
    bawah, atas = bounds[kolom]
    return int(((data[kolom] < bawah) | (data[kolom] > atas)).sum())


sebelum_train = {k: hitung_outlier(X_train, k) for k in outlier_columns}
sebelum_test = {k: hitung_outlier(X_test, k) for k in outlier_columns}

fig, axes = plt.subplots(4, 4, figsize=(10, 9.5))
for index, kolom in enumerate(outlier_columns):
    ax_sebelum = axes[index // 4, index % 4]
    sns.boxplot(y=X_train[kolom], ax=ax_sebelum, color="#a8c6f0")
    ax_sebelum.set_title(kolom + " (sebelum)", fontsize=8)
    ax_sebelum.set_ylabel("")

for kolom, (bawah, atas) in bounds.items():
    X_train[kolom] = X_train[kolom].astype(float).clip(bawah, atas)
    X_test[kolom] = X_test[kolom].astype(float).clip(bawah, atas)

for index, kolom in enumerate(outlier_columns):
    ax_sesudah = axes[2 + index // 4, index % 4]
    sns.boxplot(y=X_train[kolom], ax=ax_sesudah, color="#f0b7a4")
    ax_sesudah.set_title(kolom + " (sesudah)", fontsize=8)
    ax_sesudah.set_ylabel("")
axes[1, 3].axis("off")
axes[3, 3].axis("off")
plt.tight_layout()
plt.savefig(out("03_boxplot_handling_outlier.png"), dpi=150)
plt.close()

hasil_outlier = pd.DataFrame(
    {
        "Batas Bawah": [round(bounds[k][0], 2) for k in outlier_columns],
        "Batas Atas": [round(bounds[k][1], 2) for k in outlier_columns],
        "Outlier Train (sebelum)": [sebelum_train[k] for k in outlier_columns],
        "Outlier Train (sesudah)": [hitung_outlier(X_train, k) for k in outlier_columns],
        "Outlier Test (sebelum)": [sebelum_test[k] for k in outlier_columns],
        "Outlier Test (sesudah)": [hitung_outlier(X_test, k) for k in outlier_columns],
    },
    index=outlier_columns,
)
print(hasil_outlier.to_string())
print("\nJumlah baris training :", X_train.shape[0], "(tidak berkurang)")
print("Jumlah baris testing  :", X_test.shape[0], "(tidak berkurang)")

# ----------------------------------------------------------------------
# 3.7 TRANSFORMASI: Z-SCORE (NUMERIK) + ONE-HOT (KATEGORIKAL)
# ----------------------------------------------------------------------
judul("TRANSFORMASI Z-SCORE DAN ONE-HOT ENCODING")
categorical_columns = X_train.select_dtypes(include=["object"]).columns.tolist()
numeric_columns = X_train.select_dtypes(exclude=["object"]).columns.tolist()
print("Kolom numerik     (%d):" % len(numeric_columns), numeric_columns)
print("Kolom kategorikal (%d):" % len(categorical_columns), categorical_columns)

contoh = ["Age", "MonthlyIncome", "YearsAtCompany"]
sebelum_stat = pd.DataFrame(
    {"mean train": X_train[contoh].mean(), "std train": X_train[contoh].std(ddof=0)}
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_columns),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_columns,
        ),
    ]
)
X_train_transformed = preprocessor.fit_transform(X_train)  # fit hanya di training
X_test_transformed = preprocessor.transform(X_test)  # testing hanya transform
feature_names = preprocessor.get_feature_names_out()
X_train_transformed = pd.DataFrame(
    X_train_transformed, columns=feature_names, index=X_train.index
)
X_test_transformed = pd.DataFrame(
    X_test_transformed, columns=feature_names, index=X_test.index
)
print("\nJumlah fitur setelah transformasi:", X_train_transformed.shape[1])
print("  numerik (Z-Score)  :", len(numeric_columns))
print("  one-hot (dummy)    :", X_train_transformed.shape[1] - len(numeric_columns))

kolom_z = ["numeric__" + c for c in contoh]
print("\nStatistik fitur SEBELUM Z-Score (data training):")
print(sebelum_stat.round(4))
print("\nStatistik fitur SESUDAH Z-Score:")
sesudah_stat = pd.DataFrame(
    {
        "mean train": X_train_transformed[kolom_z].mean().values,
        "std train": X_train_transformed[kolom_z].std(ddof=0).values,
        "mean test": X_test_transformed[kolom_z].mean().values,
        "std test": X_test_transformed[kolom_z].std(ddof=0).values,
    },
    index=contoh,
)
print(sesudah_stat.round(4))
print("\nContoh 3 baris data training hasil transformasi:")
print(X_train_transformed.iloc[:3, :6].round(3))

# ----------------------------------------------------------------------
# 3.8 RESAMPLING (HANYA DATA TRAINING)
# ----------------------------------------------------------------------
judul("RESAMPLING: ROS DAN SMOTE (HANYA DATA TRAINING)")
ros = RandomOverSampler(random_state=RANDOM_STATE)
X_train_ros, y_train_ros = ros.fit_resample(X_train_transformed, y_train)
smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train_transformed, y_train)

print("Distribusi training sebelum resampling:")
print(y_train.value_counts().sort_index())
print("\nDistribusi setelah ROS:")
print(y_train_ros.value_counts().sort_index())
print("\nDistribusi setelah SMOTE:")
print(y_train_smote.value_counts().sort_index())

tabel_resampling = pd.DataFrame(
    {
        "No (0)": [(y_train == 0).sum(), (y_train_ros == 0).sum(), (y_train_smote == 0).sum()],
        "Yes (1)": [(y_train == 1).sum(), (y_train_ros == 1).sum(), (y_train_smote == 1).sum()],
    },
    index=["Baseline", "ROS", "SMOTE"],
)
tabel_resampling["Total"] = tabel_resampling.sum(axis=1)
print("\nRingkasan:")
print(tabel_resampling)

fig, ax = plt.subplots(figsize=(8, 5))
tabel_resampling[["No (0)", "Yes (1)"]].plot(
    kind="bar", ax=ax, color=["#a8c6f0", "#f0b7a4"]
)
for container in ax.containers:
    ax.bar_label(container, fontsize=9)
ax.set_title("Distribusi Kelas Data Training")
ax.set_ylabel("Jumlah sampel")
ax.set_xlabel("")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(out("06_distribusi_kelas_training.png"), dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 3.9 TRAINING DECISION TREE
# ----------------------------------------------------------------------
judul("TRAINING DECISION TREE (3 SKENARIO)")
parameters = {"criterion": "gini", "max_depth": 5, "random_state": RANDOM_STATE}
print("Parameter Decision Tree:", parameters)
models = {
    "Baseline": DecisionTreeClassifier(**parameters),
    "ROS": DecisionTreeClassifier(**parameters),
    "SMOTE": DecisionTreeClassifier(**parameters),
}
data_training = {
    "Baseline": (X_train_transformed, y_train),
    "ROS": (X_train_ros, y_train_ros),
    "SMOTE": (X_train_smote, y_train_smote),
}
for nama, model in models.items():
    X_fit, y_fit = data_training[nama]
    model.fit(X_fit, y_fit)
    print("Model %-8s dilatih dengan %d sampel, kedalaman pohon = %d"
          % (nama, len(y_fit), model.get_depth()))

# ----------------------------------------------------------------------
# 3.10 TESTING DAN EVALUASI
# ----------------------------------------------------------------------
judul("TESTING DAN EVALUASI (KELAS POSITIF = ATTRITION YES)")


def evaluate(nama, model):
    prediction = model.predict(X_test_transformed)
    matrix = confusion_matrix(y_test, prediction)
    tn, fp, fn, tp = matrix.ravel()
    hasil = {
        "Model": nama,
        "Akurasi": accuracy_score(y_test, prediction),
        "Presisi": precision_score(y_test, prediction, pos_label=1),
        "Recall": recall_score(y_test, prediction, pos_label=1),
        "F1-Score": f1_score(y_test, prediction, pos_label=1),
    }
    print("\n----- Hasil Evaluasi:", nama, "-----")
    print("Confusion matrix:  TN = %d | FP = %d | FN = %d | TP = %d" % (tn, fp, fn, tp))
    print("Akurasi   : %.4f" % hasil["Akurasi"])
    print("Presisi   : %.4f" % hasil["Presisi"])
    print("Recall    : %.4f" % hasil["Recall"])
    print("F1-Score  : %.4f" % hasil["F1-Score"])
    print(classification_report(y_test, prediction, target_names=["No (0)", "Yes (1)"]))

    display = ConfusionMatrixDisplay(matrix, display_labels=["No", "Yes"])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix - " + nama)
    plt.tight_layout()
    fig.savefig(out("04_confusion_matrix_" + nama.lower() + ".png"), dpi=150)
    plt.close(fig)
    return hasil, (tn, fp, fn, tp)


hasil_list, matrix_list = [], []
for nama, model in models.items():
    hasil, cm = evaluate(nama, model)
    hasil_list.append(hasil)
    matrix_list.append((nama,) + cm)

results = pd.DataFrame(hasil_list)
tabel_cm = pd.DataFrame(matrix_list, columns=["Skenario", "TN", "FP", "FN", "TP"])

judul("PERBANDINGAN 3 SKENARIO (DATA TESTING)")
print(results.round(4).to_string(index=False))
print("\nConfusion matrix ketiga skenario:")
print(tabel_cm.to_string(index=False))

ax = results.set_index("Model").plot(kind="bar", figsize=(9, 5.5), colormap="coolwarm")
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", fontsize=8)
plt.title("Perbandingan Metrik Evaluasi - 3 Skenario")
plt.ylabel("Nilai")
plt.xlabel("")
plt.xticks(rotation=0)
plt.ylim(0, 1.1)
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False)
plt.tight_layout()
plt.savefig(out("05_perbandingan_metrik.png"), dpi=150)
plt.close()
results.to_csv(out("hasil_perbandingan_model.csv"), index=False)

# Pengecekan overfitting
judul("PENGECEKAN OVERFITTING (TRAINING VS TESTING)")
baris = []
for nama, model in models.items():
    X_fit, y_fit = data_training[nama]
    pred_train = model.predict(X_fit)
    uji = results.set_index("Model").loc[nama]
    baris.append(
        {
            "Skenario": nama,
            "Akurasi Train": accuracy_score(y_fit, pred_train),
            "Akurasi Test": uji["Akurasi"],
            "Recall Train": recall_score(y_fit, pred_train, pos_label=1),
            "Recall Test": uji["Recall"],
            "F1 Train": f1_score(y_fit, pred_train, pos_label=1),
            "F1 Test": uji["F1-Score"],
        }
    )
print(pd.DataFrame(baris).round(4).to_string(index=False))
print("\nCatatan: metrik train untuk ROS/SMOTE dihitung pada data training hasil resampling.")

terbaik = results.loc[results["F1-Score"].idxmax(), "Model"]
print("\nSkenario dengan F1-Score kelas Yes tertinggi:", terbaik)
