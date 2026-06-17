# -*- coding: utf-8 -*-
"""
EMSI Casablanca — Projet Deep Learning 2025-2026
Partie I : MLP et ingenierie PyTorch
Theme    : Classification supervisee sur donnees tabulaires de football
           (statistiques de match -> resultat Victoire Domicile / Nul / Victoire Exterieur)

Dataset  : football-data.co.uk, Premier League (E0), saison 2023-2024
           Variante choisie a la place des jeux de donnees suggeres dans le
           cahier des charges (Wine Quality / Breast Cancer / Adult Income).
           Justification detaillee dans le rapport (section "Choix du dataset").

Auteur   : [Votre Nom]
"""

import io
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

print("PyTorch version :", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device          :", device)

# ======================================================================
# 1. THEORIE (resumee ici, developpee dans le rapport)
# ======================================================================
"""
nn.Module        : classe de base de tout modele PyTorch. Encapsule les
                    parametres et definit forward().
Parametres        : tenseurs appris (poids, biais), exposes via
                    .parameters() / .named_parameters().
Gradient          : calcule automatiquement par autograd lors de
                    loss.backward(), stocke dans .grad de chaque parametre.
state_dict()      : dictionnaire {nom_du_parametre: tenseur}, utilise pour
                    sauvegarder / recharger un modele.
device            : CPU ou GPU (cuda) ; modele et donnees doivent etre sur
                    le meme device pour que les operations fonctionnent.
Propagation avant : calcul des sorties a partir des entrees (forward()).
Retropropagation  : calcul des gradients de la loss par rapport aux
                    parametres, via la regle de la chaine (backward()).
"""

# ======================================================================
# 2. CHARGEMENT ET PREPARATION DES DONNEES
# ======================================================================
URL = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
df = pd.read_csv(io.StringIO(requests.get(URL).text))

print("Dimensions brutes :", df.shape)
print("Colonnes :", df.columns.tolist())

# Variables explicatives : statistiques de jeu (tirs, tirs cadres, fautes,
# corners, cartons jaunes) pour l'equipe a domicile (H) et a l'exterieur (A)
FEATURES = ["HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY"]
TARGET = "FTR"  # H = victoire domicile, D = nul, A = victoire exterieur

# Nettoyage : on retire les lignes incompletes
df_clean = df[FEATURES + [TARGET]].dropna()
print(f"Lignes apres nettoyage : {len(df_clean)} / {len(df)}")

X = df_clean[FEATURES].values
y_raw = df_clean[TARGET].values

# Encodage de la cible (label encoding : H/D/A -> 0/1/2)
le = LabelEncoder()
y = le.fit_transform(y_raw)
print("Classes :", le.classes_)
print("Distribution :", np.bincount(y))

# Normalisation des features (StandardScaler : moyenne 0, variance 1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Separation train (70%) / validation (15%) / test (15%), stratifiee pour
# preserver la proportion de chaque classe
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")


def make_loader(X, y, batch_size=32, shuffle=True):
    ds = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


train_loader = make_loader(X_train, y_train)
val_loader = make_loader(X_val, y_val, shuffle=False)
test_loader = make_loader(X_test, y_test, shuffle=False)

input_dim = X_train.shape[1]
num_classes = len(le.classes_)

# ======================================================================
# 3. DEUX VERSIONS DU MLP
# ======================================================================

# --- Version A : nn.Sequential -----------------------------------------
net_seq = nn.Sequential(
    nn.Linear(input_dim, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, num_classes),
)


# --- Version B : classe personnalisee -----------------------------------
class FootballMLP(nn.Module):
    """MLP a deux couches cachees avec dropout, pour classifier
    le resultat d'un match (H/D/A) a partir de statistiques de jeu."""

    def __init__(self, input_dim, hidden1=64, hidden2=32, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.out(x)


model = FootballMLP(input_dim=input_dim, num_classes=num_classes)
print(model)

# ======================================================================
# 4. INSPECTION DES PARAMETRES
# ======================================================================
print("\n=== named_parameters() ===")
for name, param in model.named_parameters():
    print(f"{name:20s} | shape: {tuple(param.shape)} | requires_grad: {param.requires_grad}")

print("\n=== state_dict() keys ===")
for key in model.state_dict():
    print(key, "->", tuple(model.state_dict()[key].shape))

# ======================================================================
# 5. STRATEGIES D'INITIALISATION (Gaussienne / Constante / Xavier)
# ======================================================================


def init_gaussian(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


def init_constant(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 1.0)
        nn.init.zeros_(m.bias)


def init_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


strategies = {"Gaussian": init_gaussian, "Constant": init_constant, "Xavier": init_xavier}


def train_model(init_fn, epochs=50):
    m = FootballMLP(input_dim=input_dim, num_classes=num_classes)
    m.apply(init_fn)
    m = m.to(device)

    optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        m.train()
        t_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(m(Xb), yb)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        m.eval()
        v_loss = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                v_loss += criterion(m(Xb.to(device)), yb.to(device)).item()

        train_losses.append(t_loss / len(train_loader))
        val_losses.append(v_loss / len(val_loader))

    return m, train_losses, val_losses


print("\n--- Entrainement comparatif des 3 strategies d'initialisation ---")
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
results = {}
best_model, best_val = None, float("inf")

for i, (name, fn) in enumerate(strategies.items()):
    m, tl, vl = train_model(fn)
    results[name] = (tl, vl)
    axes[i].plot(tl, label="Train")
    axes[i].plot(vl, label="Val")
    axes[i].set_title(f"Init : {name}")
    axes[i].set_xlabel("Epoch")
    axes[i].legend()
    print(f"  {name:10s} -> val loss finale = {vl[-1]:.4f}")
    if min(vl) < best_val:
        best_val = min(vl)
        best_model = m

plt.suptitle("Comparaison des strategies d'initialisation - Football MLP")
plt.tight_layout()
plt.savefig("../outputs/figures/part1_init_comparison.png", dpi=120)
plt.show()

# ======================================================================
# 6. SAUVEGARDE ET RECHARGEMENT DU MEILLEUR MODELE
# ======================================================================
torch.save(best_model.state_dict(), "../outputs/models/best_football_mlp.pth")
print("\nModele sauvegarde : outputs/models/best_football_mlp.pth")

loaded_model = FootballMLP(input_dim=input_dim, num_classes=num_classes)
loaded_model.load_state_dict(torch.load("../outputs/models/best_football_mlp.pth"))
loaded_model.eval()
loaded_model.to(device)
print("Modele recharge avec succes.")

# ======================================================================
# 7. EVALUATION FINALE
# ======================================================================


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            out = model(Xb.to(device)).argmax(dim=1)
            preds.extend(out.cpu().numpy())
            labels.extend(yb.numpy())
    return preds, labels


preds, labels = evaluate(loaded_model, test_loader)

print("\n--- Classification report (test set) ---")
print(classification_report(labels, preds, target_names=le.classes_))

cm = confusion_matrix(labels, preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Matrice de confusion - Prediction de resultat (MLP)")
plt.ylabel("Vrai")
plt.xlabel("Predit")
plt.tight_layout()
plt.savefig("../outputs/figures/part1_confusion_matrix.png", dpi=120)
plt.show()

# ======================================================================
# 8. QUESTION DE SYNTHESE - PARTIE I (reponse complete dans le rapport)
# ======================================================================
"""
Voir docs/rapport.docx, section "Partie I - Question de synthese", pour la
reponse structuree et argumentee a la question :

"Dans quelle mesure un MLP bien parametre constitue-t-il une solution
pertinente pour la classification tabulaire sur un dataset reel, et
quelles sont ses principales limites au regard de la structure statistique
des donnees etudiees ?"
"""
