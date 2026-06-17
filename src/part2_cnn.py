# -*- coding: utf-8 -*-
"""
EMSI Casablanca — Projet Deep Learning 2025-2026
Partie II : CNN et vision par ordinateur
Theme    : Classification d'images avec CNN (CIFAR-10)
           Fil conducteur football : un CNN apprend des representations
           hierarchiques, comme un scout video detecte des patterns locaux
           (ballon, posture) avant de comprendre l'action.

Dataset  : CIFAR-10 (10 classes d'images naturelles) - dataset suggere par
           le cahier des charges.

Auteur   : [Votre Nom]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

print("PyTorch version :", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device          :", device)

# ======================================================================
# 1. THEORIE - Pourquoi un MLP est-il peu adapte aux images ?
# ======================================================================
"""
Un MLP applique a une image 32x32x3 aplatit l'image en un vecteur de 3072
valeurs et perd ainsi toute information spatiale : deux pixels voisins ne
sont plus distinguables de deux pixels distants.

Les CNN exploitent trois proprietes fondamentales :
  1. Localite      : un filtre 3x3 capte des patterns locaux (bord, texture).
  2. Partage poids : le meme filtre balaie toute l'image -> beaucoup moins
                      de parametres qu'un MLP equivalent.
  3. Hierarchie     : couche 1 -> bords ; couche 2 -> formes ; couche 3 -> objets.

Analogie football : un scout video ne regarde pas chaque pixel isolement.
Il detecte d'abord des primitives locales (trajectoire du ballon, posture
du joueur), puis les combine pour reconnaitre une action (tir, passe,
dribble).
"""

# ======================================================================
# 2. CALCULS MANUELS DE DIMENSIONS
# ======================================================================


def output_size_conv(input_size, kernel_size, padding, stride):
    """floor((input_size + 2*padding - kernel_size) / stride) + 1"""
    return int((input_size + 2 * padding - kernel_size) / stride) + 1


def output_size_pool(input_size, kernel_size, stride):
    return int((input_size - kernel_size) / stride) + 1


print("\n--- Calculs dimensionnels ---")
print("Apres Conv(3,pad=1,stride=1) :", output_size_conv(32, 3, 1, 1))  # 32
print("Apres MaxPool(2,stride=2)    :", output_size_pool(32, 2, 2))  # 16
print("Apres Conv(3,pad=0,stride=1) :", output_size_conv(32, 3, 0, 1))  # 30

# ======================================================================
# 3. IMPLEMENTATIONS MANUELLES (cross-correlation, max/avg pooling)
# ======================================================================


def manual_cross_correlation_2d(input_tensor, kernel):
    """Correlation croisee 2D sans padding, stride=1."""
    H, W = input_tensor.shape
    kH, kW = kernel.shape
    out_H, out_W = H - kH + 1, W - kW + 1
    output = torch.zeros(out_H, out_W)
    for i in range(out_H):
        for j in range(out_W):
            output[i, j] = (input_tensor[i:i + kH, j:j + kW] * kernel).sum()
    return output


def manual_max_pool_2d(input_tensor, pool_size=2, stride=2):
    H, W = input_tensor.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    output = torch.zeros(out_H, out_W)
    for i in range(out_H):
        for j in range(out_W):
            output[i, j] = input_tensor[i * stride:i * stride + pool_size,
                                         j * stride:j * stride + pool_size].max()
    return output


def manual_avg_pool_2d(input_tensor, pool_size=2, stride=2):
    H, W = input_tensor.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    output = torch.zeros(out_H, out_W)
    for i in range(out_H):
        for j in range(out_W):
            output[i, j] = input_tensor[i * stride:i * stride + pool_size,
                                         j * stride:j * stride + pool_size].mean()
    return output


print("\n--- Verification implementations manuelles vs PyTorch ---")
x = torch.rand(6, 6)
k = torch.rand(3, 3)

manual_cc = manual_cross_correlation_2d(x, k)
torch_cc = F.conv2d(x.unsqueeze(0).unsqueeze(0), k.unsqueeze(0).unsqueeze(0)).squeeze()
print("Cross-corr max diff :", (manual_cc - torch_cc).abs().max().item())

manual_mp = manual_max_pool_2d(x)
torch_mp = F.max_pool2d(x.unsqueeze(0).unsqueeze(0), 2).squeeze()
print("MaxPool   max diff  :", (manual_mp - torch_mp).abs().max().item())

manual_ap = manual_avg_pool_2d(x)
torch_ap = F.avg_pool2d(x.unsqueeze(0).unsqueeze(0), 2).squeeze()
print("AvgPool   max diff  :", (manual_ap - torch_ap).abs().max().item())

# ======================================================================
# 4. CHARGEMENT DES DONNEES - CIFAR-10
# ======================================================================
CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                  "dog", "frog", "horse", "ship", "truck"]

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

full_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

val_size = int(0.2 * len(full_train))
train_size = len(full_train) - val_size
train_set, val_set = random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=2)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

print(f"\nTrain: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")


def show_sample_images(loader, classes, n=8):
    imgs, labels = next(iter(loader))
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    imgs = (imgs[:n] * std + mean).clamp(0, 1)
    fig, axes = plt.subplots(1, n, figsize=(14, 2))
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i].permute(1, 2, 0).numpy())
        ax.set_title(classes[labels[i]], fontsize=8)
        ax.axis("off")
    plt.suptitle("CIFAR-10 - Echantillon d'images")
    plt.tight_layout()
    plt.savefig("../outputs/figures/part2_sample_images.png", dpi=120)
    plt.show()


show_sample_images(train_loader, CIFAR_CLASSES)

# ======================================================================
# 5. MODELES : MLP de reference, LeNet modernise, CNN ameliore
# ======================================================================


class ImageMLP(nn.Module):
    """MLP naif : aplatit l'image -> perd toute info spatiale."""

    def __init__(self, input_dim=3 * 32 * 32, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class LeNetMod(nn.Module):
    """LeNet modernise : Conv-BN-ReLU-Pool x2 + Conv 1x1 + FC."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv1x1 = nn.Conv2d(16, 16, kernel_size=1)
        self.fc1 = nn.Linear(16 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.relu(self.conv1x1(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)


class CNNImproved(nn.Module):
    """CNN plus profond : stride=2, conv 1x1, average pooling final."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2),  # 16x16
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 16x16
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 8x8
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AvgPool2d(2),  # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ======================================================================
# 6. FONCTION D'ENTRAINEMENT GENERIQUE
# ======================================================================


def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, label="model"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    for epoch in range(epochs):
        model.train()
        t_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        scheduler.step()

        model.eval()
        v_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                v_loss += criterion(out, yb).item()
                correct += (out.argmax(1) == yb).sum().item()
                total += yb.size(0)

        history["train_loss"].append(t_loss / len(train_loader))
        history["val_loss"].append(v_loss / len(val_loader))
        history["val_acc"].append(100 * correct / total)

        if (epoch + 1) % 5 == 0:
            print(f"[{label}] Epoch {epoch+1:2d}/{epochs} | "
                  f"Train Loss: {history['train_loss'][-1]:.4f} | "
                  f"Val Loss: {history['val_loss'][-1]:.4f} | "
                  f"Val Acc: {history['val_acc'][-1]:.2f}%")
    return model, history


# ======================================================================
# 7. ENTRAINEMENT ET COMPARAISON DES MODELES
# ======================================================================
EPOCHS = 20

print("\n--- Entrainement MLP ---")
mlp_model, mlp_hist = train_model(ImageMLP(), train_loader, val_loader, EPOCHS, label="MLP")

print("\n--- Entrainement LeNet modifie ---")
lenet_model, lenet_hist = train_model(LeNetMod(), train_loader, val_loader, EPOCHS, label="LeNet")

print("\n--- Entrainement CNN ameliore ---")
cnn_model, cnn_hist = train_model(CNNImproved(), train_loader, val_loader, EPOCHS, label="CNN+")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
models_hist = [("MLP", mlp_hist, "steelblue"), ("LeNet Mod", lenet_hist, "tomato"), ("CNN+", cnn_hist, "seagreen")]
for ax, (name, hist, color) in zip(axes, models_hist):
    ax.plot(hist["train_loss"], label="Train loss", color=color)
    ax.plot(hist["val_loss"], label="Val loss", color=color, linestyle="--")
    ax.set_title(name)
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.3)
plt.suptitle("Comparaison MLP vs CNN - CIFAR-10")
plt.tight_layout()
plt.savefig("../outputs/figures/part2_model_comparison.png", dpi=120)
plt.show()


def test_accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for Xb, yb in loader:
            out = model(Xb.to(device))
            correct += (out.argmax(1) == yb.to(device)).sum().item()
            total += yb.size(0)
    return 100 * correct / total


for name, m in [("MLP", mlp_model), ("LeNet Mod", lenet_model), ("CNN+", cnn_model)]:
    print(f"Test Accuracy [{name}]: {test_accuracy(m, test_loader):.2f}%")

# ======================================================================
# 8. ETUDE EXPERIMENTALE : influence padding / stride / pooling
# ======================================================================
configs = {
    "no_padding": dict(padding=0, stride=1, pool="max"),
    "padding_1": dict(padding=1, stride=1, pool="max"),
    "stride_2": dict(padding=1, stride=2, pool="max"),
    "avg_pool": dict(padding=1, stride=1, pool="avg"),
}


def build_config_cnn(padding, stride, pool):
    pool_layer = nn.MaxPool2d(2) if pool == "max" else nn.AvgPool2d(2)
    s1 = output_size_conv(32, 3, padding, stride)
    s2 = s1 // 2
    s3 = output_size_conv(s2, 3, padding, 1)
    s4 = s3 // 2
    flat_dim = 32 * s4 * s4
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=padding, stride=stride), nn.ReLU(), pool_layer,
        nn.Conv2d(32, 32, 3, padding=padding), nn.ReLU(), pool_layer,
        nn.Flatten(),
        nn.Linear(flat_dim, 128), nn.ReLU(),
        nn.Linear(128, 10),
    )


print("\n--- Etude des configurations architecturales ---")
config_results = {}
for cfg_name, cfg in configs.items():
    try:
        m = build_config_cnn(**cfg)
        m, h = train_model(m, train_loader, val_loader, epochs=10, label=cfg_name)
        acc = test_accuracy(m, test_loader)
        config_results[cfg_name] = acc
        print(f"  {cfg_name:15s} -> Test Acc: {acc:.2f}%")
    except Exception as e:
        print(f"  {cfg_name:15s} -> ERREUR : {e}")

# ======================================================================
# 9. VISUALISATION DES FEATURE MAPS
# ======================================================================


def visualize_feature_maps(model, loader, layer_name="conv1", n_filters=6):
    """Visualise les activations de la premiere couche convolutionnelle."""
    model.eval()
    imgs, labels = next(iter(loader))
    img = imgs[0:1].to(device)

    activations = {}

    def hook_fn(module, input, output):
        activations["feat"] = output.detach().cpu()

    hook = model.conv1.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(img)
    hook.remove()

    feat = activations["feat"][0]
    n = min(n_filters, feat.shape[0])
    fig, axes = plt.subplots(1, n + 1, figsize=(14, 3))

    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    orig = (imgs[0] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    axes[0].imshow(orig)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i in range(n):
        axes[i + 1].imshow(feat[i].numpy(), cmap="viridis")
        axes[i + 1].set_title(f"Filtre {i+1}")
        axes[i + 1].axis("off")

    plt.suptitle(f"Feature maps - {layer_name} (LeNet Mod)")
    plt.tight_layout()
    plt.savefig("../outputs/figures/part2_feature_maps.png", dpi=120)
    plt.show()


visualize_feature_maps(lenet_model, test_loader)

# ======================================================================
# 10. EVALUATION FINALE
# ======================================================================


def full_evaluation(model, loader, classes):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            out = model(Xb.to(device)).argmax(1).cpu()
            all_preds.extend(out.numpy())
            all_labels.extend(yb.numpy())
    print(classification_report(all_labels, all_preds, target_names=classes))
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Matrice de confusion - CNN ameliore (CIFAR-10)")
    plt.ylabel("Vrai")
    plt.xlabel("Predit")
    plt.tight_layout()
    plt.savefig("../outputs/figures/part2_confusion_matrix.png", dpi=120)
    plt.show()


print("\n--- Evaluation finale CNN ameliore ---")
full_evaluation(cnn_model, test_loader, CIFAR_CLASSES)

# ======================================================================
# 11. SAUVEGARDE DU MEILLEUR MODELE
# ======================================================================
torch.save(cnn_model.state_dict(), "../outputs/models/best_cnn_cifar10.pth")
print("\nModele CNN sauvegarde : outputs/models/best_cnn_cifar10.pth")

reload_cnn = CNNImproved()
reload_cnn.load_state_dict(torch.load("../outputs/models/best_cnn_cifar10.pth"))
reload_cnn.eval()
print("Modele CNN recharge avec succes.")

# ======================================================================
# 12. QUESTION DE SYNTHESE - PARTIE II (reponse complete dans le rapport)
# ======================================================================
"""
Voir docs/rapport.docx, section "Partie II - Question de synthese", pour la
reponse structuree et argumentee a la question :

"Pourquoi un CNN est-il plus pertinent qu'un MLP pour une tache de
classification d'images sur un dataset reel, et comment les choix de
padding, stride, pooling et profondeur influencent-ils reellement les
performances du modele ?"
"""
