# -*- coding: utf-8 -*-
"""
EMSI Casablanca — Projet Deep Learning 2025-2026
Partie III : RNN, LSTM, GRU et Seq2Seq
Theme    : Modelisation de sequences de resultats de matchs de football
           (corpus sequentiel reel derive de football-data.co.uk), utilise
           comme "langage" simplifie a 3 tokens (H/D/A) pour illustrer les
           concepts de modele de langage, BPTT, et Seq2Seq.

Limite assumee : ce n'est pas un corpus textuel au sens strict (cf. cahier
des charges : IMDb, Tatoeba/fra-eng...). Le choix et ses implications sont
discutes explicitement dans le rapport (section "Choix du dataset").

Auteur   : [Votre Nom]
"""

import io
import math
import random
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

print("PyTorch version :", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device          :", device)

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ======================================================================
# 1. THEORIE - Modeles de langage, perplexite, BPTT
# ======================================================================
"""
Objectif probabiliste d'un modele de langage
----------------------------------------------
Un modele de langage apprend P(x_1, x_2, ..., x_T), la probabilite jointe
d'une sequence de tokens. Par la regle de la chaine :

    P(x_1,...,x_T) = P(x_1) * P(x_2|x_1) * P(x_3|x_1,x_2) * ... * P(x_T|x_1,...,x_{T-1})

Un RNN approxime chaque terme P(x_t | x_1,...,x_{t-1}) a l'aide d'un etat
cache h_t qui resume le contexte passe.

Perplexite
----------
PP = exp(loss_cross_entropy_moyenne). Elle mesure le degre de "surprise"
moyen du modele face a la sequence reelle : plus elle est basse, mieux le
modele predit le token suivant. Une perplexite de 1 correspond a une
prediction parfaite ; une perplexite egale au nombre de classes correspond
a une prediction aleatoire uniforme.

BPTT (Backpropagation Through Time)
------------------------------------
Le gradient de la loss est retropropage a travers chaque pas de temps de la
sequence. Sur des sequences longues, ceci peut provoquer une explosion ou
une disparition du gradient. Le gradient clipping (ecretage de la norme du
gradient a une valeur maximale) est une parade standard a l'explosion.
"""

# ======================================================================
# 2. CONSTRUCTION DU CORPUS SEQUENTIEL (football)
# ======================================================================
SEASONS = ["2021", "2122", "2223", "2324"]
all_matches = []

for season in SEASONS:
    url = f"https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
    try:
        resp = requests.get(url, timeout=10)
        df = pd.read_csv(io.StringIO(resp.text))
        df = df[["Date", "HomeTeam", "AwayTeam", "FTR"]].dropna()
        all_matches.append(df)
    except Exception as e:
        print(f"Saison {season} ignoree ({e})")

matches = pd.concat(all_matches, ignore_index=True)
print(f"Total de matchs charges : {len(matches)}")

# Vocabulaire : tokens speciaux + resultats possibles
VOCAB = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "H": 3, "D": 4, "A": 5}
INV_VOCAB = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)


def build_team_sequences(df, min_len=6, max_len=12):
    """Pour chaque equipe, construit la sequence chronologique de ses
    resultats (du point de vue de cette equipe : W/D/L recode en H/D/A
    pour rester dans le meme alphabet que la tache de prediction)."""
    sequences = []
    teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
    for team in teams:
        team_matches = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)]
        seq = []
        for _, row in team_matches.iterrows():
            if row["HomeTeam"] == team:
                seq.append(row["FTR"])  # H = victoire de l'equipe (a domicile)
            else:
                # on inverse la perspective : si FTR=A, l'equipe (a l'exterieur) gagne -> "H" (victoire)
                inv = {"H": "A", "A": "H", "D": "D"}
                seq.append(inv[row["FTR"]])
        if min_len <= len(seq) <= max_len:
            sequences.append(seq)
        elif len(seq) > max_len:
            for i in range(0, len(seq) - max_len, max_len // 2):
                sequences.append(seq[i:i + max_len])
    return sequences


sequences = build_team_sequences(matches)
print(f"Nombre de sequences construites : {len(sequences)}")
print("Exemple de sequence :", sequences[0])


def encode_sequence(seq, add_sos_eos=True):
    tokens = [VOCAB[c] for c in seq]
    if add_sos_eos:
        tokens = [VOCAB["<sos>"]] + tokens + [VOCAB["<eos>"]]
    return tokens


def pad_sequence(tokens, max_len, pad_value=VOCAB["<pad>"]):
    if len(tokens) >= max_len:
        return tokens[:max_len]
    return tokens + [pad_value] * (max_len - len(tokens))


MAX_LEN = 14
encoded = [pad_sequence(encode_sequence(s), MAX_LEN) for s in sequences]
encoded = np.array(encoded)
print("Shape du corpus encode :", encoded.shape)

split_point = int(0.85 * len(encoded))
train_idx = np.arange(len(encoded))[:split_point]
val_idx = np.arange(len(encoded))[split_point:]


class SequenceDataset(Dataset):
    def __init__(self, data, indices):
        self.data = data[indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        # tache "modele de langage" : predire le token suivant
        x = torch.LongTensor(seq[:-1])
        y = torch.LongTensor(seq[1:])
        return x, y


train_ds = SequenceDataset(encoded, train_idx)
val_ds = SequenceDataset(encoded, val_idx)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# ======================================================================
# 3. MODELES : RNN simple, LSTM, GRU
# ======================================================================


class SequenceLanguageModel(nn.Module):
    """Modele de langage generique : embedding -> RNN/LSTM/GRU -> projection."""

    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32, cell_type="rnn"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=VOCAB["<pad>"])
        cell_map = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
        self.rnn = cell_map[cell_type](embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.cell_type = cell_type

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        out, hidden = self.rnn(emb, hidden)
        logits = self.fc(out)
        return logits, hidden


def train_lm(model, train_loader, val_loader, epochs=15, lr=1e-3, clip=1.0, label="model"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=VOCAB["<pad>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "val_ppl": [], "grad_norms": []}
    for epoch in range(epochs):
        model.train()
        t_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, _ = model(Xb)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), yb.reshape(-1))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            history["grad_norms"].append(grad_norm.item())
            optimizer.step()
            t_loss += loss.item()

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits, _ = model(Xb)
                v_loss += criterion(logits.reshape(-1, VOCAB_SIZE), yb.reshape(-1)).item()

        avg_train = t_loss / len(train_loader)
        avg_val = v_loss / len(val_loader)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_ppl"].append(math.exp(avg_val))

        if (epoch + 1) % 5 == 0:
            print(f"[{label}] Epoch {epoch+1:2d}/{epochs} | "
                  f"Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | "
                  f"Val Perplexity: {history['val_ppl'][-1]:.3f}")
    return model, history


print("\n--- Entrainement comparatif RNN / LSTM / GRU ---")
cell_results = {}
for cell_type in ["rnn", "lstm", "gru"]:
    m = SequenceLanguageModel(VOCAB_SIZE, cell_type=cell_type)
    m, hist = train_lm(m, train_loader, val_loader, epochs=15, label=cell_type.upper())
    cell_results[cell_type] = hist

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for cell_type, hist in cell_results.items():
    axes[0].plot(hist["val_loss"], label=cell_type.upper())
    axes[1].plot(hist["val_ppl"], label=cell_type.upper())
axes[0].set_title("Val Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[1].set_title("Val Perplexity")
axes[1].set_xlabel("Epoch")
axes[1].legend()
plt.suptitle("Comparaison RNN vs LSTM vs GRU - sequences de resultats")
plt.tight_layout()
plt.savefig("../outputs/figures/part3_rnn_lstm_gru_comparison.png", dpi=120)
plt.show()

print("\n--- Perplexite finale (val) ---")
for cell_type, hist in cell_results.items():
    print(f"  {cell_type.upper():5s} -> {hist['val_ppl'][-1]:.3f}")

# ======================================================================
# 4. EFFET DU GRADIENT CLIPPING (illustration)
# ======================================================================
print("\n--- Effet du gradient clipping (LSTM, sans vs avec clipping) ---")


def train_lm_clip_study(clip_value, epochs=10):
    m = SequenceLanguageModel(VOCAB_SIZE, cell_type="lstm").to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=VOCAB["<pad>"])
    optimizer = torch.optim.Adam(m.parameters(), lr=1e-2)  # lr volontairement eleve
    norms = []
    for epoch in range(epochs):
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, _ = m(Xb)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), yb.reshape(-1))
            loss.backward()
            total_norm = torch.norm(torch.stack(
                [p.grad.norm() for p in m.parameters() if p.grad is not None]))
            norms.append(total_norm.item())
            if clip_value is not None:
                torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=clip_value)
            optimizer.step()
    return norms


norms_no_clip = train_lm_clip_study(clip_value=None)
norms_clip = train_lm_clip_study(clip_value=1.0)

plt.figure(figsize=(8, 4))
plt.plot(norms_no_clip, label="Sans clipping", alpha=0.7)
plt.plot(norms_clip, label="Avec clipping (max_norm=1.0)", alpha=0.7)
plt.title("Norme du gradient au cours de l'entrainement (LSTM)")
plt.xlabel("Iteration")
plt.ylabel("Norme du gradient")
plt.legend()
plt.tight_layout()
plt.savefig("../outputs/figures/part3_gradient_clipping.png", dpi=120)
plt.show()
print(f"Norme moyenne sans clipping : {np.mean(norms_no_clip):.3f} (max={np.max(norms_no_clip):.3f})")
print(f"Norme moyenne avec clipping : {np.mean(norms_clip):.3f} (max={np.max(norms_clip):.3f})")

# ======================================================================
# 5. SEQ2SEQ : encodeur-decodeur pour predire la "suite" d'une serie
# ======================================================================
"""
Tache Seq2Seq choisie : etant donnee une sequence des 6 premiers resultats
d'une equipe, predire la sequence des resultats suivants (longueur fixe).
C'est l'equivalent, dans notre corpus simplifie, d'une tache de
traduction/generation sequence-a-sequence.
"""

SRC_LEN = 6
TGT_LEN = 6


def build_seq2seq_pairs(sequences):
    pairs = []
    for seq in sequences:
        if len(seq) >= SRC_LEN + TGT_LEN:
            src = seq[:SRC_LEN]
            tgt = seq[SRC_LEN:SRC_LEN + TGT_LEN]
            pairs.append((src, tgt))
    return pairs


pairs = build_seq2seq_pairs(sequences)
print(f"\nNombre de paires Seq2Seq construites : {len(pairs)}")


class Seq2SeqDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_ids = torch.LongTensor([VOCAB[c] for c in src])
        tgt_ids = torch.LongTensor([VOCAB["<sos>"]] + [VOCAB[c] for c in tgt] + [VOCAB["<eos>"]])
        return src_ids, tgt_ids


split = int(0.85 * len(pairs))
seq2seq_train = Seq2SeqDataset(pairs[:split])
seq2seq_val = Seq2SeqDataset(pairs[split:])
seq2seq_train_loader = DataLoader(seq2seq_train, batch_size=16, shuffle=True)
seq2seq_val_loader = DataLoader(seq2seq_val, batch_size=16, shuffle=False)


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=VOCAB["<pad>"])
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        emb = self.embedding(x)
        outputs, hidden = self.gru(emb)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=VOCAB["<pad>"])
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        emb = self.embedding(x)
        out, hidden = self.gru(emb, hidden)
        logits = self.fc(out)
        return logits, hidden


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_dim, hidden_dim)
        self.decoder = Decoder(vocab_size, embed_dim, hidden_dim)
        self.vocab_size = vocab_size

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.shape
        _, hidden = self.encoder(src)

        outputs = torch.zeros(batch_size, tgt_len, self.vocab_size, device=src.device)
        dec_input = tgt[:, 0].unsqueeze(1)  # <sos>

        for t in range(1, tgt_len):
            logits, hidden = self.decoder(dec_input, hidden)
            outputs[:, t, :] = logits.squeeze(1)
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(2)
            dec_input = tgt[:, t].unsqueeze(1) if use_teacher_forcing else top1
        return outputs


def train_seq2seq(model, train_loader, val_loader, epochs=30, lr=1e-3):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=VOCAB["<pad>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        t_loss = 0
        tf_ratio = max(0.5 * (1 - epoch / epochs), 0.0)  # teacher forcing decroissant
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            outputs = model(src, tgt, teacher_forcing_ratio=tf_ratio)
            loss = criterion(outputs[:, 1:, :].reshape(-1, VOCAB_SIZE), tgt[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss += loss.item()

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                outputs = model(src, tgt, teacher_forcing_ratio=0.0)
                v_loss += criterion(outputs[:, 1:, :].reshape(-1, VOCAB_SIZE), tgt[:, 1:].reshape(-1)).item()

        history["train_loss"].append(t_loss / len(train_loader))
        history["val_loss"].append(v_loss / len(val_loader))
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}/{epochs} | Train Loss: {history['train_loss'][-1]:.4f} "
                  f"| Val Loss: {history['val_loss'][-1]:.4f}")
    return model, history


print("\n--- Entrainement Seq2Seq (encodeur-decodeur GRU) ---")
seq2seq_model = Seq2Seq(VOCAB_SIZE)
seq2seq_model, s2s_hist = train_seq2seq(seq2seq_model, seq2seq_train_loader, seq2seq_val_loader)

plt.figure(figsize=(7, 4))
plt.plot(s2s_hist["train_loss"], label="Train")
plt.plot(s2s_hist["val_loss"], label="Val")
plt.title("Seq2Seq - Courbe d'apprentissage")
plt.xlabel("Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("../outputs/figures/part3_seq2seq_training.png", dpi=120)
plt.show()

# ======================================================================
# 6. STRATEGIES DE DECODAGE : glouton et beam search
# ======================================================================


def greedy_decode(model, src, max_len=TGT_LEN + 2):
    model.eval()
    with torch.no_grad():
        src = src.unsqueeze(0).to(device)
        _, hidden = model.encoder(src)
        dec_input = torch.LongTensor([[VOCAB["<sos>"]]]).to(device)
        result = []
        for _ in range(max_len):
            logits, hidden = model.decoder(dec_input, hidden)
            top1 = logits.argmax(2)
            token = top1.item()
            if token == VOCAB["<eos>"]:
                break
            result.append(token)
            dec_input = top1
    return [INV_VOCAB[t] for t in result]


def beam_search_decode(model, src, beam_width=3, max_len=TGT_LEN + 2):
    model.eval()
    with torch.no_grad():
        src = src.unsqueeze(0).to(device)
        _, hidden = model.encoder(src)

        # chaque candidat : (sequence_de_tokens, score_log_prob, hidden_state)
        beams = [([VOCAB["<sos>"]], 0.0, hidden)]

        for _ in range(max_len):
            new_beams = []
            for seq_tokens, score, h in beams:
                if seq_tokens[-1] == VOCAB["<eos>"]:
                    new_beams.append((seq_tokens, score, h))
                    continue
                dec_input = torch.LongTensor([[seq_tokens[-1]]]).to(device)
                logits, h_new = model.decoder(dec_input, h)
                log_probs = F.log_softmax(logits.squeeze(1), dim=-1)
                top_log_probs, top_tokens = log_probs.topk(beam_width, dim=-1)
                for k in range(beam_width):
                    tok = top_tokens[0, k].item()
                    new_score = score + top_log_probs[0, k].item()
                    new_beams.append((seq_tokens + [tok], new_score, h_new))

            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]

            if all(b[0][-1] == VOCAB["<eos>"] for b in beams):
                break

        best_seq = beams[0][0]
        result = [t for t in best_seq[1:] if t not in (VOCAB["<eos>"], VOCAB["<pad>"])]
    return [INV_VOCAB[t] for t in result]


print("\n--- Comparaison decodage glouton vs beam search ---")
n_examples = 5
for i in range(n_examples):
    src, tgt = seq2seq_val[i]
    true_tgt = [INV_VOCAB[t] for t in tgt.tolist() if t not in (VOCAB["<sos>"], VOCAB["<eos>"], VOCAB["<pad>"])]
    greedy_result = greedy_decode(seq2seq_model, src)
    beam_result = beam_search_decode(seq2seq_model, src, beam_width=3)
    src_readable = [INV_VOCAB[t] for t in src.tolist()]
    print(f"Source     : {src_readable}")
    print(f"  Vrai     : {true_tgt}")
    print(f"  Glouton  : {greedy_result}")
    print(f"  Beam(3)  : {beam_result}\n")

# ======================================================================
# 7. EVALUATION : PERPLEXITE DU MODELE SEQ2SEQ
# ======================================================================


def evaluate_seq2seq_perplexity(model, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=VOCAB["<pad>"])
    total_loss, n = 0, 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            outputs = model(src, tgt, teacher_forcing_ratio=0.0)
            loss = criterion(outputs[:, 1:, :].reshape(-1, VOCAB_SIZE), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
            n += 1
    avg_loss = total_loss / n
    return avg_loss, math.exp(avg_loss)


val_loss, val_ppl = evaluate_seq2seq_perplexity(seq2seq_model, seq2seq_val_loader)
print(f"\nSeq2Seq - Val loss: {val_loss:.4f} | Val perplexity: {val_ppl:.3f}")


def sequence_accuracy(model, loader, decode_fn=greedy_decode):
    correct, total = 0, 0
    for i in range(len(loader.dataset)):
        src, tgt = loader.dataset[i]
        true_tgt = [INV_VOCAB[t] for t in tgt.tolist() if t not in (VOCAB["<sos>"], VOCAB["<eos>"], VOCAB["<pad>"])]
        pred = decode_fn(model, src)
        for a, b in zip(true_tgt, pred):
            total += 1
            correct += int(a == b)
    return 100 * correct / total if total else 0.0


greedy_acc = sequence_accuracy(seq2seq_model, seq2seq_val_loader, greedy_decode)
beam_acc = sequence_accuracy(seq2seq_model, seq2seq_val_loader,
                              lambda m, s: beam_search_decode(m, s, beam_width=3))
print(f"Exactitude token-a-token (glouton)  : {greedy_acc:.2f}%")
print(f"Exactitude token-a-token (beam = 3) : {beam_acc:.2f}%")

# ======================================================================
# 8. SAUVEGARDE DU MODELE SEQ2SEQ
# ======================================================================
torch.save(seq2seq_model.state_dict(), "../outputs/models/best_seq2seq_football.pth")
print("\nModele Seq2Seq sauvegarde : outputs/models/best_seq2seq_football.pth")

# ======================================================================
# 9. QUESTION DE SYNTHESE - PARTIE III (reponse complete dans le rapport)
# ======================================================================
"""
Voir docs/rapport.docx, section "Partie III - Question de synthese", pour la
reponse structuree et argumentee a la question :

"Dans quelle mesure les architectures recurrentes permettent-elles de
modeliser efficacement une sequence reelle, et comment justifier le
passage d'un RNN simple vers un LSTM/GRU puis vers un schema
encodeur-decodeur pour une tache de generation ou de traduction ?"
"""
