# Projet de fin de module — Deep Learning (EMSI Casablanca, 2025-2026)

**Intitule officiel :** Conception, implementation, comparaison et analyse critique de modeles de deep learning pour donnees tabulaires, images et sequences

**Auteur :** [Votre Nom]
**Filiere :** Informatique — EMSI Casablanca
**Module :** Deep Learning — Annee universitaire 2025-2026

---

## 1. Idee generale du projet

Ce repository contient le travail individuel realise pour le projet de fin de module de Deep Learning. Le projet couvre les trois grandes familles d'architectures etudiees dans le module, chacune appliquee a un jeu de donnees reel :

| Partie | Architecture | Theme | Dataset |
|---|---|---|---|
| **I** | MLP | Classification tabulaire | Statistiques de matchs de football (Premier League, football-data.co.uk) |
| **II** | CNN | Classification d'images | CIFAR-10 |
| **III** | RNN / LSTM / GRU / Seq2Seq | Modelisation de sequences | Sequences de resultats de matchs de football (corpus sequentiel reel) |

**Fil conducteur :** les trois parties exploitent, dans la mesure du possible, des donnees liees au football, dans une logique de projet personnel coherent plutot que trois exercices disjoints. Ce choix s'eloigne des datasets suggeres par le cahier des charges pour les Parties I et III (Wine Quality / Breast Cancer / Adult Income pour la Partie I ; IMDb / Tatoeba pour la Partie III). **Ce point est explicitement assume et discute dans le rapport** (`docs/rapport.docx`, section "Choix du dataset" de chaque partie) : il appartient a l'enseignant(e) de valider ou non cette substitution, conformement a la clause du cahier des charges autorisant "tout autre dataset equivalent valide par l'enseignant(e)".

Chaque partie suit la structure demandee par le cahier des charges : etude theorique, implementation PyTorch, etude experimentale, analyse critique, et question de synthese sur le dataset reel utilise. Une question transversale finale relie les trois architectures entre elles.

## 2. Structure du repository

```
.
├── README.md                          <- ce fichier
├── notebooks/                         <- notebooks Jupyter executables (.ipynb)
│   ├── Partie1_MLP.ipynb
│   ├── Partie2_CNN.ipynb
│   └── Partie3_RNN_Seq2Seq.ipynb
├── src/                                <- code source equivalent en scripts .py
│   ├── part1_mlp.py
│   ├── part2_cnn.py
│   └── part3_rnn_seq2seq.py
├── docs/                               <- rapport et documentation
│   └── rapport.docx                   <- rapport scientifique complet (toutes parties + synthese finale)
├── outputs/
│   ├── figures/                        <- courbes, matrices de confusion, feature maps (.png)
│   └── models/                         <- poids des modeles entraines (.pth)
└── requirements.txt                    <- dependances Python
```

### Pourquoi `notebooks/` et `src/` contiennent (presque) le meme code ?

Les notebooks `.ipynb` sont le livrable principal demande par le cahier des charges (executable, avec sorties et visualisations inline). Les fichiers `.py` dans `src/` sont le meme code, sans le formatage notebook, fournis pour faciliter la relecture, le diff, et une execution scriptee (par exemple via `python src/part2_cnn.py`). Les deux versions sont generees a partir de la meme source pour rester synchronisees.

## 3. Comment executer le projet

### Option 1 — Google Colab (recommande)
1. Ouvrir [Google Colab](https://colab.research.google.com)
2. `Fichier > Ouvrir un notebook > GitHub`, coller l'URL de ce repository
3. Choisir le notebook de la partie souhaitee (`notebooks/PartieX_*.ipynb`)
4. Activer un runtime GPU : `Execution > Modifier le type d'execution > GPU`
5. Executer toutes les cellules (`Execution > Tout executer`)

### Option 2 — Environnement local
```bash
git clone <url-du-repo>
cd <nom-du-repo>
pip install -r requirements.txt
jupyter notebook notebooks/Partie1_MLP.ipynb
```

Chaque notebook telecharge ses propres donnees au lancement (CSV football-data.co.uk en Partie I/III, CIFAR-10 via `torchvision` en Partie II) — aucune donnee volumineuse n'est versionnee dans le repository.

## 4. Contenu du rapport (`docs/rapport.docx`)

Le rapport scientifique complet contient, pour chaque partie : introduction, fondements theoriques, methodologie, resultats experimentaux commentes, analyse critique, limites, et la reponse a la question de synthese sur dataset reel. Il se conclut par la discussion transversale finale demandee par le cahier des charges, reliant MLP, CNN et modeles sequentiels.

## 5. Limites connues et points a valider avec l'enseignant(e)

- **Partie I** : dataset tabulaire football plutot que Wine Quality / Breast Cancer / Adult Income.
- **Partie III** : corpus sequentiel football (alphabet de 3 symboles : victoire / nul / defaite) plutot qu'un corpus textuel reel (IMDb, Tatoeba). La tache de modelisation de langage et le Seq2Seq sont implementes selon les memes principes theoriques (factorisation par la regle de chaine, BPTT, teacher forcing, beam search), mais sur un alphabet tres reduit comparé a un vrai corpus linguistique — cette difference est discutee dans le rapport.

## 6. Dependances

Voir `requirements.txt`. Bibliotheques principales : `torch`, `torchvision`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `requests`.
