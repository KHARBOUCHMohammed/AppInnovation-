import spacy
import os
import networkx as nx
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


nlp = spacy.load("fr_core_news_md")

chemins_corpus = ["corpus_asimov_leaderboard/lca", "corpus_asimov_leaderboard/paf"]

books = [
    (list(range(0, 19)), "corpus_asimov_leaderboard/paf"),
    (list(range(0, 18)), "corpus_asimov_leaderboard/lca"),
]

occurrences_personnages = {}


graphe = nx.Graph()

def extraire_entites_personnages(texte):
    doc = nlp(texte)
    entites_personnages = [
        {"nom": doc[start:end].text.strip(), "debut": start, "fin": end}
        for ent in doc.ents 
        if ent.label_ == "PER" and len(ent.text) > 1 and "'" not in ent.text and not any(token.pos_ == "VERB" for token in ent)
        for start, end in [(ent.start, ent.end)]
    ]
    return entites_personnages

def detecter_co_occurrences(entites_personnages):
    co_occurrences = []
    for i in range(len(entites_personnages) - 1):
        for j in range(i + 1, len(entites_personnages)):
            entite1 = entites_personnages[i]
            entite2 = entites_personnages[j]
            

            if entite1["nom"] != entite2["nom"]:
                distance = abs(entite1["fin"] - entite2["debut"])
                if 0 < distance <= 25:
                    co_occurrences.append((entite1["nom"], entite2["nom"]))
    return co_occurrences


for chemin_corpus in chemins_corpus:
    for nom_fichier in os.listdir(chemin_corpus):
        if nom_fichier.endswith(".txt.preprocessed"):
            chemin_fichier = os.path.join(chemin_corpus, nom_fichier)
            with open(chemin_fichier, "r", encoding="utf-8") as fichier:
                texte = fichier.read()
                entites_personnages = extraire_entites_personnages(texte)
                

                for entite in entites_personnages:
                    personnage = entite["nom"]
                    if personnage not in graphe.nodes:
                        graphe.add_node(personnage, names=personnage)

                relations = [
                    (entites_personnages[i]["nom"], entites_personnages[j]["nom"])
                    for i in range(len(entites_personnages) - 1)
                    for j in range(i + 1, len(entites_personnages))
                ]


                for relation in relations:
                    personnage1, personnage2 = relation
                    poids = graphe.get_edge_data(personnage1, personnage2, default={"weight": 0})["weight"] + 1
                    graphe.add_edge(personnage1, personnage2, weight=poids)

                co_occurrences = detecter_co_occurrences(entites_personnages)
                
                if co_occurrences:
                    occurrences_personnages[nom_fichier] = co_occurrences

print("Co-occurrences des personnages :")
for nom_fichier, co_occurrences in occurrences_personnages.items():
    print(f"{nom_fichier}: {co_occurrences}")

nx.write_graphml(graphe, "graphe.graphml")


df_dict = {"ID": [], "graphml": []}

for doc_id, co_occurrences in occurrences_personnages.items():
    df_dict["ID"].append(doc_id)
    graphml = "".join(nx.generate_graphml(graphe))
    df_dict["graphml"].append(graphml)


for chapters, book_code in books:
    for chapter in chapters:
        G = nx.Graph()
        entites_personnages = extraire_entites_personnages(texte)
        df_dict["ID"].append("{}{}".format(book_code, chapter))
        graphml = "".join(nx.generate_graphml(G))
        df_dict["graphml"].append(graphml)


df = pd.DataFrame(df_dict)
df.set_index("ID", inplace=True)


df_filtered = df[df.index.str.startswith(("paf", "lca"))]


nom_fichier_csv = "subm.csv"
df_filtered.to_csv(nom_fichier_csv)

print(f"Données exportées en {nom_fichier_csv}")
