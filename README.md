# Inspection visuelle industrielle avec PatchCore

Application Streamlit de détection et de localisation d’anomalies sur
trois catégories du jeu de données MVTec AD :

- bottle ;
- cable ;
- tile.

## Méthode

Le projet utilise PatchCore avec un Wide ResNet-50-2 préentraîné.
Les poids du réseau restent figés. Une mémoire de référence distincte
est construite pour chaque catégorie.

## Résultats

| Catégorie | AUROC image | F1-score | AUROC pixel |
|---|---:|---:|---:|
| Bottle | 100 % | 96,77 % | 98,73 % |
| Cable | 98,53 % | 93,48 % | 98,06 % |
| Tile | 100 % | 97,62 % | 95,44 % |

## Exécution locale

```bash
pip install -r requirements.txt
streamlit run app.py
