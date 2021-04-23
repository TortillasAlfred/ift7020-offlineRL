# Projet IFT-7020
## Apprentissage par renforcement hors ligne pour problèmes linéaires avec nombres entiers

## Utilisation

```shell
# Installer les librairies:
pip install -r requirements.txt

# Préparer le jeu de données: 
python core/prepare_rl_training.py 

# Entraîner les modèles: 
python core/rl.py

# Tester les modèles: 
python core/rl_testing.py

# Visualiser les résultats (figures et CSVs): 
python core/format_results.py 
```

Le code du projet est basé sur l'implémentation simplifié du papier de Gasse et al. (2019)
Exact combinatorial optimization with graph convolutional neural networks, que l'on peut retrouver à l'adresse suivante: https://github.com/ds4dm/ecole/blob/master/examples/branching-imitation.ipynb
