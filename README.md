pred_experimentations
=====================
#Visualiserl es résultats
Dans le dossier "resultats" tu trouveras l'ensemble des matrices telles que :

|           | clust Predit               | ...
|ref Clust  |(precision, rappel, fmesure)| ...
| ...       |                            | ...

Il y a aussi un graphe dans le fichier image line-stripes.png qui montre l'évolution de la FMesure globale en fonction du nombre groupe (pour la coupe de la CAH).

Le main est dans le fichier compare.py. Pour le lancer il faut utiliser la commande python qui est situé dans le package de l'application Orange
    /Applications/Orange.app/Contents/MacOS/python

Il y a des dépendances à installer avec ce programme python par exemple : pyGoogleChart et Imagin pour afficher les groupes dans la CAH.

