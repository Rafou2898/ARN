# Practical Work 3
#### Auteurs : Dousse Rafael & Baquerizo Emily

## Introduction

Ce travail pratique a pour but de mettre en pratique les connaissances acquises durant nos précédent laboratoires afin d'implémenter des MLPs (Multi-Layer Perceptrons) afin de résoudre des problèmes de classification.  Dans notre cas, nous utilisons la base de données `EEG_mouse_data` qui contient des données d'EEG de souris c'est a dire des données prises sur le cerveau de souris et qui représentent le cycle de sommeil de ces dernières.

Pour pouvoir réaliser ce travail, nous avons utilisé la librairie keras qui est une librairie open-source qui permet de créer des réseaux de neurones de manière simple et rapide. Finalement, nous vérifions les données de notre modèle en faisant une validation croisée et en affichant les matrices de confusions et les F1-scores.

## Exercice 1 :  Classification de 2 classes

1. Préparation des données

Pour commencer la première partie de ce travail, nous avons commencé par importer les données de la base de données `EEG_mouse_data` et nous avons effectué un prétraitement des données en utilisant la colonne des états et en la transformant en deux classes. On a regroupé les états n-rem et rem ensemble en un état asleep qu on a donné la valeur 1 et l'état awake qu'on a donné la valeur de -1. Ensuite, nous avons normalisé les données grâce a la fonction `StandardScaler` de sklearn.

2. Création de modèle

Pour le modèle nous avons utilisé un MLP avec 1 couche caché de 4 neuronnes et une sortie de 1 neuronne. Nous avons utilisé la fonction d'activation tanh comme pour la couche cachée ainsi que pour la couche de sortie. Ce n'est pas un choix original mais elle a été moticé par le fait que nous avons classifié nos valeurs dans un interval de [-1,1]. Enfin, nous avons utilisé comme learning rate 0.01 et momentum 0.9.

3. Entraînement et résultats

Nous avons fait un entrainement sur 50 epochs et avons décider de split nos données en 3 k-folds pour la validation croisée. Le graphique qui suit nous montre que la perte de nos medèles diminue au fur et à mesure des epochs et ne semble pas diverger en remontant ce qui montre qu'on a pas d'overfitting. Par contre, le modèle ne semble pas converger non plus. Plus d'epochs ou un changement de learning rate pourraient améliorer les résultats. Un learning rate plus petit pourrait permettre de converger plus rapidement et d'avoir moir de fluctuation dans la perte.

On a pas une perte très élevée mais elle reste assez haute avec un minimum de 0.36 et cela pourrait être amélioré.

![alt text](image.png)

La matrice de confusion et la moyenne des F1-scores que nous obtenons et qui est de : 0.855 indique ce que nous avons pu voir avec le graphique de la perte. Un score de 0.85 n'est pas mauvais mais il pourrait être amélioré.

![alt text](image-1.png)

## Exercice 2 : Classification de 3 classes

Pour ce 2ème modèle nous avons fait la même chose que pour le premier modèle mais en ajoutant une classe supplémentaire. 

1. Préparation des données

Nous avons à nouveau importé les données de la base de données `EEG_mouse_data` et effectué un prétraitement des données en utilisant la colonne des états et en la transformant en trois classes. Nous avons regroupé les états n-rem, rem et awake en trois classes différentes. Ensuite, nous avons normalisé les données grâce à la fonction `StandardScaler` de sklearn. Les valeurs que nous avons attribué à nos classes sont les suivantes : n-rem = 0, rem = 1 et awake = 2.
