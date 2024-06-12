# Object recognition in the wild using Convolutional Neural Networks
### Practical Work 05 – Transfer learning, part 2
#### Étudiants : Rafael Dousse & Emily Baquerizo

## 1. Introduction
L'application créée lors de ce labo est un identificateur de panneau de circulation. 
[Méthodologie]
Pour pouvoir commencer à entraîner notre modèle, nous avons dû de construire une base de données de panneaux. Pour cela, nous avons pris nos propres photos et pour compléter un peu plus notre dataset, nous avons également récupéré quelques photos sur le net.


## 2. Problèmatique
Dû à un manque de variété dans notre dataset, nous nous sommes concentrés sur l'identification des panneaux triangle, rectangle et rond; ces panneaux étants les plus communs sur nos trajets de tous les jours. Ainsi, nous nous retrouvons avec un dataset rassemblant un peu moins de 180 images.
Notre dataset se trouve être plus ou moins équilibré à l'exception des images de panneaux rectangles qui sont légèrement moins nombreuses que les ronds ou triangles. La taille de notre dataset est un peu petit pour une utilsation plus poussée du projet mais pour notre utilisation dans le cadre de notre laboratoire c'est une taille qui peut être acceptable.

## 3. Préparation des données
Notre dataset d'image a subit un pré-traitement pour les préparer pour le model. Pour cela, elles ont tout d'abord été redimensionnée en 224x224 pixels puis normalisée.
Nous avons ensuite procédé à une augmentation des images aléatoires en appliquant des filtres tel qu'un retournement de l'image (horizontal, vertical, ...) ou encore un ajustement du contraste.

[Describe the train, validation and test datasets split.]

Pour le test et la validation, l'utilisation des images augmentée n'est pas idéale puisqu'elles ne reflètent pas les panneaux dans la réalité. Par manque de temps, nous avons tout de même utilisé les images augmentées dans le test et la validation mais dans une situation optimale, cela aurait été évité.

## 4. Création du model

##### a. What hyperparameters did you choose (nb epochs, optimizer, learning rate, ...) ?

##### b. What is the architecture of your final model ? How many trainable parameters does it have?

##### c. How did you perform the transfer learning ? Explain why did you use transfer learning, e.g., what is the advantage of using transfer learning for this problem and why it might help ?


## 5. Résultat

##### a. Provide your plots and confusion matrices

##### b. Provide the f-score you obtain for each of your classes.

##### c. Provide the results you have after evaluating your model on the test set. Comment if the performance on the test set is close to the validation performance. What about the performance of the system in the real world ?

##### d. Present an analysis of the relevance of the trained system using the Class Activation Map methods (grad-cam)

##### e. Provide some of your misclassified images (test set and real-world tests) and comment those errors.

##### f. Based on your results how could you improve your dataset ?

##### g. Observe which classes are confused. Does it surprise you? In your opinion, what can cause those confusions ? What happens when you use your embedded system to recognize objects that don’t belong to any classes in your dataset ? How does your system work if your object is placed in a different background ?


## 6. Conclusion
Après finalisation du projet, nous en sommes venus à la conclusion que nous aurions pu finalement faire une application reconnaissant les formes de manière générale au lieu de se focaliser sur les panneaux.
