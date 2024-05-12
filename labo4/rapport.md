# Practical work 4
## Auteurs :  Dousse Rafael & Baquerizo Emily


## Question 1
 
 Le code fournis dans les trois premier notebook utilise les même optimiseur et la même fonction de perte.
 L'optimiseur utilisé est RMSprop() et la fonction de perte est categorical_crossentropy.
Les paramètres que l'on utilise pour le modèe et que nous modifions nous sont le batch_size et le nombre d'époques.
 RMSprop() a ses porpres paramètres déjà initialisé que nous aurions pu modifié mais nous ne l'avons pas fait. Les paramètres de RMSprop() que nous avons trouvé en ligne sont les suivants:
``` python
keras.optimizers.RMSprop(
    learning_rate=0.001,
    rho=0.9,
    momentum=0.0,
    epsilon=1e-07,
    centered=False,
    weight_decay=None,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    loss_scale_factor=None,
    gradient_accumulation_steps=None,
    name="rmsprop",
    **kwargs
)
```

Nous avons aussi trouvé en ligne l'équation pour la fonction de perte categorical_crossentropy:

$$\begin{equation}
-\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} 1_{y_i \in C_c} \log p_{\text{model}}(y_i \in C_c)
\end{equation}$$

Enfin, nous avons remarqué que pour le notebook que nous avons du compléter pour le CNN_pneumonia, l'optimiseur et la fonction de perte sont différents. C'est `Adam()` qui est utilisé pour l'optimiseur et la fonction de perte est `binary_crossentropy` ce qui fait sens vu que nous avons que deux classes. Voici son équation:

$$\begin{equation}
 H_p(q) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \cdot \log(p(y_i)) + (1 - y_i) \cdot \log(1 - p(y_i)) \right]
  \end{equation}$$

  Et voici les paramètres de Adam() que nous avons trouvé en ligne:

  ``` python
  keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    weight_decay=None,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    loss_scale_factor=None,
    gradient_accumulation_steps=None,
    name="adam",
    **kwargs
)
  ```

### Question 3 

Ce n'est pas forcément le cas. Avec un MLP chaque pixel va être connecté à chaque neurone de la couche suivante ce qui peux nous donner un grand nombre de poids. Alors que pour un CNN, on va avoir des filtres qui vont être appliqués sur l'image et qui peuvent avoir la même taille à chaque fois ce qui va réduire le nombre de poids. On peut le remarquer avec les notebook fournis. Le résumé du modèle utilisé dans le fichier `MLP_from_raw_data` nous indique que le nombre total de paramètre est de 7'960 alors qu'on a que de couche de 10 neuronnes. Alors que pour le modèle de convolution utilisé dans `CNN`  utilise beaucoup plus de couche et plus de neuronnes mais le nombre total de paramètre est de 1'890.

### Question 4
Pour ce modèle de convolution, nous avons du compléter le code fournis dans le notebook et suivre les instructions données dans une image pour reproduire l'architecture du modèle. 
Voici le code du modèle que nous avons fait:

``` python
conv_1 = Conv2D(8, (3, 3), padding='same', activation='relu', name='conv_1')(input)
max_pooling_1 = MaxPooling2D(pool_size=(2, 2), name='max_pooling_1')(conv_1)

conv_2 = Conv2D(16, (3, 3), padding='same', activation='relu', name='conv_2')(max_pooling_1)
max_pooling_2 = MaxPooling2D(pool_size=(2, 2), name='max_pooling_2')(conv_2)

conv_3 = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv_3')(max_pooling_2)    
max_pooling_3 = MaxPooling2D(pool_size=(2, 2), name='max_pooling_3')(conv_3)

conv_4 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv_4')(max_pooling_3)    
max_pooling_4 = MaxPooling2D(pool_size=(2, 2), name='max_pooling_4')(conv_4)

conv_5 = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv_5')(max_pooling_4)    
max_pooling_5 = MaxPooling2D(pool_size=(2, 2), name='max_pooling_5')(conv_5)

flatten_7 = Flatten(name='flatten_7')(max_pooling_5)

dense_21 = Dense(32, activation='relu', name='dense_21')(flatten_7)
dense_22 = Dense(16, activation='relu', name='dense_22')(dense_21)

cnn_output = layers.Dense(1, activation='sigmoid')(dense_22)
```

 Comme dit dans la question 1, l'optimiseur utilisé est Adam() et la loss est binary_crossentropy qui fonctionne bien ensemble et qui est bien pour classifé deux classes. Nous avons entrainé notre modèle avec 10 époques. Nous avons testé plusieurs epoch entre 5 8 10 12 et 14 et le paramètre de 10 est celui ou on a eu les meilleurs résultats. Voici le graph pour le loss et l'accuracy pour le modèle:


 ![alt text](image-1.png)
 ![alt text](image-2.png)

Chaque fois que nous faisions tourner le modèle, nous obtenions des résultats plus tot similaire avec la validation qui fait des sauts.

Voici les matrices de confusions pour la validation et le test:
 ![alt text](image-3.png)
 ![alt text](image-4.png)

Nous avons obtenu les résultats suivants pour le modèle:

*Validation F1 score: 1.0000 <br>
Validation accuracy: 1.0000 <br>
Test F1 score: 0.8505 <br>
Test accuracy: 0.7821 <br>*


On a des très bon résultats pour la validation ou il classifie tout correctement. Pour le test par contre on a des résultats plus bas ou il a tendance a classifier plus de faux-positif. En fait, le dataset de la validation est assez pauvre avec pas beaucoup de données et il est possible que le modèle ait appris par coeur les données. Ce qui fait que pour le test, il a peut être plus de mal à bien classifier surtout qu'on peut remarquer le datatest de la validation a un nombre similaire de pneumonie et de normal alors que le dataset de test a plus d'images de pneumonie que de normal. Nous n'avons pas réussi a trouver de bonne manière de faire un dataset de validation qui soit plus représentatif du dataset de test. Il nous faudrait plus de données pour pouvoir faire un dataset de validation plus représentatif.