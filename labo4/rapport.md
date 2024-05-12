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
*Train a CNN for the chest x-ray pneumonia recognition. In order to do so, complete the
code to reproduce the architecture plotted in the notebook. Present the confusion matrix,
accuracy and F1-score of the validation and test datasets and discuss your results.*

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
 