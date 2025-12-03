import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

import lab2_utils

# rafraichissement de l'IHM
# augmenter ces valeurs si cela ralentit trop l'apprentissage
lab2_utils.error_refresh_rate = 50 # toutes les 50 itérations 
lab2_utils.contour_refresh_rate = 50 # toutes les 50 itérations 

# Generation des exemples d'apprentissage
train_samples_number = 200

Xc1train, Xc2train, Yc1train, Yc2train = lab2_utils.generate_samples_spiral( train_samples_number )

Xtrain = np.append(Xc1train, Xc2train,axis=0)
Ytrain = np.append(Yc1train, Yc2train,axis=0)

# Generation des exemples de test
validation_samples_number = 100

Xc1valid, Xc2valid, Yc1valid, Yc2valid = lab2_utils.generate_samples_spiral( validation_samples_number )

Xvalid = np.append(Xc1valid, Xc2valid,axis=0)
Yvalid = np.append(Yc1valid, Yc2valid,axis=0)

# construction d'un réseau de neurones complètement connecté par empilement de couches avec KERAS





model = Sequential()
model.add(Dense(50, input_dim=2, activation='relu'))
model.add(Dense(25, input_dim=2, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.summary() # affichage d'un résumé du RN construit dans le terminal

# compilation du RN et choix de sa méthode d'apprentissage
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# déclaration d'un callback pour l'affichage
plot_losses = lab2_utils.PlotLosses(model, Xc1train, Xc2train, Xc1valid, Xc2valid)

# entrainement du RN
history = model.fit(Xtrain, Ytrain,
          batch_size=32, 
          epochs=1000,
          validation_data = (Xvalid, Yvalid),
          verbose=0,callbacks=[plot_losses]
          )

input("Appuyez sur ENTER pour terminer")
