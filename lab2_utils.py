# gestion des fichiers 
# permet de manipuler des fichiers et des répertoires.
from pickle import FALSE

# pour la manipulation de tableaux
#manipuler facilement des tableaux et faire des calculs mathématiques.
import numpy as np 

#créer et entraîner des réseaux de neurones.
import keras

#fonctions mathématiques classiques (sin, cos, pi, etc.).
import math

#(alias rd) → générer des nombres aléatoires.
import random as rd

#pour tracer des graphiques
import matplotlib.pyplot as plt






# rafraichissement de l'IHM
# augmenter ces valeurs si cela ralentit trop l'apprentissage
error_refresh_rate = 2
contour_refresh_rate = 20




#number : nombre d'exemples à générer (doit être pair)
#xmean : décalage en x du centre des données
#spacing : espacement entre les deux classes
#noise : niveau de bruit aléatoire ajouté aux coordonnées des exemples
def generate_samples_circle(number, xmean = 0, spacing = 1, noise = 0):
    Xc1 = np.array([])
    Xc2 = np.array([])
    sample_number_percat = int(number/2)
    for i in range (sample_number_percat):
        a=rd.random()*2*math.pi
        r=0.45*rd.random()*spacing
        x1 = r*math.sin(a) + noise*(rd.random()-0.5) + xmean
        y1 = r*math.cos(a) + noise*(rd.random()-0.5)
        Xc1 = np.append(Xc1, [x1 ,y1])    

        a=rd.random()*2*math.pi
        r=(0.55+0.45*rd.random())*spacing
        x2 = r*math.sin(a) + noise*(rd.random()-0.5) + xmean
        y2 = r*math.cos(a) + noise*(rd.random()-0.5)
        Xc2 = np.append(Xc2, [x2 ,y2])

    Xc1 = Xc1.reshape(sample_number_percat,2)
    Yc1 = np.array([1, 0])
    Yc1 = np.tile(Yc1,sample_number_percat).reshape(sample_number_percat,2)

    Xc2 = Xc2.reshape(sample_number_percat,2)
    Yc2 = np.array([0, 1])
    Yc2 = np.tile(Yc2,sample_number_percat).reshape(sample_number_percat,2)

    return Xc1, Xc2, Yc1, Yc2




def generate_samples_spiral(number, xmean = 0, spacing = 1, noise = 0):
    Xc1 = np.array([])
    Xc2 = np.array([])
    sample_number_percat = int(number/2)
    for i in range (sample_number_percat):
        a=rd.random()*3*math.pi
        r=a/15*(1+0.4*rd.random())+0.1

        x1 = r*math.sin(a) + 0.1*(rd.random()-0.5)+noise*(rd.random()-0.5)
        y1 = r*math.cos(a) + 0.1*(rd.random()-0.5)+noise*(rd.random()-0.5)
        Xc1 = np.append(Xc1, [x1 ,y1])    

        a=rd.random()*3*math.pi
        r=a/15*(1+0.4*rd.random())+0.1
        x2 = r*math.sin(a+math.pi) + 0.1*(rd.random()-0.5)+noise*(rd.random()-0.5)
        y2 = r*math.cos(a+math.pi) + 0.1*(rd.random()-0.5)+noise*(rd.random()-0.5)
        Xc2 = np.append(Xc2, [x2 ,y2])

    Xc1 = Xc1.reshape(sample_number_percat,2)
    Yc1 = np.array([1, 0])
    Yc1 = np.tile(Yc1,sample_number_percat).reshape(sample_number_percat,2)

    Xc2 = Xc2.reshape(sample_number_percat,2)
    Yc2 = np.array([0, 1])
    Yc2 = np.tile(Yc2,sample_number_percat).reshape(sample_number_percat,2)

    return Xc1, Xc2, Yc1, Yc2

class PlotLosses(keras.callbacks.Callback):
    def __init__(self,nn_model, Xc1train,Xc2train,Xc1valid,Xc2valid):
        
        # body of the constructor
        self.nn_model = nn_model
    
        self.epoch = 0
        self.Xc1train = Xc1train
        self.Xc2train = Xc2train
        self.Xc1valid = Xc1valid
        self.Xc2valid = Xc2valid

        self.xmax = max(np.amax(Xc1train[:,0]),np.amax(Xc2train[:,0]),np.amax(Xc1valid[:,0]),np.amax(Xc2valid[:,0]))
        self.xmin = min(np.amin(Xc1train[:,0]),np.amin(Xc2train[:,0]),np.amin(Xc1valid[:,0]),np.amin(Xc2valid[:,0]))
        self.ymax = max(np.amax(Xc1train[:,1]),np.amax(Xc2train[:,1]),np.amax(Xc1valid[:,1]),np.amax(Xc2valid[:,1]))
        self.ymin = min(np.amin(Xc1train[:,1]),np.amin(Xc2train[:,1]),np.amin(Xc1valid[:,1]),np.amin(Xc2valid[:,1]))

        self.xx = np.linspace(self.xmin, self.xmax, 50)
        self.yy = np.linspace(self.ymin, self.ymax, 50)
        X, Y = np.meshgrid(self.xx, self.yy)

        self.Xtest = np.append( X.reshape(2500,1), Y.reshape(2500,1),axis=1)




    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []

        self.fig = plt.figure("TABLEAU DE BORD 2D",figsize=(16,5))
        self.gs = self.fig.add_gridspec(2,3)
        self.ax1 = self.fig.add_subplot(self.gs[0, 0])
        self.ax2 = self.fig.add_subplot(self.gs[1, 0])
        self.ax3 = self.fig.add_subplot(self.gs[0:2, 1])
        self.ax4 = self.fig.add_subplot(self.gs[0:2, 2])

        self.fig.tight_layout(pad=2)
        self.fig.subplots_adjust(left=0.05)
        # on atend un peu avant le premier affichage
     
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.epoch = self.epoch + 1
        drawFigure = False
        self.i += 1

        if((self.epoch==1) or (self.epoch % error_refresh_rate ==0)):
            self.logs.append(logs)
            self.x.append(self.i)
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            self.acc.append(100*logs.get('accuracy'))
            self.val_acc.append(100*logs.get('val_accuracy'))
            
            self.ax1.cla()
            self.ax1.plot(self.x, self.losses, label="apprentissage : "+str(round(logs.get('loss'),3)))
            self.ax1.plot(self.x, self.val_losses, label="validation : "+str(round(logs.get('val_loss'),3)))
            min, max = self.ax1.get_ylim()
            self.ax1.legend()
            
            self.ax1.grid()
            self.ax1.set_ylabel("Erreur")   

            self.ax2.cla()
            self.ax2.plot(self.x, self.acc, label="apprentissage : "+str(round(100*logs.get('accuracy'),1))+" %")
            self.ax2.plot(self.x, self.val_acc, label="validation : "+str(round(100*logs.get('val_accuracy'),1))+" %")
            
            self.ax2.legend()
            self.ax2.set_xlabel("Itérations")
            self.ax2.set_ylabel("Taux classifications correctes")
            self.ax2.grid()
            drawFigure = True
        
        if((self.epoch==1) or (self.epoch % contour_refresh_rate ==0)):
            Yt = self.nn_model.predict(self.Xtest)
            
            ct3 = self.ax3.contourf(self.xx, self.yy, Yt[:,1].reshape(50,50))

            self.ax4.contourf(self.xx, self.yy, Yt[:,1].reshape(50,50))
            
            self.ax3.title.set_text('Frontière de classification + exemples d\'apprentissage')
            self.ax3.plot(self.Xc1train[:,0],self.Xc1train[:,1], 'o', color='blue')
            self.ax3.plot(self.Xc2train[:,0],self.Xc2train[:,1], 'o', color='yellow')
            
            self.ax3.set_xlim([self.xmin, self.xmax])
            self.ax3.set_ylim([self.ymin, self.ymax])

            self.ax4.title.set_text('Frontière de classification + exemples de validation')
            self.ax4.plot(self.Xc1valid[:,0],self.Xc1valid[:,1], 'P', color='blue')
            self.ax4.plot(self.Xc2valid[:,0],self.Xc2valid[:,1], 'P', color='yellow')
            
            self.ax4.set_xlim([self.xmin, self.xmax])
            self.ax4.set_ylim([self.ymin, self.ymax])
            
            drawFigure = True                       
        
        if(drawFigure):
            plt.draw()
            plt.pause(0.001)

        
        
        
        
        
        
        
        
        
        
        
        
        
# Xc1 (classe 1) :
#  [[ 0.12  0.05]
#  [-0.10  0.15]
#  [ 0.08 -0.12]
#  [-0.05 -0.08]
#  [ 0.03  0.10]]
# Yc1 (labels classe 1) :
#  [[1 0]
#  [1 0]
#  [1 0]
#  [1 0]
#  [1 0]]

# Xc2 (classe 2) :
#  [[ 0.60 -0.20]
#  [-0.55  0.50]
#  [ 0.70  0.05]
#  [ 0.45 -0.55]
#  [ 0.65  0.30]]
# Yc2 (labels classe 2) :
#  [[0 1]
#  [0 1]
#  [0 1]
#  [0 1]
#  [0 1]]