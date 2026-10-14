# les bibliothèques
import numpy as np
#import tensorflow as tf

# les paramètres
log_conductivity_ratio = np.log(0.3) #logarithme du rapport de conductivité entre l'inclusion et le matériau de base
xc,yc=0,0 #centre de l'inclusion
r=0.1 #rayon de l'inclusion
sigma = 0.08 #largeur de la source gaussienne
A=1 #amplitude de la source
xs,ys=0,0 # centre de la source

# le vecteur de paramètres
params = np.array([log_conductivity_ratio,xc,yc,r,sigma,A,xs,ys])

# la grille de sortie des températures
T = np.zeros((32, 32))

# la grille des positions adimensionnées entre 0 et 1
l =np.linspace(0,1,32)
X,Y=np.meshgrid(l,l)

#conductivité du matériau
k=np.ones((32,32))
for i in range(32):
    for j in range(32):
        if (X[i,j]-params[1])**2+(Y[i,j]-params[2])**2<params[3]**2:#si on est dans la zone d'inclusion
            k[i,j]=np.exp(params[0])#alors on a la connductivité kappa (on stocke log(kappa) dans les params)
        else:
            k[i,j]=1#conductivité du matériau de base


#calcul de la source de chaleur sur la grille spatiale
q =params[6]*np.exp(-((X-params[7])**2+(Y-params[8])**2)/(2*params[5]**2))#source gaussienne 