# les bibliothèques
import numpy as np
#import tensorflow as tf

N=32#taille de la grille spatiale
h=1/(N+1)#pas de discrétisation

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
T = np.zeros((N,N))

# la grille des positions adimensionnées entre 0 et 1
l =np.linspace(0,1,N)
X,Y=np.meshgrid(l,l)

#conductivité du matériau
k=np.ones((N,N))
for i in range(N):
    for j in range(N):
        if (X[i,j]-params[1])**2+(Y[i,j]-params[2])**2<params[3]**2:#si on est dans la zone d'inclusion
            k[i,j]=np.exp(params[0])#alors on a la connductivité kappa (on stocke log(kappa) dans les params)
        else:
            k[i,j]=1#conductivité du matériau de base


#calcul de la source de chaleur sur la grille spatiale
q =params[6]*np.exp(-((X-params[7])**2+(Y-params[8])**2)/(2*params[5]**2))#source gaussienne 

#fonction de moyenne harmonique
def harmonic_mean(a: float, b: float) -> float:
    return 2.0*a*b/(a+b)

A=np.zeros((N+1,N+1))
b=np.zeros((N+1)**2)
for i in range(N+1):
    for j in range(N+1):
        p=i-1+(j-1)*N
        if i==0 or i==N+1 or j==0 or j==N+1:
            A[i,j]=k[i,j]/h**2
            b[p]=0
        else:
            kw=harmonic_mean(k[i,j],k[i,j-1])
            ke=harmonic_mean(k[i,j],k[i,j+1])
            kn=harmonic_mean(k[i,j],k[i-1,j])
            ks=harmonic_mean(k[i,j],k[i+1,j])
            A[p,p]=ke+kw+kn+ks
            A[p,p+1]=-ke
            A[p,p-1]=-kw
            A[p,p+N]=-kn
            A[p,p-N]=-ks
            b[p]=q[i,j]*h**2