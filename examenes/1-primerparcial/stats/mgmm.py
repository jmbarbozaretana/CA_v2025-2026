# (C) 2021-2023 Pablo Alvarado
# Escuela de Ingeniería Electrónica
# Tecnológico de Costa Rica
# Manejo de dos mezclas de gaussianas para modelado de poblaciones en exámenes

import numpy as np
from scipy.stats import norm
from scipy.special import erf
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import RandomizedSearchCV
from sklearn.mixture import GaussianMixture
import pickle


import time

class MGMM:
    ''' 
    Clase diseñada para facilitar el trabajo con una
    mezcla de mezclas de gaussianas
        
    Esta clase recibe en su constructor un GaussianMixture de sklearn
    y un valor de cuántas de las gaussianas superiores se consideran
    como la población de personas que dominan la materia.

    El resto de métodos ayudan a manejar las mezclas de gaussianas de
    cada población.
    '''

    # Pesos originales de cada gaussiana
    orig_w_ = []
    
    # Pesos corresponden a las probabilidades a-priori de cada gaussiana
    # pero ya re-normalizados para las dos poblaciones
    w_ = []
    
    # Probabilidades a-priori de las dos poblaciones
    p_ = [0.5,0.5]
    
    # Las medias ordenadas de menor a mayor
    mu_ = []
    
    # Las desviaciones estándar
    sd_ = []
    
    # Cuántas gaussianas en total
    n_components = 0
    
    # Cuántas gaussianas para la población que no-domina (0) o sí domina (1)
    n_ = [0,0]
    
    # Índices de las gaussianas de no-domina (Fila 0) y domina (Fila 1)
    i_ = [[],[]]
    
    def __init__(self, gmm, dominan):
        '''
        :param gmm: GaussianMixture con todas las gaussianas
        :param dominan: Cuántas gaussianas deben asignarse a la población que domina materia
        '''
        
        # Necesitamos ordenar primero las gaussianas por sus medias, de menor a mayor
        idx=np.argsort(gmm.means_.flatten())
        self.mu_=gmm.means_.flatten()[idx]
        self.sd_=np.sqrt(gmm.covariances_.flatten()[idx])
        self.orig_w_=gmm.weights_.flatten()[idx]
        self.w_=self.orig_w_.copy()
        
        # Renormalicemos los pesos / probabilidades a priori
        self.n_components = gmm.n_components
        self.n_[0]=gmm.n_components-dominan
        self.n_[1]=dominan
       
        self.i_[0]=np.arange(0,self.n_[0])
        self.i_[1]=np.arange(self.n_[0],self.n_components)

        
        self.p_[0] = np.sum(self.orig_w_[self.i_[0]])
        self.w_[self.i_[0]]=self.orig_w_[self.i_[0]]/self.p_[0]
                
        self.p_[1] = np.sum(self.orig_w_[self.i_[1]])
        self.w_[self.i_[1]]=self.orig_w_[self.i_[1]]/self.p_[1]
    
    def likelihood(self, pob, rho ):
        '''
        Calcule la verosimilitud ponderada por la probabilidad a priori
        para la población pob (0=no-domina, 1=domina)
        :param pob: 0 para población no-domina, 1 para población domina
        :param rho: valores en puntos para los cuales calcular la probabilidad
        '''             
        i=self.i_[pob]

        p=self.w_[i[0]]*norm(self.mu_[i[0]],self.sd_[i[0]]).pdf(rho)
        for n in i[1:]:
            p+=self.w_[n]*norm(self.mu_[n],self.sd_[n]).pdf(rho)
        
        return p.flatten()

    def posterior(self, pob, rho ):
        wl = [0,0]
        wl[0] = self.likelihood(0,rho)*self.p_[0]
        wl[1] = self.likelihood(1,rho)*self.p_[1]

        return wl[pob]/(wl[0]+wl[1])

    def optimal_bayesian_threshold(self):
        '''
        En caso de que el número de componentes sea dos, es posible
        encontrar un óptimo analíticamente
        '''
        if self.n_components != 2:
            raise Exception("Umbral óptimo solo es posible con 2 gaussianas")

        # Abreviaciones para solución (nd=no dominan, d=dominan)
        mnd=self.mu_[0]
        snd=self.sd_[0]
        vnd=snd**2
        pnd=self.p_[0]

        md=self.mu_[1]
        sd=self.sd_[1]
        vd=sd**2
        pd=self.p_[1]

        # Abreviaciones para solución
        alpha=2*np.log((pnd*sd)/(pd*snd))
        vdiff=vd - vnd
        delta=sd*snd*np.sqrt(alpha*vdiff + (md-mnd)**2)

        # El umbral óptimo (¿puede ser necesario cambiar el +delta a -delta?)
        rho=(mnd*vd- md*vnd + delta)/vdiff

        return rho

    
    def Phi(self,z):
        '''
        This function is the integral of a gaussian of mean 0 and 
        stdev 1.
        '''
        return 0.5*(1 + erf(z/np.sqrt(2)))

    
    def errors(self,thresholds):
        '''
        Calcule los errores de clasificación para los umbrales dados
        Devuelve los errores de la población "no-domina" y la población "domina"
        por separado, para cada uno de los umbrales dados
        '''

        i=self.i_[0]
        f0=self.w_[i[0]]*self.Phi((-thresholds+self.mu_[i[0]])/self.sd_[i[0]])
        for n in i[1:]:
            f0+=self.w_[n]*self.Phi((-thresholds+self.mu_[n])/self.sd_[n])
        f0=f0*self.p_[0]

        i=self.i_[1]
        f1=self.w_[i[0]]*self.Phi((thresholds-self.mu_[i[0]])/self.sd_[i[0]])
        for n in i[1:]:
            f1+=self.w_[n]*self.Phi((thresholds-self.mu_[n])/self.sd_[n])
        f1=f1*self.p_[1]
        
        return f0,f1

    def cumulative(self,thresholds):
        '''
        Calcule la distribución cumulativa total
        '''

        f1=self.orig_w_[0]*self.Phi((thresholds-self.mu_[0])/self.sd_[0])
        #print("w[0]={0} µ[0]={1} σ[0]={2}".format(self.orig_w_[0],
        #                                          self.mu_[0],
        #                                          self.sd_[0]));

        for n in np.arange(1,self.n_components):
            f1+=self.orig_w_[n]*self.Phi((thresholds-self.mu_[n])/self.sd_[n])
            #print("w[{3}]={0} µ[{3}]={1} σ[{3}]={2}".format(self.orig_w_[n],
            #                                                self.mu_[n],
            #                                                self.sd_[n],
            #                                                n));
        # fuerce a que la probabilidad máxima sea 1
        f1=f1/f1[-1]
        
        return f1


###############################################################################
#  Bandwidth and GMM
###############################################################################

    
def progress(string):
    print(string,"            ",end='\r')
    
def bandwidth(data,
              bw_range=np.logspace(0,1,1000,base=10.0),
              cv_range=np.arange(8,13),
              cv_rep=100,
              n_iter=20,
              prog=progress,
              load_file=None):
    '''
    Bandwidth estimation is a difficult and relatively unreliable 
    task.
    
    We use here cross-validation as recommended, but we repeat the
    estimation several times, for a range of folding values of the 
    cross-validation.

    :param data:     data-set for with the bandwidth is estimated
    :param bw_range: list of all bandwidths to be evaluated
    :param cv_range: list of cross-validation values to be tested
    :param cv_rep:   how many times a cross-validation experiment 
                     should be repeated
    :param n_iter:   number of iterations used in the RandomizedSearchCV

    '''

    tic=time.time()


    # Try to load all the bandwidth data from a file if it exists
    if load_file is not None:
        try:
            infile=open(load_file,'rb')
            saved=pickle.load(infile)
            infile.close()

            ok = ( ( np.min(bw_range).item() == saved['min_bw_range']) and
                   ( np.max(bw_range).item() == saved['max_bw_range']) and
                   ( cv_range == saved['cv_range'] ).all() and
                   ( cv_rep == saved['cv_rep'] ) and
                   ( n_iter == saved['n_iter'] ) )

            if ok:
                print("Recovering data from file '{0}'".format(load_file))
                toc=time.time()-tic
                return saved['bw'],toc
            else:
                print("File '{0}' exists but something changed.  Estimating again...".format(load_file))
        except Exception as err:
                print("File '{0}': {1}".format(load_file,err))

    # Set up the bandwidth range to be evaluated
    params = {'bandwidth': bw_range}

    # Este rango especifica los rangos de correlación cruzada a usar
    # (8 a 12 p.ej) y cuántos experimentos para cada uno de ellos
    # usar.
    
    
    # Create a list with all cv values to be tested, with the number
    # of desired repetitions
    exp_range=np.repeat(cv_range,cv_rep)
    num_exp=len(exp_range) # Total number of experiments
    
    bw=[] # All evaluated bandwidth values
    shuffledData=data.copy()
    
    for i,cv in enumerate(exp_range):
        grid = RandomizedSearchCV(KernelDensity(),
                                  params,
                                  cv=cv,
                                  n_iter=n_iter,
                                  n_jobs=-1) # Use cv-fold cross-validation
        np.random.shuffle(shuffledData)
        grid.fit(shuffledData)
        bw.append(grid.best_estimator_.bandwidth)
        
        prog("Exp {2}/{3}:  cv={1}  bandwidth: {0}".format(grid.best_estimator_.bandwidth,
                                                           cv,
                                                           i+1,
                                                           num_exp))


    if load_file is not None:
        saved={'min_bw_range': np.min(bw_range).item(),
               'max_bw_range': np.max(bw_range).item(),
               'cv_range': cv_range,
               'cv_rep': cv_rep,
               'n_iter': n_iter,
               'bw': bw}
        print("\nSaving file '{0}'".format(load_file))
        outfile=open(load_file,'wb')
        pickle.dump(saved,outfile)
        outfile.close()
        
    toc=time.time()-tic

    return bw,toc

def fitGMM(samples,
           n_components_range=range(1,8),
           quiet=False):
    
    lowest_bic = np.infty
    best_n_components = -1
    bic = []

    cv_type = 'spherical'
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gm = GaussianMixture(n_components=n_components,covariance_type=cv_type)
        gm.fit(samples)
        bic.append(gm.bic(samples))
        #bic.append(gm.aic(samples))
        if not quiet:
            print("  bic con {0} componentes = {1}".format(n_components,bic[-1]))
            
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gm
            best_n_components = n_components

    if not quiet:
        print("Best components:",best_n_components) 
    
    return best_gmm

def dominantGaussian(gmm,rank=1):
    idx=np.argsort(gmm.weights_.flatten())[::-1].item(rank-1)
    return gmm.means_.item(idx),np.sqrt(gmm.covariances_.item(idx))


def estimateThresholds(pts,
                       total_pts,
                       bw,
                       n_components_range,
                       thres10=0.1,
                       thres100=0.99):
    
    kde = KernelDensity(bandwidth=bw).fit(pts)
    samples = kde.sample(50000) # Usemos MUCHOS datos (muestra aleatoria)
    
    best_gmm=fitGMM(samples,
                    n_components_range=n_components_range,
                    quiet=True)

    best_n_components=best_gmm.n_components

    num_gauss_domina=int(best_gmm.n_components/2)

    mgmm=MGMM(best_gmm,num_gauss_domina)

    xplot = np.linspace(0,total_pts,10001)
    
    pnD=mgmm.posterior(0,xplot)
    pD=mgmm.posterior(1,xplot)

    idx=np.argmin(np.abs(pnD-pD))
    th70=xplot[idx]

    t=np.linspace(0,total_pts,total_pts*10000+1)[:,np.newaxis]

    cumulative=mgmm.cumulative(t)
    th10=t[np.argmax(cumulative>=thres10),0]
    th100=t[np.argmax(cumulative>=thres100),0]

    
    return th10,th70,th100,best_n_components

def thresholdStats(pts,
                   total_pts,
                   bws, # list of bandwidths to be tested
                   n_components_range,
                   reps=500, # Repetitions for each bandwidth in bws
                   thres10=0.1, # Threshold for "quit course"
                   thres100=0.99, # Threshold for best grade
                   load_file=None, # File to be loaded / saved
                   force=False, # Force computation (inhibit loading file)
                   prog=progress):
    '''
    :param pts: complete dataset of obtained points
    :param total_pts: total number of points in the exam
    :param bws: list of bandwidhts to be tested
    :param n_components_range: range for the number of GMM components
    :param reps: how many times the experiment is repeated for a given bandwidth
    :param load_file: filename to cache the results
    :param force: force computation (inhibit loading data file)
    :param prog: progress object to display the progress
    :returns: a list of lines, where each line holds the following information
              th10,th70,th100,n_components,bandwidth
              and the total time used
    '''

    tic=time.time()

    # Try to load all the bandwidth data from a file if it exists
    if not force and load_file is not None:
        try:
            infile=open(load_file,'rb')
            saved=pickle.load(infile)
            infile.close()

            ok = ( (len(pts) == saved['len_pts']) and
                   (total_pts == saved['total_pts']) and
                   (len(bws) == saved['len_bws']) and
                   (len(n_components_range) == saved['len_n_components']) and
                   (reps == saved['reps']) )

            if ok:
                print("Recovering data from file '{0}'".format(load_file))
                toc=time.time()-tic
                return saved['thresholds'],toc
            else:
                print("File '{0}' exists but something changed.  Estimating again...".format(load_file))
        except Exception as err:
                print("File '{0}': {1}".format(load_file,err))

    total = len(bws)*reps
    firstTime=True
    expnum=1
    for bw in bws:
        for _ in range(reps):
            line=list(estimateThresholds(pts,total_pts,bw,n_components_range,thres10,thres100))
            line.append(bw)
            line=np.array(line)

            if firstTime:
                ths=line
                firstTime=False
            else:
                ths=np.vstack((ths,line))
            prog("{0}/{1} {2}      ".format(expnum,total,line))
            expnum+=1
                
    if load_file is not None:

        saved={'len_pts':len(pts),
               'total_pts':total_pts,
               'len_bws':len(bws),
               'len_n_components':len(n_components_range),
               'reps':reps,
               'thresholds':ths}
        
        print("\nSaving file '{0}'".format(load_file))
        outfile=open(load_file,'wb')
        pickle.dump(saved,outfile)
        outfile.close()
        
    toc=time.time()-tic

    return ths,toc
