import random
import numpy as np
import math
from numpy.random import seed
from numpy.random import rand
from Environment import *
from tqdm import tqdm

    

    
class UCBT_AW: ##Algorithm 1
    def __init__(self,delta,T,seed,Environment):
        print('UCBT-ASW')
        np.random.seed(seed)
        self.Env=Environment
        self.r=np.zeros(T,float)
        self.r_Exp=np.zeros(T,float)
        k=0
        self.r_Exp[0],self.r[0]=self.Env.observe(k,0)
        t_=0
        if T>1:
            for t in tqdm(range(1,T)):

                for i in range(math.floor(math.log2(t-t_))+1):  
                    w=2**i
                    s=max(t-w,t_)
                    sum_r=np.sum(self.r[s:t])
                    n=t-s
                    mu=sum_r/n
                    ucb=mu+math.sqrt(12*math.log(T)/n)

                    if ucb<1-delta:
                        k=k+1
                        t_=t
                        break

                self.r_Exp[t],self.r[t]=self.Env.observe(k,t)

    def rewards(self):
        return self.r_Exp  
        

class AUCBT_AW: ##Algorithm 2
    def __init__(self,T,seed,Environment):
        print('AUCBT-ASW')
        
        np.random.seed(seed)
        self.Env=Environment
        self.r=np.zeros(T,float)
        self.r_Exp=np.zeros(T,float)
        if T==1:
            k=0
            self.r_Exp[0],self.r[0]=self.Env.observe(k,0)
        else:
            k=0
            H=math.ceil(math.sqrt(T))
            B=math.ceil(math.log2(H))
            alpha=min(1,math.sqrt(B*math.log(B)/((math.e-1)*math.ceil(T/H))))
            w=np.ones(B)
            p=np.zeros(B)
            for i in tqdm(range(math.ceil(T/H))):
                if k!=0: 
                    k=k+1
                self.r_Exp[i*H],self.r[i*H]=self.Env.observe(k,0)
                
                p=(1-alpha)*w/w.sum()+alpha/B
                j=np.random.choice(B,1,p=p)
                delta=(1/2)**(j)
                t_=i*H

                for t in range(i*H+1,min(H*(i+1),T)):


                    for l in range(math.floor(math.log2(t-t_))+1):  
                        win=2**l
                        s=max(t-win,t_)
                        sum_r=np.sum(self.r[s:t])
                        n=t-s
                        mu=sum_r/n
                        ucb=mu+math.sqrt(12*math.log(H)/n)

                        if ucb<1-delta:
                            k=k+1
                            t_=t
                            break
                    self.r_Exp[t],self.r[t]=self.Env.observe(k,t)
                w[j]=w[j]*math.exp(alpha/(B*p[j])*(1/2+self.r[i*H:H*(i+1)].sum()/(100*H*math.log(T)+4*math.sqrt(H*math.log(T)))))    
    def rewards(self):
        return self.r_Exp 
    


class UCB_TP: #previously suggested one
    def __init__(self,delta,epsilon,T,seed,Environment):
        print('UCB-TP')
        
        np.random.seed(seed)
        self.Env=Environment
        self.r=np.zeros(T,float)
        self.r_Exp=np.zeros(T,float)
        n=0
        k=0
        mu=0
        for t in tqdm(range(T)):
                
            self.r_Exp[t],self.r[t]=self.Env.observe(k,t)
            n=n+1
            mu=(mu*(n-1)+self.r[t]+epsilon*(n-1))/n
            ucb=mu-epsilon*n+math.sqrt(8*math.log(T)/n)
                
            if ucb<1-delta:
                n=0
                mu=0
                k=k+1
            
    def rewards(self):
        return self.r_Exp  

class SSUCB:##previously suggested one
    def __init__(self,K,T,seed,Environment):
        print('SSUCB')
                    
        np.random.seed(seed)
        self.Env=Environment
        self.r=np.zeros(T,float)
        self.r_Exp=np.zeros(T,float)
        n=np.zeros(K)
        mu=np.zeros(K)
        ucb=np.zeros(K)

        for t in tqdm(range(T)):
            if t<K:
                k=t
            else:         
                k=np.argmax(ucb)    
                
            self.r_Exp[t],self.r[t]=self.Env.observe(k,n[k])
            mu[k]=(mu[k]*(n[k])+self.r[t])/(n[k]+1)                
            n[k]=n[k]+1
            ucb[k]=mu[k]+math.sqrt(2*math.log(1+(t+1)*(math.log(t+1))**2)/n[k])

    def rewards(self):
        return self.r_Exp  
