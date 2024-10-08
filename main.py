from Environment import *
from Algorithms import *
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import sys
from pathlib import Path


def run(T,num,repeat,beta, boolean=True): #regret vs T
    T_1=int(T/num)
    num=num+1
    std_list1=np.zeros(num)
    regret_list1=np.zeros(num)
    std_list2=np.zeros(num)
    regret_list2=np.zeros(num)
    std_list3=np.zeros(num)
    regret_list3=np.zeros(num)
    std_list4=np.zeros(num)
    regret_list4=np.zeros(num)
    T_list=np.zeros(num)
    if boolean: ##save data
        for i in range(num):
            print('num:', i)
            if i==0:
                T=1
            else:
                T=T_1*i
            if T==1:
                rho=1
            else:
                rho=1/math.log(T)
            
            delta=max(rho**(1/3),1/math.sqrt(T))
            V=rho*math.log(T)
            delta_V=max((V/T)**(1/(beta+2)),1/T**(1/(beta+1)),(V/T)**(1/3),1/T**(1/2)) #threshold value with V
            K=int(np.sqrt(T))
            T_list[i]=T
            regret=np.zeros(T,float)
            regret_sum=np.zeros(T,float)
            regret_sum_list1=np.zeros((repeat,T),float)
            regret_sum_list2=np.zeros((repeat,T),float)
            regret_sum_list3=np.zeros((repeat,T),float)
            regret_sum_list4=np.zeros((repeat,T),float)

            avg_regret_sum1=np.zeros(T,float)
            avg_regret_sum2=np.zeros(T,float)
            avg_regret_sum3=np.zeros(T,float)
            avg_regret_sum4=np.zeros(T,float)
        ###Run model
            for j in range(repeat):
                print('repeat: ',j)
                seed=j
                Env=rotting_many_Env(rho,seed,T,beta)
                algorithm1=UCBT_AW(delta_V,T,seed,Env)
                Env=rotting_many_Env(rho,seed,T,beta)
                algorithm2=AUCBT_AW(T,seed,Env)
                Env=rotting_many_Env(rho,seed,T,beta)
                algorithm3=UCB_TP(delta,rho,T,seed,Env)
                Env=rotting_many_Env(rho,seed,T,beta)
                algorithm4=SSUCB(K,T,seed,Env)
          
                opti_rewards=Env.optimal
                      
                regret=opti_rewards-algorithm1.rewards()
                regret_sum=np.cumsum(regret)
                regret_sum_list1[j,:]=regret_sum
                avg_regret_sum1+=regret_sum
                
                regret=opti_rewards-algorithm2.rewards()
                regret_sum=np.cumsum(regret)
                regret_sum_list2[j,:]=regret_sum
                avg_regret_sum2+=regret_sum
                
                regret=opti_rewards-algorithm3.rewards()
                regret_sum=np.cumsum(regret)
                regret_sum_list3[j,:]=regret_sum
                avg_regret_sum3+=regret_sum
                
                regret=opti_rewards-algorithm4.rewards()
                regret_sum=np.cumsum(regret)
                regret_sum_list4[j,:]=regret_sum
                avg_regret_sum4+=regret_sum
                
            avg1=avg_regret_sum1/repeat
            sd1=np.std(regret_sum_list1,axis=0)
            avg2=avg_regret_sum2/repeat
            sd2=np.std(regret_sum_list2,axis=0)
            avg3=avg_regret_sum3/repeat
            sd3=np.std(regret_sum_list3,axis=0)
            avg4=avg_regret_sum4/repeat
            sd4=np.std(regret_sum_list4,axis=0)

            regret = dict()
            std=dict()
            regret['algorithm1']=avg1[T-1]
            std['algorithm1']=sd1[T-1]
            regret['algorithm2']=avg2[T-1]
            std['algorithm2']=sd2[T-1]
            regret['UCB-TP']=avg3[T-1]
            std['UCB-TP']=sd3[T-1]
            regret['SSUCB']=avg4[T-1]
            std['SSUCB']=sd4[T-1]
            
            
            regret_list1[i]=avg1[T-1]
            std_list1[i]=sd1[T-1]
            regret_list2[i]=avg2[T-1]
            std_list2[i]=sd2[T-1]
            regret_list3[i]=avg3[T-1]
            std_list3[i]=sd3[T-1]
            regret_list4[i]=avg4[T-1]
            std_list4[i]=sd4[T-1]            
            
            ##Save data
            Path("./result").mkdir(parents=True, exist_ok=True)
            filename_1='T'+str(T)+'num'+str(num)+'beta'+str(beta)+'repeat'+str(repeat)+'regret.txt'
            with open('./result/'+filename_1, 'wb') as f:
                pickle.dump(regret, f)
                f.close()

            filename_2='T'+str(T)+'num'+str(num)+'beta'+str(beta)+'repeat'+str(repeat)+'std.txt'
            with open('./result/'+filename_2, 'wb') as f:
                pickle.dump(std, f)
                f.close()
    
    else: ##load data
        print('load data without running')
        for i in range(num):
            if i==0:
                T=1
            else:
                k=i
                T=T_1*k
            T_list[i]=T
            filename_1='T'+str(T)+'num'+str(num)+'beta'+str(beta)+'repeat'+str(repeat)+'regret.txt'
            filename_2='T'+str(T)+'num'+str(num)+'beta'+str(beta)+'repeat'+str(repeat)+'std.txt'
            pickle_file1 = open('./result/'+filename_1, "rb")
            pickle_file2 = open('./result/'+filename_2, "rb")
            objects = []

            while True:
                try:
                    objects.append(pickle.load(pickle_file1))
                except EOFError:
                    break
            pickle_file1.close()
            regret=objects[0]
            objects = []
            while True:
                try:
                    objects.append(pickle.load(pickle_file2))
                except EOFError:
                    break
            pickle_file2.close()
            std=objects[0]
            avg1=regret['algorithm1']
            sd1=std['algorithm1']
            avg2=regret['algorithm2']
            sd2=std['algorithm2']
            avg3=regret['UCB-TP']
            sd3=std['UCB-TP']
            avg4=regret['SSUCB']
            sd4=std['SSUCB']
            filename_1='T'+str(T)+'num'+str(num)+'beta'+str(beta)+'repeat'+str(repeat)+'regret.txt'
            filename_2='T'+str(T)+'num'+str(num)+'beta'+str(beta)+'repeat'+str(repeat)+'std.txt'
            pickle_file1 = open('./result/'+filename_1, "rb")
            pickle_file2 = open('./result/'+filename_2, "rb")
            objects = []

            while True:
                try:
                    objects.append(pickle.load(pickle_file1))
                except EOFError:
                    break
            pickle_file1.close()
            regret=objects[0]
            objects = []
            while True:
                try:
                    objects.append(pickle.load(pickle_file2))
                except EOFError:
                    break
            pickle_file2.close()
            std=objects[0]
            avg1=regret['algorithm1']
            sd1=std['algorithm1']          
            regret_list1[i]=avg1
            std_list1[i]=sd1
            regret_list2[i]=avg2
            std_list2[i]=sd2
            regret_list3[i]=avg3
            std_list3[i]=sd3
            regret_list4[i]=avg4
            std_list4[i]=sd4
            
    fig,(ax)=plt.subplots(1,1)


    # regret_ref_upper_1=[(max(t**((beta+1)/(beta+2))*np.log(t),t**((2)/(3))*np.log(t))) for t in T_list]
    # regret_ref_upper_2=[(max(t**((beta+1)/(beta+2))*np.log(t),t**((2)/(3))*np.log(t))+max(t**((2*beta+1)/(2*beta+2))*np.log(t),t**((3)/(4))*np.log(t))) for t in T_list]

    ax.errorbar(x=T_list, y=regret_list1, yerr=1.96*std_list1/np.sqrt(repeat), color="royalblue", capsize=7,capthick=2, elinewidth=2,linewidth=3,
                 marker="^", markersize=0,label='Algorithm 1',zorder=3,ls='-') 
    ax.errorbar(x=T_list, y=regret_list2, yerr=1.96*std_list2/np.sqrt(repeat), color="lightseagreen", capsize=7,capthick=2,elinewidth=2,linewidth=3,
                 marker="o", markersize=0,label='Algorithm 2',zorder=2,ls='--')
    ax.errorbar(x=T_list, y=regret_list3, yerr=1.96*std_list3/np.sqrt(repeat), color="salmon", capsize=7,capthick=2,elinewidth=2,linewidth=3,
                 marker="s", markersize=0,label='UCB-TP',zorder=1,ls=':')
    ax.errorbar(x=T_list, y=regret_list4, yerr=1.96*std_list4/np.sqrt(repeat), color="gray", capsize=7,capthick=2,elinewidth=2,linewidth=3,
                 marker="s", markersize=0,label='SSUCB',zorder=1,ls='-.')
    # ax.errorbar(x=T_list, y=regret_ref_upper_1, yerr=0, color="lightsteelblue", capsize=7,capthick=2,elinewidth=2,linewidth=3,
    #              marker="s", markersize=0,label='Regret upper bound (Alg1)',zorder=1,ls='--')
    # ax.errorbar(x=T_list, y=regret_ref_upper_2, yerr=0, color="palegreen", capsize=7,capthick=2,elinewidth=2,linewidth=3,
    #              marker="s", markersize=0,label='Regret upper bound (Alg2)',zorder=1,ls='--')  

    Path("./plot").mkdir(parents=True, exist_ok=True)



    #font size
    ax.tick_params(labelsize=18)
    plt.rc('legend',fontsize=14)
    ax.yaxis.get_offset_text().set_fontsize(18)
    ax.xaxis.get_offset_text().set_fontsize(18)
    # remove the errorbars in legend
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    labels = [labels[3], labels[1], labels[2],labels[0]]
    handles=[handles[3], handles[1],handles[2],handles[0]]
    # labels = [labels[3], labels[2], labels[1], labels[0]]
    # handles=[handles[3],handles[2], handles[1],handles[0]]
    ax.legend(handles, labels,numpoints=1)
    # plot 
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.xlabel(r'$T$',fontsize=18)
    plt.ylabel(r'E$[R^{\pi}(T)]$',fontsize=18)
    plt.savefig('./plot/T'+str(T)+'num'+str(num)+'beta'+str(beta)+'repeat'+str(repeat)+'V.pdf',bbox_inches='tight')
    plt.show()
    plt.clf()    
    


        
if __name__=='__main__':
    # Read input
    opt = int(sys.argv[1]) # '1': (left) in Figure 1, '2': (right) in Figure 4
    
    run_bool=True # True: run model and save data with plot, False: load data with plot.
    
    
    if opt==1: 
        T=5*10**6  # Maximum Time horizon
        num=5 # number of investigated horizon times over maximum time horizon
        repeat=10    # number of running algorithms using different seeds.
        beta=1
        run(T,num,repeat, beta, run_bool)
    if opt==2: 
        T=5*10**6  
        num=5 
        repeat=10   
        beta=0.5
        run(T,num,repeat, beta, run_bool)
    if opt==3: 
        T=5*10**6  
        num=5 
        repeat=10   
        beta=2
        run(T,num,repeat, beta, run_bool)