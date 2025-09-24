

import numpy as np
import matplotlib.pyplot as plt
import math
#plt.rcParams['pdf.fonttype'] = 42


rc = {"font.family" : "serif", 
      "mathtext.fontset" : "stix"}
plt.rcParams.update(rc)
#plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]


plt.rcParams['ps.fonttype'] = 42
#plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

plt.rcParams['ps.fonttype'] = 42

def readFile(path):
    f = open(path)
    first_ele = True
    for data in f.readlines():
        data = data.strip('\n')
        nums = data.split(" ")
        if first_ele:
            nums = [float(x) for x in nums ]
            matrix = np.array(nums)
            first_ele = False
        else:
            nums = [float(x) for x in nums ]
            matrix = np.c_[matrix,nums]
    return matrix
    f.close()


if __name__ == '__main__':
    
#########################################################
#Please select different file to generate different figures: TAU4, TAU8, TAU12, AND TAU16
    
    DSCMD=np.transpose(readFile('Result_FedRecu_TAU4.txt'))
    DSCMDD=np.transpose(readFile('Result_FedTrack_TAU4.txt'))
    DSCMDDD=np.transpose(readFile('Result_FedLin_TAU4.txt'))
    DSCMDDDD=np.transpose(readFile('Result_SCAFFOLD_TAU4.txt'))
    
    
    DSCMD1=DSCMD
    DSCMDD1=DSCMDD
    DSCMDDD1=DSCMDDD
    DSCMDDDD1=DSCMDDDD
    
    
    
    
    k=190
    kk=160
    
    BASE1=np.zeros(k)
    BASE2=np.zeros(k)
    BASE3=np.zeros(kk)
    BASE4=np.zeros(kk)
    BASE5=np.zeros(k)
    BASE6=np.zeros(k)
    BASE7=np.zeros(kk)
    BASE8=np.zeros(kk)
    BASE9=np.zeros(k)
    BASE10=np.zeros(k)
    BASE11=np.zeros(kk)
    BASE12=np.zeros(kk)
    BASE13=np.zeros(k)
    BASE14=np.zeros(k)
    BASE15=np.zeros(kk)
    BASE16=np.zeros(kk)
    
    
    
    
    k0=0
    
    
    for j in range(0,k-1):
        BASE1[j]=j
        BASE2[j]=DSCMD1[j]
        
        
    for j in range(0,k-1):
        BASE5[j]=j
        BASE6[j]=DSCMDD1[j]
        
    for j in range(0,k-1):
        BASE9[j]=j
        BASE10[j]=DSCMDDD1[j]
        
    
    for j in range(0,k-1):
        BASE13[j]=j
        BASE14[j]=DSCMDDDD1[j]
    
    
    
    for jj in range(0,kk):
        BASE3[jj]=BASE1[k0+jj]
        BASE4[jj]=BASE2[k0+jj]
      
        
    for jj in range(0,kk):
        BASE7[jj]=BASE5[k0+jj]
        BASE8[jj]=BASE6[k0+jj]
       
    
    
    for jj in range(0,kk):
        BASE11[jj]=BASE9[k0+jj]
        BASE12[jj]=BASE10[k0+jj]
        
    
    for jj in range(0,kk):
        BASE15[jj]=BASE13[k0+jj]
        BASE16[jj]=BASE14[k0+jj]
        
        
    
    x = BASE3
    y = BASE4
    
    
    x2=BASE7
    y2=BASE8
    
    x3=BASE11
    y3=BASE12
    
    
    x4=BASE15
    y4=BASE16
    
    
   
    
    
    
    plt.xticks(fontsize = 10, fontname = 'times new roman')
    plt.yticks(fontsize = 10, fontname = 'times new roman')
    
    
    plt.semilogy(x4,y4,label='SCAFFOLD',color='g',linewidth=2.2,linestyle=':')
    plt.semilogy(x2,y2,label='FedTrack',color='y',linewidth=2.2,linestyle='-.')
    plt.semilogy(x3,y3,label='FedLin',color='b',linewidth=2.2,linestyle='--')
    plt.semilogy(x,y,label='FedRecu', color='r',linewidth=2.2,linestyle='solid')
    
   
    font1 = {'family' : 'Times New Roman','weight' : 'normal','size'   : 18, }
 
    plt.legend(prop = {'size':14}) 
    plt.grid()
    plt.xlabel(r'${k}$',fontsize=19)
    plt.ylabel(r'${f}(x(k\tau))-f(x\mathcal{*})$',fontsize=19)
    
  
 
    plt.savefig(fname="different_algorithm_TAU4.jpg",format="jpg", bbox_inches='tight')
    plt.savefig(fname="different_algorithm_TAU4.pdf",format="pdf", bbox_inches='tight')
    plt.show() 
    
 

