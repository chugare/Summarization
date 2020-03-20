
import numpy as np

from matplotlib import pyplot as plt

def lrschedule(step,warmup,dmodel):

    arg1 = 1/np.sqrt(step)
    arg2 = step * (warmup ** -1.5)
    return 1/np.sqrt(dmodel) * np.minimum(arg1, arg2)

font1 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 23,
         }
font2 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 23,
         }

x = np.arange(0,200000,1000)
y = lrschedule(x,40000,200)
# print(y)
# fig,ax1 = plt.subplot()
plt.figure(1)
plt.plot(x,y)
plt.ylabel("Learning Rate",fontdict=font1)
plt.xlabel("Train Step",fontdict=font1)
plt.tight_layout()
plt.show()

plt.savefig('t1.pdf',dpi=600,format='pdf')