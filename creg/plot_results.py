import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

font_size=14

def model_training_time_ensemble():
    x=[200,2000,20000,200000,2000000]
    X=[0.00523,0.0144,0.12209,1.09748,13.21596]
    G=[0.00834,0.01452,0.0636,0.58815,7.70868]
    C=[0.074486,0.322717,2.94475,31.227796,306.769922]
    plt.loglog(x,X,"x-",label="XGboost")
    plt.loglog(x,G,"o-",label="GBoost")
    plt.loglog(x,C,"v-",label="Qreg")
    plt.legend()
    plt.xlabel("Number of training points", fontsize=font_size)
    plt.ylabel("Training time (s)", fontsize=font_size)
    plt.tick_params(labelsize = font_size)

    plt.show()

def model_training_time_base_models():
    x=[100,1000,10000,100000,1000000]
    LR=[0.00056 ,    0.00061   ,  0.00073  ,   0.00221 ,    0.02457 ]
    PR=[0.00260  ,   0.00613   ,  0.02261  ,   0.57020  ,   8.13341 ]
    DTR=[0.00019,    0.00042 ,    0.00291  ,   0.02642  ,   0.26859 ]
    KNN=[0.00028  ,  0.00041  ,   0.00178  ,   0.01814  ,   0.56045 ]
    SVR=[0.00032 ,   0.01614  ,   1.21145  ,   1078.83512 ]
    Gaussian=[0.13289 ,  2.15688  ,   780.21678 ]
    plt.loglog(x,LR,"x-",label="LR")
    plt.loglog(x,PR,"o-",label="PR")
    plt.loglog(x,DTR,"v-",label="DTR")
    plt.loglog(x,KNN,"*-",label="KNN")
    plt.loglog(x[:-1],SVR,".-",label="SVR")
    plt.loglog(x[:-2],Gaussian,"h-",label="Gaussian")
    plt.legend()
    plt.xlabel("Number of training points", fontsize=font_size)
    plt.ylabel("Training time (s)", fontsize=font_size)
    plt.tick_params(labelsize = font_size)

    plt.show()



def draw_tpcds_bars():
    def to_percent(y,pos):
        return '%.1f%%' % (y * 100)
    x=[1,3]
    x1 = [0.8,2.8]
    x2= [1.2,3.2]
    y1 = [0.0531, 0.0137]
    y2 = [0.436,2.592]

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    p1=ax1.bar(x1, y1, color='g',width=0.3)
    
    formatter = FuncFormatter(to_percent)
    ax1.yaxis.set_major_formatter(formatter)
    p2=ax2.bar(x2, y2,color='r',width=0.3)
    #ax1.set_xlabel('X data')
    plt.xticks(x,("Relative Error (%)",'Response Time (s)'))
    plt.legend((p1[0],p2[0]),('10k','100k'),loc='center')
    ax1.set_ylabel("Relative Error (%)")
    ax2.set_ylabel('Response Time (s)')
    
    plt.show()


if __name__=="__main__":
    draw_tpcds_bars()