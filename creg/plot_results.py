import matplotlib.pyplot as plt

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


if __name__=="__main__":
    model_training_time_ensemble()