import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
import matplotlib.patches as mpatch

overlap = {name for name in mcd.CSS4_COLORS
           if "xkcd:" + name in mcd.XKCD_COLORS}

font_size = 14
colors = {
    "DBEst_1k": mcd.XKCD_COLORS['xkcd:coral'],
    "DBEst_10k": mcd.XKCD_COLORS['xkcd:orange'],  # blue
    "DBEst_100k": mcd.XKCD_COLORS['xkcd:orangered'],  # green
    "DBEst_1m": mcd.XKCD_COLORS['xkcd:red'],  # yellow
    "BlinkDB_1k": mcd.XKCD_COLORS['xkcd:lightblue'],  # red
    "BlinkDB_10k": mcd.XKCD_COLORS['xkcd:turquoise'],  # red
    "BlinkDB_100k": mcd.XKCD_COLORS['xkcd:teal'],  # cyan
    "BlinkDB_1m": mcd.XKCD_COLORS['xkcd:azure'],  # magenta
    "BlinkDB_5m": mcd.XKCD_COLORS['xkcd:blue'],  # red
    "BlinkDB_26m": mcd.XKCD_COLORS['xkcd:darkblue'],  # red
    "lightgreen": mcd.XKCD_COLORS['xkcd:lightgreen'],
    "green": mcd.XKCD_COLORS['xkcd:green'],
    "orange": mcd.XKCD_COLORS['xkcd:orange'],
    "orangered": mcd.XKCD_COLORS['xkcd:orangered'],
    "red": mcd.XKCD_COLORS['xkcd:red'],
}

# colors = {
#     "DBEst_1k": mcd.XKCD_COLORS['xkcd:lime'],
#     "DBEst_10k": mcd.XKCD_COLORS['xkcd:lightgreen'],  # blue
#     "DBEst_100k": mcd.XKCD_COLORS['xkcd:green'],  # green
#     "DBEst_1m": mcd.XKCD_COLORS['xkcd:darkgreen'],  # yellow
#     "BlinkDB_1k": mcd.XKCD_COLORS['xkcd:lightblue'],  # red
#     "BlinkDB_10k": mcd.XKCD_COLORS['xkcd:turquoise'],  # red
#     "BlinkDB_100k": mcd.XKCD_COLORS['xkcd:teal'],  # cyan
#     "BlinkDB_1m": mcd.XKCD_COLORS['xkcd:azure'],  # magenta
#     "BlinkDB_5m": mcd.XKCD_COLORS['xkcd:blue'],  # red
# }


def model_training_time_ensemble():
    x = [200, 2000, 20000, 200000, 2000000]
    X = [0.00523, 0.0144, 0.12209, 1.09748, 13.21596]
    G = [0.00834, 0.01452, 0.0636, 0.58815, 7.70868]
    C = [0.074486, 0.322717, 2.94475, 31.227796, 306.769922]
    plt.loglog(x, X, "x-", label="XGboost")
    plt.loglog(x, G, "o-", label="GBoost")
    plt.loglog(x, C, "v-", label="Qreg")
    plt.legend()
    plt.xlabel("Number of training points", fontsize=font_size)
    plt.ylabel("Training time (s)", fontsize=font_size)
    plt.tick_params(labelsize=font_size)

    plt.show()


def model_training_time_base_models():
    x = [100, 1000, 10000, 100000, 1000000]
    LR = [0.00056,    0.00061,  0.00073,   0.00221,    0.02457]
    PR = [0.00260,   0.00613,  0.02261,   0.57020,   8.13341]
    DTR = [0.00019,    0.00042,    0.00291,   0.02642,   0.26859]
    KNN = [0.00028,  0.00041,   0.00178,   0.01814,   0.56045]
    SVR = [0.00032,   0.01614,   1.21145,   1078.83512]
    Gaussian = [0.13289,  2.15688,   780.21678]
    plt.loglog(x, LR, "x-", label="LR")
    plt.loglog(x, PR, "o-", label="PR")
    plt.loglog(x, DTR, "v-", label="DTR")
    plt.loglog(x, KNN, "*-", label="KNN")
    plt.loglog(x[:-1], SVR, ".-", label="SVR")
    plt.loglog(x[:-2], Gaussian, "h-", label="Gaussian")
    plt.legend()
    plt.xlabel("Number of training points", fontsize=font_size)
    plt.ylabel("Training time (s)", fontsize=font_size)
    plt.tick_params(labelsize=font_size)

    plt.show()


def draw_tpcds_bars():
    def to_percent(y, pos):
        return '%.1f%%' % (y * 100)
    x = [1, 3]
    x1 = [0.8, 2.8]
    x2 = [1.2, 3.2]
    y1 = [0.0531, 0.0137]
    y2 = [0.436, 2.592]

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    p1 = ax1.bar(x1, y1, color='g', width=0.3)

    formatter = FuncFormatter(to_percent)
    ax1.yaxis.set_major_formatter(formatter)
    p2 = ax2.bar(x2, y2, color='r', width=0.3)
    #ax1.set_xlabel('X data')
    plt.xticks(x, ("Relative Error (%)", 'Response Time (s)'))
    plt.legend((p1[0], p2[0]), ('10k', '100k'), loc='center')
    ax1.set_ylabel("Relative Error (%)")
    ax2.set_ylabel('Response Time (s)')



    plt.show()

def to_percent(y, pos):
    return '%.1f%%' % (y * 100)

def plt_tpcds_multi_cp_relative_error():
    plt.rcParams.update({'font.size': 12})
    width = 0.20
    data = [
      [0.0650,  0.0279,  0.0393, 0.0441],#, 0.0486],
      [0.0727,  0.0596,  0.0155, 0.0493],#, 0.0335],
      [0.0317,  0.0220,  0.0104, 0.0214],#, 0.0182],
      [0.0628,  0.0707,  0.0069, 0.0468] #, 0.0212]
      ]



    X = np.arange(4)
  
    fig, ax = plt.subplots()

    p1 = plt.bar(X + 0.00, data[0], color=colors["DBEst_10k"], width=width)
    p2 = plt.bar(X + 0.20, data[1], color=colors["BlinkDB_10k"], width=width)
    p3 = plt.bar(X + 0.40, data[2], color=colors["DBEst_100k"], width=width)
    p4 = plt.bar(X + 0.60, data[3], color=colors["BlinkDB_100k"], width=width)

    
    plt.legend((p1[0], p2[0], p3[0], p4[0]),
        ('DBEst_10k', 'BlinkDB_10k','DBEst_100k','BlinkDB_100k'), loc='1')

    plt.xticks(X+1.5*width, ("COUNT", 'SUM','AVG','OVERALL'))
    ax.set_ylabel("Relative Error (%)")
    # ax.set_yscale('log')
    formatter = FuncFormatter(to_percent)
    ax.yaxis.set_major_formatter(formatter)

    plt.show()

def plt_tpcds_multi_cp_response_time():
    plt.rcParams.update({'font.size': 12})
    width = 0.20
    data = [
      [0.015  ,  0.588  , 0.436 ],#, 0.0486],
      [91.32  ,  64.290 , 56.140 ],#, 0.0335],
      [0.085  ,  2.551  , 2.592 ],#, 0.0182],
      [187.05 ,  143.99 , 171.57] #, 0.0212]
      ]



    X = np.arange(3)
  
    fig, ax = plt.subplots()

    p1 = plt.bar(X + 0.00, data[0], color=colors["DBEst_10k"], width=width)
    p2 = plt.bar(X + 0.20, data[1], color=colors["BlinkDB_10k"], width=width)
    p3 = plt.bar(X + 0.40, data[2], color=colors["DBEst_100k"], width=width)
    p4 = plt.bar(X + 0.60, data[3], color=colors["BlinkDB_100k"], width=width)

    
    plt.legend((p1[0], p2[0], p3[0], p4[0]),
        ('DBEst_10k', 'BlinkDB_10k','DBEst_100k','BlinkDB_100k'), loc='center left')

    plt.xticks(X+1.5*width, ("COUNT", 'SUM','AVG','OVERALL'))
    ax.set_ylabel("Response Time (s)")
    ax.set_yscale('log')
    formatter = FuncFormatter(to_percent)
    # ax.yaxis.set_major_formatter(formatter)

    plt.show()

def plt_tpcds_multi_cp_training_time():
    plt.rcParams.update({'font.size': 12})
    width = 0.30
    data = [
      [281.5, 286   ],#, 0.0486],
      [39.8,  278.3 ],#, 0.0335],
      [2328,  2342  ]
      ]



    X = np.arange(2)
  
    fig, ax = plt.subplots()

    p1 = plt.bar(X + 0.00, data[0], color=colors["orange"], width=width)
    p2 = plt.bar(X + 0.00, data[1], color=colors["red"], width=width,bottom=data[0])
    p3 = plt.bar(X + 0.30, data[2], color=colors["BlinkDB_10k"], width=width)


    plt.legend((p1[0], p2[0], p3[0]),
        ('DBEst Sampling', 'DBEst training','BlinkDB Sampling','BlinkDB_100k'), loc='center')

    plt.xticks(X+0.5*width, ("10k", '100k','AVG','OVERALL'))
    ax.set_ylabel("Total Training Time (s)")
    # ax.set_yscale('log')
    formatter = FuncFormatter(to_percent)
    # ax.yaxis.set_major_formatter(formatter)

    plt.show()


def plt_tpcds_multi_cp_memory_usage():
    plt.rcParams.update({'font.size': 12})
    width = 0.30
    data = [
      [5.9959, 22.195   ],#, 0.0486],
      [1024,  1340 ],#, 0.0335],
      ]



    X = np.arange(2)
  
    fig, ax = plt.subplots()

    p1 = plt.bar(X + 0.00, data[0], color=colors["DBEst_10k"], width=width)
    p2 = plt.bar(X + width, data[1], color=colors["BlinkDB_10k"], width=width)
    # p3 = plt.bar(X + 0.30, data[2], color=colors["BlinkDB_10k"], width=width)


    plt.legend((p1[0], p2[0]),
        ('DBEst', 'BlinkDB','BlinkDB Sampling','BlinkDB_100k'), loc='center')

    plt.xticks(X+0.5*width, ("10k", '100k','AVG','OVERALL'))
    ax.set_ylabel("Memory Usage (MB)")
    ax.set_yscale('log')
    formatter = FuncFormatter(to_percent)
    # ax.yaxis.set_major_formatter(formatter)

    plt.show()



##----------------------------------------------------------------------------------------##

def plt_tpcds_single_cp_sample_size_response_time():
    plt.rcParams.update({'font.size': 12})
    width = 0.20
    data =  [
            [0.00829, 0.2673276 ,  0.03481, 0.0352 , 0.04278,0.05179, 0.07853 ,0.07707],
            [0.09   , 3.1       ,   0.22  ,  0.23  , 0.22   , 0.29  ,0.053    ,0.053],
            [1.13   , 41.46232  ,  3.07   , 3.08   , 3.01   ,4.66   , 0.06    ,0.06],
            [7.07   , 200       , 21.07   ,  21.13 , 19.07  , 26.44 ,0.056    ,0.056],
            ]





    X = np.arange(8)
  
    fig, ax = plt.subplots()

    p1 = plt.bar(X + 0.00, data[0], color=colors["lightgreen"], width=width)
    p2 = plt.bar(X + 0.20, data[1], color=colors["green"], width=width)
    p3 = plt.bar(X + 0.40, data[2], color=colors["BlinkDB_1m"], width=width)
    p4 = plt.bar(X + 0.60, data[3], color=colors["BlinkDB_5m"], width=width)

    
    plt.legend((p1[0], p2[0], p3[0], p4[0]),
        ('10k', '100k','1m','5m'), loc='1')

    plt.xticks(X+1.5*width, ("COUNT", 'PERCENTILE','VARIANCE','STDDEV','SUM','AVG','MIN','MAX'))
    ax.set_ylabel("Query Response Time (s)")
    ax.set_yscale('log')
    formatter = FuncFormatter(to_percent)
    # ax.yaxis.set_major_formatter(formatter)

    for item in ax.axes.get_xticklabels():
        item.set_rotation(60)
    plt.subplots_adjust(bottom=0.22)

    plt.show()

def plt_tpcds_single_cp_sample_size_relative_error():
    plt.rcParams.update({'font.size': 12})
    width = 0.20
    data =  [
            [0.0607,  0.01   , 0.0704,  0.0151,  0.0831,  0.0397,  0.3092,  0.1971],
            [0.0259,  0.00607, 0.0103,  0.0045,  0.0248, 0.00620,  0.3021,  0.204],
            [0.0099,  0.00360, 0.0092,  0.0039,  0.0103,  0.0021,  0.3139,  0.2088],
            [0.0048,  0.0035 , 0.009 ,  0.0038,  0.0051,  0.0015,  0.3163,  0.2104],
            ]

    X = np.arange(8)
  
    fig, ax = plt.subplots()

    p1 = plt.bar(X + 0.00, data[0], color=colors["lightgreen"], width=width)
    p2 = plt.bar(X + 0.20, data[1], color=colors["green"], width=width)
    p3 = plt.bar(X + 0.40, data[2], color=colors["BlinkDB_1m"], width=width)
    p4 = plt.bar(X + 0.60, data[3], color=colors["BlinkDB_5m"], width=width)

    
    plt.legend((p1[0], p2[0], p3[0], p4[0]),
        ('10k', '100k','1m','5m'), loc='1')

    plt.xticks(X+1.0*width, ("COUNT", 'PERCENTILE','VARIANCE','STDDEV','SUM','AVG','MIN','MAX'))
    ax.set_ylabel("Relative Error (%)")
    ax.set_yscale('log')
    formatter = FuncFormatter(to_percent)
    ax.yaxis.set_major_formatter(formatter)

    for item in ax.axes.get_xticklabels():
        item.set_rotation(60)
    plt.subplots_adjust(bottom=0.22)
    plt.subplots_adjust(left=0.15)
    plt.show()

def plt_tpcds_single_cp_sample_size_relative_error_blinkdb():
    plt.rcParams.update({'font.size': 12})
    width = 0.16
    data =  [
            [0.31361,     0.31337,     0.10307], 
            [0.09527,     0.09525,     0.01158], 
            [0.07199,     0.07227,     0.00375], 
            [0.06938,     0.06757,     0.00020], 
            [0.06760,     0.06760,     0.00020],
            ]


    X = np.arange(3)
  
    fig, ax = plt.subplots()

    p1 = plt.bar(X + 0.00, data[0], color=colors["lightgreen"], width=width)
    p2 = plt.bar(X + 1*width, data[1], color=colors["green"], width=width)
    p3 = plt.bar(X + 2*width, data[2], color=colors["BlinkDB_1m"], width=width)
    p4 = plt.bar(X + 3*width, data[3], color=colors["BlinkDB_5m"], width=width)
    p5 = plt.bar(X + 4*width, data[4], color=colors["BlinkDB_26m"], width=width)

    
    plt.legend((p1[0], p2[0], p3[0], p4[0],p5[0]),
        ('1k', '10k','100k','1m','26m'), loc='1')

    plt.xticks(X+2*width, ("COUNT", 'SUM','AVG','STDDEV','SUM','AVG','MIN','MAX'))
    ax.set_ylabel("Relative Error (%)")
    ax.set_yscale('log')
    formatter = FuncFormatter(to_percent)
    ax.yaxis.set_major_formatter(formatter)

    # for item in ax.axes.get_xticklabels():
    #     item.set_rotation(60)
    # plt.subplots_adjust(bottom=0.22)
    plt.subplots_adjust(left=0.15)
    plt.show()


def plt_tpcds_single_cp_sample_size_response_time_blinkdb():
    plt.rcParams.update({'font.size': 12})
    width = 0.16
    data =  [
            [1.763  , 1.872 , 1.871], 
            [11.65  , 12.52 , 12.28], 
            [115.36 , 131.86, 126.04], 
            [174.65 , 193.32, 185.24], 
            [291.19 , 265.98, 558.94],
            ]







    X = np.arange(3)
  
    fig, ax = plt.subplots()

    p1 = plt.bar(X + 0.00, data[0], color=colors["lightgreen"], width=width)
    p2 = plt.bar(X + 1*width, data[1], color=colors["green"], width=width)
    p3 = plt.bar(X + 2*width, data[2], color=colors["BlinkDB_1m"], width=width)
    p4 = plt.bar(X + 3*width, data[3], color=colors["BlinkDB_5m"], width=width)
    p5 = plt.bar(X + 4*width, data[4], color=colors["BlinkDB_26m"], width=width)

    
    plt.legend((p1[0], p2[0], p3[0], p4[0],p5[0]),
        ('1k', '10k','100k','1m','26m'), loc='1')

    plt.xticks(X+2*width, ("COUNT", 'SUM','AVG','STDDEV','SUM','AVG','MIN','MAX'))
    ax.set_ylabel("Response Time (s)")
    ax.set_yscale('log')
    # formatter = FuncFormatter(to_percent)
    # ax.yaxis.set_major_formatter(formatter)

    # for item in ax.axes.get_xticklabels():
    #     item.set_rotation(60)
    # plt.subplots_adjust(bottom=0.22)
    plt.subplots_adjust(left=0.15)
    plt.show()


def plt_tpcds_single_cp_training_time_comparison():
    fig, ax = plt.subplots()
    y1=[411, 421.295, 442.629, 779.5,   3847,    17228]
    y2=[2321,    2326,    2343,    2342,    2355,    2378 ]
    x=[1E3,1E4,1E5,1E6,5E6,26E6]
    p1=plt.plot(x,y1,'rx-')
    p2=plt.plot(x,y2,'gh-')

    plt.legend((p1[0], p2[0]),
        ('DBEst', 'BlinkDB'), loc='1')

    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Training Time (s)")

    plt.xticks([5E6,26E6],('5m','26m'))
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.show()

def plt_tpcds_single_cp_memory_cost_comparison():
    fig, ax = plt.subplots()
    y1=[78 , 99 , 208, 787.6923077, 4096]
    y2=[0.486134 ,   1.995219 ,   20.265429 ,  94.901193 ,  484.876566 ]
    x=[1E4,1E5,1E6,5E6,26E6]
    p1=plt.plot(x,y1,'rx-')
    p2=plt.plot(x,y2,'gh-')

    plt.legend((p1[0], p2[0]),
        ('DBEst', 'BlinkDB'), loc='1')

    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Memory Overhead (MB)")

    plt.xticks([5E6,26E6],('5m','26m'))
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.show()

if __name__ == "__main__":
    
    # print(mcd.XKCD_COLORS['xkcd:darkblue'])
    plt_tpcds_single_cp_memory_cost_comparison()
