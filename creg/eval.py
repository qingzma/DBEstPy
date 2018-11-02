import re
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
import matplotlib.patches as mpatch

overlap = {name for name in mcd.CSS4_COLORS
           if "xkcd:" + name in mcd.XKCD_COLORS}

font_size = 14 #14
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
alpha = {
    "1": 0.1,
    "2": 0.3,
    "3": 0.5,
    "4": 0.7,
    "5": 0.9,
    '6': 1.0
}

def to_percent(y, pos):
    return '%.1f%%' % (y * 100)

def read_results(file, b_remove_null=True):
    """read the group by value and the corresponding aggregate within
    a given range, used to compare the accuracy.the

    Output: a dict contating the 

    Args:
        file (file): path to the file
    """

    key_values = {}
    with open(file) as f:
        # print("Start reading file " + file)
        index = 1
        for line in f:
            # ignore empty lines
            if  line.strip():
                key_value = line.replace(
                    "(", " ").replace(")", " ").replace(";", "").replace(",", "")
                # self.logger.logger.info(key_value)
                key_value = re.split('\s+', key_value)
                # remove empty strings caused by sequential blank spaces.
                key_value = list(filter(None, key_value))
                key_values[key_value[0]] = key_value[1]
    if ('NULL' in key_values) and b_remove_null:
        key_values.pop('NULL', None)


    key_values.pop('9.0', None)
    # print(key_values)
    return key_values


def avg_relative_error(ground_truth, predictions):
    """calculate the relative error between ground truth and predictions

    Args:
        ground_truth (dict): the ground truth, with keys and values
        predictions (dict): the predictions, with keys and values

    Returns:
        float: the average relative error 
    """
    if len(ground_truth) != len(predictions):
        print("Length mismatch!")
        print("Length of ground_truth is " + str(len(ground_truth)))
        print("Length of predictions is " + str(len(predictions)))
        print("System aborts!")
        sys.exit(1)

    relative_errors = []
    # ground_truth.pop('9.0', None)
    # ground_truth.pop('NULL', None)
    for key_gt, value_gt in ground_truth.items():
        if (ground_truth[key_gt] != 0):
            re = abs(float(ground_truth[key_gt]) -
                     float(predictions[key_gt])) / float(ground_truth[key_gt])
            # print(key_gt + str(re))
            relative_errors.append(re)
        else:
            print(
                "Zero is found in ground_truth, so removed to calculate relative error.")
    # print(sum(relative_errors))
    # print((relative_errors))
    # print(len(relative_errors))
    return sum(relative_errors) / len(relative_errors)

def avg_relative_errors():
    averag_errors_blinkdb=[]
    averag_errors_DBEst=[]
    for func in ['count','sum','avg']:
        print("---------------------"+func+"---------------------")
        errors_blinkdb = []
        errors_DBEst = []
        for index in range(1,11):
            file_name=func+str(int(index))
            ground_truth = read_results('../data/tpcds5m/groundtruth/'+file_name+'.result')
            predictions_blinkdb = read_results('../data/tpcds5m/blinkdb/'+file_name+'.txt') #../data/tpcds5m/blinkdb/sum1.txt
            predictions_DBEst = read_results('../data/tpcds5m/DBEst_integral/xgboost/'+file_name+'.txt')
            # if index == 1 and (func =='avg'):
            #     print("groundtruth"+str(ground_truth))
            #     print("blinkdb"+str(predictions_blinkdb))
            #     print("DBEst"+str(predictions_DBEst))
            #     print(len(ground_truth))
            #     print(len(predictions_blinkdb))
            #     print(len(predictions_DBEst))


            errors_blinkdb.append(avg_relative_error(ground_truth,predictions_blinkdb))
            errors_DBEst.append(avg_relative_error(ground_truth,predictions_DBEst))
            # print("averge is "+str(sum(errors_DBEst)/len(errors_DBEst)))
        averag_errors_blinkdb.append(sum(errors_blinkdb)/len(errors_blinkdb))
        averag_errors_DBEst.append(sum(errors_DBEst)/len(errors_DBEst))
        
        # print(errors_blinkdb)
        # print("errors_DBEst"+str(errors_DBEst))
        # print(sum(errors_DBEst)/len(errors_DBEst))
        # print("errors_Blinkdb"+str(errors_blinkdb))
        # print(sum(errors_blinkdb)/len(errors_blinkdb))
    print(averag_errors_blinkdb)
    print(averag_errors_DBEst)

def avg_relative_errors_per_group_value(group_num=501,function='count',size="100k"):
    res_per_group_DBEst={}
    res_per_group_blinkdb={}
    for func in ['count','sum','avg']:
        print("---------------------"+func+"---------------------")
        res_per_group_DBEst[func]=[]
        res_per_group_blinkdb[func]=[]
        
        re_per_group_DBEst={}
        re_per_group_blinkdb={}
        for index in range(1,11):
            print("Query Number: "+str(index)+"------------------")
            file_name=func+str(int(index))
            if group_num == 8:
                if size=="100k":
                    ground_truth = read_results('../data/tpcds_groupby_few_groups/groundtruth/'+file_name+'.result')
                    predictions_blinkdb = read_results('../data/tpcds_groupby_few_groups/blinkdb_100k_new/'+file_name+'.txt') #../data/tpcds5m/blinkdb/sum1.txt
                    predictions_DBEst = read_results('../data/tpcds_groupby_few_groups/DBEst_integral_100k/'+file_name+'.txt')
                if size=="1m":
                    ground_truth = read_results('../data/tpcds_groupby_few_groups/groundtruth/'+file_name+'.result')
                    predictions_blinkdb = read_results('../data/tpcds_groupby_few_groups/blinkdb_1m_new/'+file_name+'.txt') #../data/tpcds5m/blinkdb/sum1.txt
                    predictions_DBEst = read_results('../data/tpcds_groupby_few_groups/DBEst_integral_1m/'+file_name+'.txt')
            if group_num == 501:
                # ground_truth = read_results('../data/tpcds5m/groundtruth/'+file_name+'.result')
                # predictions_blinkdb = read_results('../data/tpcds5m/blinkdb/'+file_name+'.txt') #../data/tpcds5m/blinkdb/sum1.txt
                # predictions_DBEst = read_results('../data/tpcds5m/DBEst_integral/'+file_name+'.txt')
                ground_truth = read_results('../data/tpcds5m/groundtruth/'+file_name+'.result')
                predictions_blinkdb = read_results('../data/tpcds5m/DBEst_integral/xgboost/'+file_name+'.txt') #../data/tpcds5m/blinkdb/sum1.txt
                predictions_DBEst = read_results('../data/tpcds5m/DBEst_integral/'+file_name+'.txt')
            
            for group_id in  ground_truth:
                print(group_id)
                if index == 1:  #initialize the error as an empty list, so as to contain further error for the same group
                    re_per_group_DBEst[group_id] = []
                    re_per_group_blinkdb[group_id] =[]

                re_per_group_DBEst[group_id].append(abs(float(predictions_DBEst[group_id])-float(ground_truth[group_id]))/float(ground_truth[group_id]))
                re_per_group_blinkdb[group_id].append(abs(float(predictions_blinkdb[group_id])-float(ground_truth[group_id]))/float(ground_truth[group_id]))

        for group_id in re_per_group_DBEst:
            avg_re_DBEst=sum(re_per_group_DBEst[group_id])/len(re_per_group_DBEst[group_id])
            avg_re_blinkdb=sum(re_per_group_blinkdb[group_id])/len(re_per_group_blinkdb[group_id])

            res_per_group_DBEst[func].append(avg_re_DBEst)
            res_per_group_blinkdb[func].append(avg_re_blinkdb)
        # averag_errors_blinkdb.append(sum(errors_blinkdb)/len(errors_blinkdb))
        # averag_errors_DBEst.append(sum(errors_DBEst)/len(errors_DBEst))
        
       
    print(res_per_group_DBEst)
    print(res_per_group_blinkdb)

    
    
    if group_num == 501:
        plt_histogram(res_per_group_DBEst[function],res_per_group_blinkdb[function])
    if group_num == 8:
        plt_bar(res_per_group_DBEst[function],res_per_group_blinkdb[function],size)

    return res_per_group_DBEst, res_per_group_blinkdb
    

def plt_bar(x1,x2,size="100k"):
    plt.rcParams.update({'font.size': 12})
    width = 0.20
    

    X = np.arange(8)
  
    fig, ax = plt.subplots()

    p1 = plt.bar(X + 0.00, x1, color=colors["DBEst_10k"], width=width,alpha=alpha["2"])
    p2 = plt.bar(X + 0.20, x2, color=colors["BlinkDB_10k"], width=width,alpha=alpha["6"])
    # p3 = plt.bar(X + 0.40, data[2], color=colors["DBEst_100k"], width=width)
    # p4 = plt.bar(X + 0.60, data[3], color=colors["BlinkDB_100k"], width=width)

    if size == "100k":
        plt.legend((p1[0], p2[0]),
            ('DBEst_1o0k', 'BlinkDB_100k','DBEst_100k','BlinkDB_100k'), loc='1')
    if size =="1m":
        plt.legend((p1[0], p2[0]),
            ('DBEst_1m', 'BlinkDB_1m','DBEst_100k','BlinkDB_100k'), loc='1')
    plt.xticks(X+0.5*width, ("1", '2','3','4',"5", '6','7','8'))
    ax.set_ylabel("Relative Error (%)")
    ax.set_xlabel("Group By ID")
    # ax.set_yscale('log')
    formatter = FuncFormatter(to_percent)
    ax.yaxis.set_major_formatter(formatter)
    plt.subplots_adjust(bottom=0.10)
    plt.subplots_adjust(left=0.13)

    plt.show()


def plt_bar4(x1,x2,x3,x4,function="count"):
    plt.rcParams.update({'font.size': 12})
    width = 0.20
    

    X = np.arange(8)
  
    fig, ax = plt.subplots()

    p1 = plt.bar(X + 0.00, x1, color=colors["DBEst_10k"], width=width,alpha=alpha["2"])
    p2 = plt.bar(X + 0.20, x2, color=colors["BlinkDB_10k"], width=width,alpha=alpha["3"])
    p3 = plt.bar(X + 0.40, x3, color=colors["DBEst_100k"], width=width,alpha=alpha["4"])
    p4 = plt.bar(X + 0.60, x4, color=colors["BlinkDB_100k"], width=width,alpha=alpha["6"])

    
    plt.legend((p1[0], p2[0],p3[0], p4[0]),
        ('DBEst_100k', 'BlinkDB_100k','DBEst_1m','BlinkDB_1m'), loc='1')
    
    plt.xticks(X+0.5*width, ("1", '2','3','4',"5", '6','7','8'))
    ax.set_ylabel("Relative Error (%)")
    ax.set_xlabel("Group By ID")
    # ax.set_yscale('log')
    formatter = FuncFormatter(to_percent)
    ax.yaxis.set_major_formatter(formatter)
    plt.subplots_adjust(bottom=0.10)
    plt.subplots_adjust(left=0.13)

    plt.show()



def plt_histogram(x1,x2,b_cumulative=False):

    plt.rcParams.update({'font.size': 12})
    x1.sort()
    x2.sort()

    fig, ax = plt.subplots()
    # p1 = plt.hist(x1,501,normed=1,cumulative=b_cumulative,color=colors["DBEst_1m"],alpha=0.3, label='DBEst')
    # p2 = plt.hist(x2,501,normed=1,cumulative=b_cumulative,color=colors["BlinkDB_1m"],alpha=0.7, label='BlinkDB')

    p1 = plt.hist(x1,100,normed=1,cumulative=b_cumulative,color=colors["DBEst_1m"],alpha=0.3, label='DBEst')
    p2 = plt.hist(x2,100,normed=1,cumulative=b_cumulative,color=colors["BlinkDB_1m"],alpha=0.7, label='DBEst_XGboost')

    plt.legend( loc='1')
    # plt.legend((p1[0], p2[0]),
    #     ('DBEst_10k', 'BlinkDB_10k','DBEst_100k','BlinkDB_100k'), loc='1')

    # plt.xlim(0,0.062)
    # plt.ylim(0,1)
    formatter = FuncFormatter(to_percent)
    ax.xaxis.set_major_formatter(formatter)
    # ax.yaxis.set_major_formatter(formatter)
    plt.xlabel("Relative Error (%)")
    # plt.ylabel("Cumulative Probability")
    plt.ylabel("Number of Occurence")
    plt.subplots_adjust(bottom=0.12)
    plt.subplots_adjust(left=0.16)


    
    plt.show()
    # fig.savefig("/home/u1796377/Desktop/figures/group_by_histgram.pdf", bbox_inches='tight')


if __name__ == '__main__':
    avg_relative_errors_per_group_value(function='avg', group_num=501)
    avg_relative_errors()

    # import numpy as np
    # DBEst_100k, blinkdb_100k = avg_relative_errors_per_group_value(group_num=8,function="count",size="100k")
    # DBEst_1m, blinkdb_1m = avg_relative_errors_per_group_value(group_num=8,function="count",size="1m")
    # function = "count"
    # plt_bar4(DBEst_100k[function],blinkdb_100k[function],DBEst_1m[function], blinkdb_1m[function])
    # print(np.var(DBEst_100k[function]))
    # print(np.var(DBEst_1m[function]))
    # print(np.var(blinkdb_100k[function]))
    # print(np.var(blinkdb_1m[function]))

