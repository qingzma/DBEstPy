#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function, division
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
# from functools import reduce
import pickle
from datetime import datetime

from dbest import logs


class Confidence:
    def __init__(self,file,headerX,headerY, qreg=None,bucket_size=10,data_size=2685596178):
        self.data=pd.read_csv(file)
        self.data.dropna(subset=[headerX,headerY])
        self.data=self.data[~self.data[headerX].isin(["nan"])]
        self.data=self.data[~self.data[headerY].isin(["nan"])]
        self.x=self.data[[headerX]].values.reshape(-1)
        self.x=sorted(self.x)
        self.y=self.data[[headerY]].values.reshape(-1)
        self.headerX=headerX
        self.headerY=headerY
        self.qreg=qreg
        self.bucket_size=bucket_size
        self.data_size=2685596178
        self.sample_size=len(self.x)
        # self.x=[1,1,1.1,2,2,2,3,3,4,5,5]
        # self.y=[1,1,1.1,2,2,2,3,3,4,5,5]
        # self.x=np.linspace(0,100,1000)
        # self.y=np.linspace(0,200,1000)

    def fit_count(self):
        # backet_size=2
        self.x, self.y = zip(*sorted(zip(self.x, self.y)))
        # print(self.x)
        # print(self.y)
        self.buckets_count=generate_bucket_count(self.x, self.y, bucket_size=self.bucket_size)
        # self.buckets_count.print()

    def predict_count_point(self,x):
        index_left=0
        index_right=len(self.buckets_count.buckets)-1
        index = int(len(self.buckets_count.buckets)/2)
        loop_count=0
        index_last=5
        while True:
            loop_count+=1

            #check if within the last buckets
            if self.buckets_count.buckets[len(self.buckets_count.buckets)-1].interval.if_contains(x)==0:
                index=len(self.buckets_count.buckets)-1
                return self.buckets_count.buckets[index].count,index,None, None
            else:

                # print(index)

                if self.buckets_count.buckets[index].interval.if_contains(x)==0:
                    # print(self.buckets_count.buckets[index].name)
                    return self.buckets_count.buckets[index].count,index,None, None
                    
                # less 
                elif self.buckets_count.buckets[index].interval.if_contains(x)==-1:
                    # print("less")
                    index_right=index
                    index=int((index_left+index_right)/2)
                # higher
                else:
                    # print("higher")
                    index_left=index
                    index=int((index_left+index_right)/2)
                    # print(index)
                    # print(index_right)
                    index=min(index,len(self.x)-1)
                    # print(index)

                if index_last==index:
                    # print("Info---No records! return None!")
                    return None, None,self.buckets_count.buckets[index].count,index
                

                if loop_count>200:
                    print("maximum loop reached!")
                    return None, None,self.buckets_count.buckets[index].count,index  #self.buckets_count.buckets[index].count,index
                index_last=index



    def predict_count_range(self,x1,x2):
        count_low,index_low,count_low_right,index_low_right=self.predict_count_point(x1)
        if count_low==None:
            count_low=0
        if x1==x2:
            if self.buckets_count.buckets[index_low].unique:
                return count_low, 0.00
            else:
                return count_low/2, 0.50
        
        count_high,index_high,count_high_right,index_high_right=self.predict_count_point(x2)
        if count_high ==None:
            count_high=0
        # print(index_low)
        # print(index_high)
        counts=0
        error=0
        if index_low is not None:
            if self.buckets_count.buckets[index_low].unique:
                counts=count_low
                error=0
            else:
                counts+=count_low/2
                error+=count_low/2
        else: 
            counts+=0
            error+=0
            index_low = index_low_right 

        if index_high is not None:
            if self.buckets_count.buckets[index_high].unique:
                counts+=count_high
                error=0
            else:
                counts+=count_high/2
                error+=count_high/2
        else: 
            counts+=0
            error+=0
            index_high=index_high_right

        


        for index in range(index_low+1,index_high):
            counts+= self.buckets_count.buckets[index].count

        return int(counts*self.data_size/self.sample_size), error/counts
            










        
    def fit(self):
        n_division=2 # int(len(self.x)/10)
        self.n, self.bins, patches = plt.hist(self.x, histedges_equalN(self.x, n_division))
        print(self.n)
        print(self.bins)
        return
        # plt.show()
    def get_size(self):
        data = self.data
        x=self.x
        y=self.y

        del self.data
        del self.x
        del self.y
        str_size=pickle.dumps(self)

        self.data = data
        self.x = x
        self.y = y

        return sys.getsizeof(str_size)
    def predict(self,x):
        for index in range(len(self.bins)-1):
            # print(self.bins[index])
            if self.bins[index] <= x  and self.bins[index+1] >= x :
                return self.cummulative_y[index]
        return  self.cummulative_y[0]
         
         
    
    def ci_point(self, x,func='count', confidence=0.95):
        time_start = datetime.now()
        if func == 'count':
            y=self.n
        if func == 'sum':
            pass

        self.cummulative_y=to_cummulative(y)
        


        self.num_training_points_model = len(self.x)
        self.averageX_training_points_model = sum(self.x)/self.num_training_points_model
        self.variance_training_points_model=np.var(self.cummulative_y)
        t = stats.t.ppf(confidence, max(self.num_training_points_model - 2,1))
        s = self.variance_training_points_model**0.5
        tmp = (1 / self.num_training_points_model + (x - self.averageX_training_points_model)
               ** 2 / (self.num_training_points_model - 1) / self.variance_training_points_model)**0.5
        time_end= datetime.now()
        time_cost =(time_end-time_start).microseconds
        return t * s * tmp,time_cost
    def ci_range(self, low, high, func,confidence=0.95):
        ci1,t1=self.ci_point(low,func,confidence)
        ci2,t2=self.ci_point(high,func,confidence)
        return ci2+ci1,t1+t2


    def plt_cummulative(self):
        plt.clf()
        plt.plot(self.bins[:-1], self.cummulative_y)
        # x = np.linspace(min(self.bins[:-1]),max(self.bins[:-1]),100)
        maxs=[yy+self.ci_point(xx,func='count')[0] for xx, yy in zip(self.bins, self.cummulative_y)]
        mins=[2*avgi - maxi for avgi, maxi in zip(self.cummulative_y,maxs)]
        plt.plot(self.bins[:-1], self.cummulative_y)
        plt.plot(self.bins[:-1] , mins)
        plt.plot(self.bins[:-1] , maxs)

        plt.xlabel(self.headerX)
        plt.ylabel("Count of "+self.headerY)
        plt.title("Cummulative count")
        plt.show()

class Interval:
    def __init__(self,left, right):
        self.left=left
        self.right=right
    def set_left(self, left):
        self.left=left
    def set_right(self, right):
        self.right=right
    def name(self):
        return str('['+str(self.left)+', '+str(self.right)+')')
    def if_contains(self, x):

        if x>=self.left and x<self.right:
            return 0
        if self.left == self.right and self.left ==x:
            return 0
        if x<self.left:
            return -1
        if x>=self.right:
            return 1
        
class Bucket:
    def __init__(self,interval_i, count_i,sum_i, unique=False):
        self.interval=interval_i
        self.count=count_i
        self.sum=sum_i
        self.name=interval_i.name()
        self.unique=unique


class Buckets:
    def __init__(self):
        self.buckets=[]
    def add_bucket(self,bucket):
        if bucket not in self.buckets:
            self.buckets.append(bucket)
        else:
            print('bucket exists! skip adding!')
    def print(self):
        print('-------------------------------------------------------')
        print('Buckets contains {} buckets'.format(len(self.buckets)))
        for i in range(len(self.buckets)):
            print('Interval: {},     count: {},     sum: {}.'.format(self.buckets[i].name, self.buckets[i].count, self.buckets[i].sum))






def generate_bucket_count(x,y,bucket_size):
    if len(x) != len(y):
        print('size mismatch in generate_bucket(), quit!')
        return

    count_not_full=0
    count_unique=0
    b_last_is_unique=False

    buckets = Buckets()


    if len(x)<=bucket_size:
        # print('Only one bucket is needed to hold all data!')
        buckets.add_bucket(Interval(min(x),max(x)),len(x),sum(x))
        
    index_low=0
    index_high=bucket_size-1
    while index_high < len(x)-1:
        b_unique=False
        #check if bucket contains only unique x
        if x[index_low] == x[index_high]:
            b_unique=True
            count_unique +=1
            index_high+=1
            while x[index_low] == x[index_high]:
                index_high+=1
                if  index_high ==len(x):
                    b_last_is_unique=True
                    break
            index_high-=1

        # check if x should go to the next bucket
        elif x[index_high] == x[index_high+1]:
            index_high-=1
            while x[index_high] == x[index_high+1]:
                index_high-=1
            # print('bucket shrinks, low and high are [{},{})'.format(x[index_low],x[index_high]))
            count_not_full+=1
  
        # return
        if  b_last_is_unique:
            # index_low=index_high+1
            # print("insert last unique bucket")
            index_high=index_low+bucket_size-1
            # print(index_low)
            # print(index_high)
            interval = Interval(x[index_low], x[len(x)-1] )
            bucket = Bucket(interval,len(x)-index_low,sum(y[index_low:len(x)]),b_unique)
            buckets.add_bucket(bucket)
            break
        else:
            interval = Interval(x[index_low], x[index_high] )
            bucket = Bucket(interval,index_high-index_low+1,sum(y[index_low:index_high+1]),b_unique)
            buckets.add_bucket(bucket)

            #update the index, for next loop
            index_low=index_high+1
            index_high=index_low+bucket_size-1
    if not b_last_is_unique: 
        # print("insert last simple bucket")
        interval = Interval(x[index_low], x[len(x)-1] )
        bucket = Bucket(interval,len(x)-index_low,sum(y[index_low:len(x)]),b_unique)
        buckets.add_bucket(bucket)
    return buckets
    
def generate_bucket_sum(x,y,max_proportion=0.02):
    if len(x) != len(y):
        print('size mismatch in generate_bucket(), quit!')
        return

    count_not_full=0
    count_unique=0
    b_last_is_unique=False

    buckets = Buckets()
    total_sum=sum(y)
    bucket_max_sum=total_sum*max_proportion


    if len(x)<=bucket_size:
        print('Only one bucket is needed to hold all data!')
        buckets.add_bucket(Interval(min(x),max(x)),len(x),sum(x))
        
    index_low=0
    index_high=bucket_size-1
    while index_high < len(x)-1:
        #check if bucket contains only unique x
        if x[index_low] == x[index_high]:
            count_unique +=1
            index_high+=1
            while x[index_low] == x[index_high]:
                index_high+=1
                if  index_high ==len(x):
                    b_last_is_unique=True
                    break
            index_high-=1

        # check if x should go to the next bucket
        elif x[index_high] == x[index_high+1]:
            index_high-=1
            while x[index_high] == x[index_high+1]:
                index_high-=1
            print('bucket shrinks, low and high are [{},{})'.format(x[index_low],x[index_high]))
            count_not_full+=1
  
        # return
        if  b_last_is_unique:
            # index_low=index_high+1
            # print("insert last unique bucket")
            index_high=index_low+bucket_size-1
            # print(index_low)
            # print(index_high)
            interval = Interval(x[index_low], x[len(x)-1] )
            bucket = Bucket(interval,len(x)-index_low,sum(y[index_low:len(x)]))
            buckets.add_bucket(bucket)
            break
        else:
            interval = Interval(x[index_low], x[index_high] )
            bucket = Bucket(interval,index_high-index_low+1,sum(y[index_low:index_high+1]))
            buckets.add_bucket(bucket)

            #update the index, for next loop
            index_low=index_high+1
            index_high=index_low+bucket_size-1
    if not b_last_is_unique: 
        print("insert last simple bucket")
        interval = Interval(x[index_low], x[len(x)-1] )
        bucket = Bucket(interval,len(x)-index_low,sum(y[index_low:len(x)]))
        buckets.add_bucket(bucket)


    

    buckets.print()

    # print(index_low)
    # print(len(x)-1)       
    # print(bucket.name)
    # print(bucket.count)
    # print(bucket.sum)
    # print('---------')



    
    



    # for index in range(1,len(x)-1):
    #     if 




def to_cummulative(y):
    cummulative_y=[]
    for index in range(len(y)):
        cummulative_y.append(sum(y[:index+1]))
    return cummulative_y


def histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))
def run():
    confidence =  Confidence(file='../data/tpcDs10k/store_sales.csv',headerX='ss_quantity',headerY='ss_ext_sales_price')
    confidence.fit()

    print('Memory Overheads for this model: {} (bytes).'.format(confidence.get_size()))
    print('----------------------------------------------')
    confidence.ci_point(20)
    print('Point query: select count(y) from table where x<20 , exact answer {}'.format(confidence.predict(20)))
    print('confidence interval for point 20 (count):  {}'.format(confidence.ci_point(20)[0]))
    # print(confidence.ci_point(20)[0])
    print("Response Time: {} microseconds.".format(confidence.ci_point(20)[1]))
    print('----------------------------------------------')
    print('confidence interval for range [20,60] (count): {}'.format(confidence.ci_range(20,60,func='count')[0]))
    print("Response Time: {} microseconds.".format(confidence.ci_range(20,60,func='count')[1]))
    print('----------------------------------------------')

    # confidence.plt_cummulative()

    # print('Response Time: {} microseconds.'.format(confidence.ci_point(20)[1])
    # print('----------------------------------------------')
    # print('confidence interval for range [20,60] (count) {}'.format(confidence.ci_range(20,60,func='count')))
    # print("Response Time: {} microseconds.".format(confidence.ci_range(20,60,func='count')[1]))
    # print('----------------------------------------------')
    # print(to_cummulative([1,2,3,4,0]))
def run_tpcds():
    confidence =  Confidence(file='../data/tpcDs100k/store_sales.csv',headerX='ss_quantity',headerY='ss_list_price',bucket_size=10)
    confidence.fit_count()

    print(confidence.predict_count_range(0    , 5))
    print(confidence.predict_count_range(6    , 10))
    print(confidence.predict_count_range(11   , 15))
    print(confidence.predict_count_range(16   , 20))
    print(confidence.predict_count_range(21   , 25))
    print(confidence.predict_count_range(26   , 30))
    confidence =  Confidence(file='../data/tpcDs100k/store_sales.csv',headerX='ss_list_price',headerY='ss_list_price',bucket_size=10)
    confidence.fit_count()
    print(confidence.predict_count_range(90 ,100 ))
    print(confidence.predict_count_range(70 ,80   ))
    print(confidence.predict_count_range(80 ,90  ))
    print(confidence.predict_count_range(100, 110))
    print(confidence.predict_count_range(110, 120))
    print(confidence.predict_count_range(120, 130))

    confidence =  Confidence(file='../data/tpcDs100k/store_sales.csv',headerX='ss_coupon_amt',headerY='ss_list_price',bucket_size=10)
    confidence.fit_count()
    print(confidence.predict_count_range(7000    , 8000  ))
    print(confidence.predict_count_range(8000    , 9000  ))
    print(confidence.predict_count_range(9000    , 10000))
    print(confidence.predict_count_range(10000   , 11000))
    print(confidence.predict_count_range(11000   , 12000))
    print(confidence.predict_count_range(12000   , 13000))


    confidence =  Confidence(file='../data/tpcDs100k/store_sales.csv',headerX='ss_wholesale_cost',headerY='ss_list_price',bucket_size=10)
    confidence.fit_count()
    print(confidence.predict_count_range(10     , 30))
    print(confidence.predict_count_range(20     , 40))
    print(confidence.predict_count_range(30     , 50))
    print(confidence.predict_count_range(40     , 60))
    print(confidence.predict_count_range(50     , 70))
    print(confidence.predict_count_range(60     , 80))

if __name__=="__main__":
    run_tpcds()
    # run()
    

    