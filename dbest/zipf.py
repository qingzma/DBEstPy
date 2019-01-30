import matplotlib.pyplot as plt
from scipy import special
import numpy as np

def generate():
	a = 1.01 # 2# parameter

	with open("zipfan.csv",mode='w+') as f:
		for index in range(1,101):
			print(str(index)+" million..................")
			x= np.random.uniform(1,100,1000000)
			y = x**(-a) / special.zetac(a)
			x=x.tolist()
			y=y.tolist()

			for xi,yi in zip(x,y):
				# print(str(x)+","+str(y)+"\n")
				f.write(str(xi)+","+str(yi)+"\n")

	print("Finished!")

def plt_zipf():
	a = 1.01 # 2# parameter
	# x= np.random.uniform(1,10,1000)
	x=np.linspace(1, 10,1000)
	y = x**(-a) / special.zetac(a)
	plt.plot(x, y/max(y), linewidth=1, color='r')
	# plt.xlim(1,15)
	plt.show()
# print(y)
# plt.plot(x, y/max(y), linewidth=1, color='r')
# plt.scatter(x, y/max(y), color='r')
# plt.show()


if __name__=='__main__':
	plt_zipf()
	