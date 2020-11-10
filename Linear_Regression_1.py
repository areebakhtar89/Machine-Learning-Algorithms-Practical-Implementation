import matplotlib.pyplot as plt
from scipy import stats

X = [5,7,8,7,2,17,2,9,4,11,12,9,6]
Y = [96,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err  = stats.linregress(X,Y)

def myfunc(x):
    return(slope*x+intercept)

mymodel = list(map(myfunc,X))

plt.scatter(X,Y)
plt.plot(X, mymodel)
plt.show()