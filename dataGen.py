import numpy as np
import matplotlib.pyplot as plt

def sign(x):
    if x>0:
        return 1
    return 0

w = np.random.rand(2,1)*2 -1;
coeff = np.random.random(1)*10 - 5;
x= np.linspace(-10,10);
y= -w[0]/w[1]*x-coeff/w[1];
plt.plot(x,y,'k-');
print(w)
xlabel = np.random.rand(10,2)*20 - 10
ylabel = []
for row in xlabel:
    ylabel.append(np.sign(np.dot(row,w)+coeff)[0]);

ylabel = np.array(ylabel);
data = np.vstack((xlabel[:,0],xlabel[:,1],ylabel[:])).T;
print(data)
np.savetxt('data_labelled.csv',data,delimiter=',');
plt.scatter(xlabel[:,0],xlabel[:,1], c=ylabel);
plt.show();
