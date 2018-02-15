import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

def sign(x):
    """returns sign of x"""
    if x>0:
        return 1
    return 0

#Create our line randomly
w = np.random.rand(2,1)*2 -1;
coeff = np.random.random(1)*10 - 5;

#plot the line
x= np.linspace(-10,10);
y= -w[0]/w[1]*x-coeff/w[1];
fig = plt.figure()
plt.plot(x,y,'k-');
print(w)

#Generate labelled data
xlabel = np.random.rand(10,2)*20 - 10
ylabel = []
for row in xlabel:
#assigning the labels to our data
    ylabel.append(np.sign(np.dot(row,w)+coeff)[0]);

#change data to required format
ylabel = np.array(ylabel);
data = np.vstack((xlabel[:,0],xlabel[:,1],ylabel[:])).T;
#put data into csv file data_labelled.csv
np.savetxt('data_labelled.csv',data,delimiter=',');
plt.scatter(xlabel[:,0],xlabel[:,1], c=ylabel);

#Generate unlabelled data
xlabel = np.random.rand(100,2)*20 - 10
ylabel = []
for row in xlabel:
#assigning the labels to our data (known only to oracle)
    ylabel.append(np.sign(np.dot(row,w)+coeff)[0]);

#change data to required format
ylabel = np.array(ylabel);
data = np.vstack((xlabel[:,0],xlabel[:,1],ylabel[:])).T;
#put data into csv file data_unlabelled.csv
np.savetxt('data_unlabelled.csv',data,delimiter=',');
print('Done generating data');

#plotting data
plt.draw();
plt.waitforbuttonpress(0)
plt.close(fig)

print('Exitting');
