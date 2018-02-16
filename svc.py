import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import pandas as pd
from matplotlib import style
from random import randint
style.use("ggplot")

#supporting functions
def plotLine(w,c,special=[]):
    """plot our line"""
    m = -w[0]/ w[1]; #slope
    c = -clf.intercept_[0]/w[1]; #intercept
    xx = np.linspace(-12,12)
    yy = m*xx + c;
    fig = plt.figure();
    h0 = plt.plot(xx,yy,'k-', label='non weighted div');
    if len(special) != 0:
        plt.plot(special[0],special[1],'g*');
    plt.scatter(X[:,0], X[:,1], c=y);
    plt.legend();
    plt.draw();
    plt.waitforbuttonpress(0);
    plt.close(fig);
    return fig;

def ClosestToLine(hyperplane,points,intercept):
    """Calculates distance vector from points to hyperplane"""
    dist = abs(np.matmul(points,hyperplane)+intercept);
    return np.argmin(dist);


#points in 2D (for visualization)
#W is our actual classifer
data = pd.read_csv("data_labelled.csv",header=None);
data = np.array(data);
data = data.tolist();
X = []
y = []
#Getting data in required format
for row in data:
    X.append([row[0],row[1]]);
    y.append(row[2]);

#X represents our points and Y their labels
X =np.array(X);
y = np.array(y);
Xinit = X  #To preserve it
yinit = y


#loading unlabelled data
udata = np.array(pd.read_csv("data_unlabelled.csv",header=None))
Xu = []
yu = []
#Getting data in required format
for row in udata:
    Xu.append([row[0],row[1]]);
    yu.append(row[2]);

#X represents our points and Y their labels
Xu =np.array(Xu);
yu = np.array(yu);

score1 = [];
maxIter=10;
i=0;
for i in range(maxIter):
#training our svm classifer
    print("Iteration no: ",end='');
    print(i);
    clf = LinearSVC(random_state=0)
    clf.fit(X,y);
    w = clf.coef_[0];
    print("Hyperplane paramters :"+str(w));
    print("Score :",end='');
    score1.append(clf.score(Xu,yu,sample_weight=None))
    print(score1[-1]);
    #print("TestScore:",end='');
    #print(clf.score(X,y,sample_weight=None),end='\n\n\n');

    closeIdx  =ClosestToLine(clf.coef_[0],Xu,clf.intercept_[0])
    close = Xu[closeIdx]


#plot our line
    plotLine(clf.coef_[0],clf.intercept_[0],close)
#add closest point to our data
    X = X.tolist();
    X.append(close);
    X = np.array(X);
    y = y.tolist();
    y.append(yu[closeIdx]);
    y = np.array(y);

score2 = [];
i=0;
X = Xinit
y = yinit
for i in range(maxIter):
#training our svm classifer
    clfRnd = LinearSVC(random_state=0)
    clfRnd.fit(X,y);
    w = clfRnd.coef_[0];
    score2.append(clfRnd.score(Xu,yu,sample_weight=None))
    print(score2[-1]);

#add random point to our data
    Idx  = randint(0,99);
    print(Idx);
    X = X.tolist();
    X.append(Xu[Idx]);
    X = np.array(X);
    y = y.tolist();
    y.append(yu[Idx]);
    y = np.array(y);

print("Comparisn between random selection of points and closest to hyperplane selection of points :");
print("\n\n\n");

plt.clf();
plt.plot(score1,'r-');
plt.plot(score2,'b-');
plt.ylabel('Score (accuracy)');
plt.xlabel('No of data points added');
plt.show();
