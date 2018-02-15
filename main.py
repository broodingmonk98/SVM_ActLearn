import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
from matplotlib import style
style.use("ggplot")

#supporting functions
def plotLine(w,c):
    """plot our line"""
    m = -w[0]/ w[1]; #slope
    c = -clf.intercept_[0]/w[1]; #intercept
    xx = np.linspace(-12,12)
    yy = m*xx + c;
    fig = plt.figure();
    h0 = plt.plot(xx,yy,'k-', label='non weighted div');
    plt.scatter(X[:,0], X[:,1], c=y);
    plt.legend();
    plt.draw();
    plt.waitforbuttonpress(0);
    plt.close(fig);

def ClosestToLine(hyperplane,points):
    """Calculates distance vector from points to hyperplane"""
    dist = abs(np.dot(line,point));
    return np.argmin(dist);

def Error(hyperplane,points):
    """Calculates error on test data"""



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
clf = svm.SVC(kernel='linear', C=1.0);
clf.fit(X,y);

X0 = [[0.58,0.96]];
X1 = [[10.58,10.96]];
print(clf.predict(X0));
print(clf.predict(X1));

w = clf.coef_[0];
print(w);
#plot our line

plotLine(clf.coef_[0],clf.intercept_[0])


