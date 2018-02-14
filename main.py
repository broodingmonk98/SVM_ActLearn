import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
from matplotlib import style
style.use("ggplot")

#points in 2D (for visualization)
#W is our actual classifer
#W = np.random.random(3);
data = pd.read_csv("data.csv");
data = np.array(data);
data = data.tolist();
X = []
y = []
for row in data:
    X.append([row[0],row[1]]);
    y.append(row[2]);

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

a = -w[0]/ w[1];

xx = np.linspace(-12,12)
yy = a*xx - clf.intercept_[0]/w[1]
h0 = plt.plot(xx,yy,'k-', label='non weighted div');
plt.scatter(X[:,0], X[:,1], c=y);
plt.legend();
plt.show();
