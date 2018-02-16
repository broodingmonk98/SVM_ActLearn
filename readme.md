The dataGen.py file generates 10 points [x1,x2] pairs along with associated labels and stores them in data_labelled.csv.
It also generates 100 points [x1,x2] along with labels and stores it in data_unlabelled.csv.

Svc first trains on the labelled data. It then finds the closest point to our hyperplane (line) and adds it to its train data and retrains. It does this 10  times.

In the end we plot a graph of score (accuracy on test data set) of our nearest point to hyperplane strategy vs randomly selecting a point and adding it to our train set strategy. Note the blue line represents the latter and the red line the former of the two strategies.

Note: While svc.py is running, keep clicking on the graph to make it close and to generate the next graph. The last graph (the comparison graph) should be closed using CTRL-W.
