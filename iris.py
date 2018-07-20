# Import Prepared Data

import sklearn.datasets as datasets

iris = datasets.load_iris()
features = iris.feature_names
data = iris.data
target = iris.target


# Generate Model

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(pd.DataFrame(data, columns=features), target)


# Plot Tree Graph

import pydotplus
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz

dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data, feature_names=features, precision=2, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('output/tree.png')


# Modal Data Export using Graphviz

export_graphviz(dtree, out_file='output/model.dot', feature_names=features, precision=10, special_characters=True)


# Output Production code

from sklearn_porter import Porter

porter = Porter(dtree, language='java')
output = porter.export(embed_data=True)
file = open('output/code.java', 'w')
file.write(output)
file.close()
