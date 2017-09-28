# import a dataset

from sklearn import datasets
iris = datasets.load_iris()

x = iris.data #input
y = iris.target #output

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size =  0.5)
# 0.5 means half of data is in train and half is in test

# using DEcisionTreeClassifier()
from sklearn import tree
my_classifer  = tree.DecisionTreeClassifier()

my_classifer.fit(X_train, y_train)

predictions = my_classifer.predict(X_test)


from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)