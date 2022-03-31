import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,roc_curve
from IPython.display import display
from sklearn.model_selection import cross_val_score
import category_encoders as ce
traindata=pandas.read_csv("Traindata.csv")

X = traindata.drop(['class'], axis=1)
y = traindata['class']
X_train, X_test, y_train, y_test=train_test_split(X,y,train_size=0.8, test_size=0.2, random_state=10)
X_test_original=X_test.copy(deep=False)
X_train_original=X_train.copy(deep=False)



encoder = ce.OrdinalEncoder(cols=['age', 'job', 'marital', 'education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome'])


X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)
clf = DecisionTreeClassifier(max_depth=3,random_state=0) #max_depth is maximum number of levels in the tree
clf.fit(X_train,y_train)
tree.plot_tree(clf.fit(X_train, y_train)) 
