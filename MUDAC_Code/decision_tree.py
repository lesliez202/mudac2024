import pandas as pd
from sklearn import datasets
import sklearn.tree as skt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics


diabetes = datasets.load_diabetes()
df = pd.DataFrame(diabetes.data, columns = diabetes.feature_names)
df['target'] = diabetes.target

y = np.array(df['target'])
X = np.array(df[['age', 'sex', 'bmi']])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

tree = skt.DecisionTreeClassifier()

result = tree.fit(X_train, y_train)

print(tree.feature_importances_)
print(tree.predict(X_test))

#MSE section graphed

new_df = pd.DataFrame()
new_df['Yay'] = ['One', 'Two', 'Zero', 'Five']
val = pd.get_dummies(new_df, drop_first=True)
print(val)



#tools
'''
ml, logistic regression(can derive CI sklearn)
rfdecision trees
linear regression
support vector 

wrangle missing data and document
prune outliers
standardize data
corr tables


Visualizations
scatter
bar

'''