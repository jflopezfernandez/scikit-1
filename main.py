
########################################################################################################################
#
#                           MACHINE LEARNING MODULE
#
########################################################################################################################


# Import Decision Tree from SciKit Learn
from sklearn import tree
from sklearn import neighbors
from sklearn import discriminant_analysis

X = [[181, 81, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37],
     [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37],
     [171, 75, 42], [181, 85, 43]]

Y = ['male', 'female', 'female', 'female', 'male', 'male',
     'male', 'female', 'male', 'female', 'male']

cls1 = tree.DecisionTreeClassifier()
cls1 = cls1.fit(X, Y)

cls2 = neighbors.KNeighborsClassifier()
cls2 = cls2.fit(X,Y)

cls3 = discriminant_analysis.QuadraticDiscriminantAnalysis()
cls3 = cls3.fit(X,Y)

prediction1 = cls1.predict([[190, 70, 43]])
prediction2 = cls2.predict([[190, 70, 43]])
prediction3 = cls3.predict([[190, 70, 43]])

print(prediction1)
print(prediction2)
print(prediction3)
