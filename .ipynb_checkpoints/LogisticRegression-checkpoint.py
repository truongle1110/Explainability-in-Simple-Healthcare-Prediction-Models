import pandas as pd
# from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('dataset/diabetes.csv')

target = 'Outcome'
x = data.drop(target, axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LogisticRegression(random_state=100, max_iter=1000)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
for i,j in zip(y_predict, y_test):
    print("Predict: {}. Actual: {}".format(i,j))
print(classification_report(y_test, y_predict))
print(confusion_matrix(y_test, y_predict))

cm = np.array(confusion_matrix(y_test, y_predict))
confusion = pd.DataFrame(cm, index=['negative','positive'], columns=['negative','positive'])
sns.heatmap(confusion, annot=True)
plt.show()