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
from shap_explainability import apply_shap

data = pd.read_csv('dataset/diabetes.csv')
# profile = ProfileReport(data, title='Diabetes Report',explorative=True)
# profile.to_file('report.html')
print(data)
#split data
target = 'Outcome'
x = data.drop(target, axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#preprocessing
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

params = {
    'n_estimators': [50, 100, 200],
    'criterion': ['gini', 'entropy','log_loss'],
    'max_depth': [None, 2, 5],
    'min_samples_split': [2, 5, 10]
}

#model
# model = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=100)
model = GridSearchCV(RandomForestClassifier(random_state = 100), param_grid=params, scoring='precision',cv=6, verbose=2, n_jobs = 6)
model.fit(x_train, y_train)
print(model.best_score_)
print(model.best_params_)

y_predict = model.predict(x_test)
for i,j in zip(y_predict, y_test):
    print("Predict: {}. Actual: {}".format(i,j))
print(classification_report(y_test, y_predict))
print(confusion_matrix(y_test, y_predict))

# Optional SHAP explainability for the positive diabetes class.
# This works with GridSearchCV because apply_shap uses model.best_estimator_.
# Uncomment after installing shap from dataset_processing/requirements.txt.
# shap_result = apply_shap(
#     model,
#     x_train,
#     x_test,
#     feature_names=x.columns,
#     output_dir='shap_outputs/random_forest',
#     plot_name='random_forest',
# )
# print(shap_result['feature_importance'])

cm = np.array(confusion_matrix(y_test, y_predict))
confusion = pd.DataFrame(cm, index=['negative','positive'], columns=['negative','positive'])
sns.heatmap(confusion, annot=True)
plt.show()
