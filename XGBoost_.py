import pandas as pd

from sklearn.metrics import accuracy_score


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier

class drug_type_prediction:
    def __init__(self):
        self.load_dataset()
        self.map_lable()
        self.test_train_split()
        self.fit_model()
    def load_dataset(self):
        self.data = pd.read_csv("drug_data.csv")

    def map_lable(self):
        le = LabelEncoder()

        self.data["Drug"] = self.data["Drug"].str[-1]
        self.data["Drug"] = le.fit_transform(self.data["Drug"])
        self.data["BP"] = le.fit_transform(self.data["BP"])
        self.data["Cholesterol"] = le.fit_transform(self.data["Cholesterol"])
        self.data["Sex"] = le.fit_transform(self.data["Sex"])

    def plot_counter(self, column_name):

        sns.countplot(x=column_name, data=self.data)
        plt.show()

    def print_correlation_matrix(self):
        plt.figure(figsize=(14, 12))
        foo = sns.heatmap(self.data.corr(), vmax=0.6, square=True, annot=True)
        plt.show()

    def test_train_split(self):
        X = self.data.loc[:, self.data.columns != 'Drug']

        y = self.data["Drug"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.33, random_state=1)


    def fit_model(self):
        model = XGBClassifier()
        both_scoring = {'Accuracy': make_scorer(accuracy_score), 'Loss': 'neg_log_loss'}
        params = {
            'n_estimators': [100, 200, 500, 1000, 1500],
            'learning_rate': [0.05, 0.1, 0.2],
        }
        self.clf = GridSearchCV(model, params, cv=5, scoring=both_scoring, refit="Accuracy", return_train_score=True)

        self.clf.fit(self.X_train, self.y_train)
        print((self.clf.best_score_, self.clf.best_params_))
        print("=" * 30)

        print("Grid scores on training data:")

        log_losses = self.clf.cv_results_['std_test_Loss']


        for  log_loss, params in zip(log_losses, self.clf.cv_results_['params']):
            print(" Log Loss: %0.3f for %r" % ( log_loss, params))

    def test(self):
        predictions = self.clf.predict(self.X_test).astype(int)
        self.test_accuracy = accuracy_score(self.y_test, predictions)

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)
one = drug_type_prediction()
one.plot_counter("Drug")
one.print_correlation_matrix()
one.test()
print(one.test_accuracy)