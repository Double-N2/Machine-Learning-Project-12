import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

data = pd.read_csv("bedtime_screentime_sleep_debt.csv")

# Cleaning the data
data.drop_duplicates()
data.reset_index(drop=True)

# data.info()

# print(data['sleep_debt_category'].unique())

y = data['sleep_debt_category']
x = data.drop('sleep_debt_category', axis=1)

y = y.map({'Severe Sleep Debt': 0,'Mild Deficit':1,'Moderate Debt':2, 'Optimal Recovery': 3}) # Converting strings into numbers

numerical = x.select_dtypes(include='number').columns # splitting data into numerical and characters
character = x.select_dtypes(include=['object','string']).columns

# Splitting the model using a train test split
x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42)

# Using a column transformer
preprocessing = ColumnTransformer(
    [
        (
            "Numerical",
            Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('std_scaler', StandardScaler())]),
            numerical,
        ),
        (
            "Character",
            Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('OneHot', OneHotEncoder(handle_unknown='ignore'))]),
            character,
        )
    ]
)

# Training the various models
decision_tree = Pipeline(steps=[
    ('preprocessor', preprocessing),
    ('decision_tree', DecisionTreeClassifier())
])
random_forest = Pipeline(steps=[
    ('preprocessor', preprocessing),
    ('random_forest', RandomForestClassifier())
])
svm = Pipeline(steps=[
    ('preprocessor', preprocessing),
    ('svm', SVC())
])
knn = Pipeline(steps=[
    ('preprocessor', preprocessing),
    ('knn', KNeighborsClassifier())
])
logistic = Pipeline(steps=[
    ('preprocessor', preprocessing),
    ('logistic', LogisticRegression(max_iter=2000))
])

decision_tree = decision_tree.fit(x_train, y_train)
random_forest = random_forest.fit(x_train, y_train)
svm = svm.fit(x_train, y_train)
knn = knn.fit(x_train, y_train)
logistic = logistic.fit(x_train, y_train)

# Predicting the score
result = \
    {
          0 : 'Severe Sleep Debt',
          1 : 'Mild Deficit',
          2 : 'Moderate Debt',
          3 : 'Optimal Recovery'
    }
# Decision Tree
decision_tree_pred = decision_tree.predict(x_test)
decision_tree_conversion = [result[i] for i in decision_tree_pred]
decision_tree_accuracy = accuracy_score(y_test, decision_tree_pred)

# Random Forest
random_forest_pred = random_forest.predict(x_test)
random_forest_conversion = [result[i] for i in random_forest_pred]
random_forest_accuracy = accuracy_score(y_test, random_forest_pred)

# SVM
svm_pred = svm.predict(x_test)
svm_conversion = [result[i] for i in svm_pred]
svm_accuracy = accuracy_score(y_test, svm_pred)

# KNN
knn_pred = knn.predict(x_test)
knn_conversion = [result[i] for i in knn_pred]
knn_accuracy = accuracy_score(y_test, knn_pred)

# Logistic Regression
logistic_pred = logistic.predict(x_test)
logistic_conversion = [result[i] for i in logistic_pred]
logistic_accuracy = accuracy_score(y_test, logistic_pred)

print("=" * 45)
print("       MACHINE LEARNING MODEL RESULTS")
print("=" * 45)

print(f"Decision Tree Accuracy:    {decision_tree_accuracy * 100:.2f}%")
print(f"Random Forest Accuracy:    {random_forest_accuracy * 100:.2f}%")
print(f"SVM Accuracy:              {svm_accuracy * 100:.2f}%")
print(f"KNN Accuracy:              {knn_accuracy * 100:.2f}%")
print(f"Logistic Regression:       {logistic_accuracy * 100:.2f}%")

print("=" * 45)
print("\n========== MODEL PREDICTIONS ==========")

print("\nDecision Tree:")
print(decision_tree_conversion)

print("\nRandom Forest:")
print(random_forest_conversion)

print("\nSVM:")
print(svm_conversion)

print("\nKNN:")
print(knn_conversion)

print("\nLogistic Regression:")
print(logistic_conversion)