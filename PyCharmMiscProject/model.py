import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
from sklearn.preprocessing import LabelEncoder
import pickle
warnings.filterwarnings('ignore')
df=pd.read_csv('dataset/patient_data.csv')
df.head()
df.rename(columns={"C":"Gender"},inplace=True)
df.head()
df.shape
df.info()
for col in df.columns:
  print(f"Unique values in {col}: ")
  print(df[col].unique())
  df['TakeMedication'].replace(to_replace={'Yes ': 'Yes'}, inplace=True)
  df['TakeMedication'].unique()
df['NoseBleeding'].replace(to_replace={'No ':'No'},inplace=True)
df.NoseBleeding.unique()
df['Systolic'].replace(to_replace={'121- 130':'121 - 130'},inplace=True)
df['Systolic'].unique()

df['Stages'].replace(to_replace={'HYPERTENSIVE CRISI':'HYPERTENSIVE CRISIS', 'HYPERTENSION (Stage-2).':'HYPERTENSION (Stage-2)'},inplace=True)
df.Stages.unique()

df.isnull().sum()
df['Stages'].value_counts()
from sklearn.preprocessing import LabelEncoder
import pickle

label_encoders = {}
target_encoder = LabelEncoder()

# Encoding X columns (all except 'Stages')
for col in df.columns:
    if col != 'Stages' and df[col].dtype == 'object':
        df[col] = df[col].str.strip()  # clean strings
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Encoding target column separately
df['Stages'] = df['Stages'].str.strip()
df['Stages'] = target_encoder.fit_transform(df['Stages'])

# Saving encoders
with open('encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

with open('stage_encoder.pkl', 'wb') as f:
    pickle.dump(target_encoder, f)

df.describe()
gender_counts=df['Gender'].value_counts()

#Plotting the pie chart
plt.figure(figsize=(6,6))
plt.pie(gender_counts,labels=gender_counts.index,autopct='%1.1f%%',startangle=90)
plt.title('Gender Distribution')
plt.show()

frequency=df['Stages'].value_counts()

frequency.plot(kind='bar',figsize=(10,6))
plt.title('Frequency of Stages')
plt.xlabel('Stages')
plt.ylabel('Frequency')
plt.show()

frequency=df['Stages'].value_counts()

frequency.plot(kind='bar',figsize=(10,6))
plt.title('Frequency of Stages')
plt.xlabel('Stages')
plt.ylabel('Frequency')
plt.show()
sns.pairplot(df[['Age', 'Systolic', 'Diastolic']])
plt.show()


sns.countplot(x='TakeMedication', hue='Severity',data=df)
plt.title('Medication vs Severity')
plt.xlabel('Take Medication')
plt.ylabel('Count')
plt.show()

sns.countplot(x='TakeMedication', hue='Severity',data=df)
plt.title('Medication vs Severity')
plt.xlabel('Take Medication')
plt.ylabel('Count')
plt.show()

X=df.drop('Stages',axis=1)
y=df['Stages']

X
y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from imblearn.over_sampling import SMOTE

# SMOTE balancing(Handling Imbalanced Data)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Visualizing class distribution after balancing
decoded_y_res = target_encoder.inverse_transform(y_train_res)
plt.figure(figsize=(10, 4))
sns.countplot(x=decoded_y_res, order=target_encoder.classes_)
plt.title("After SMOTE Balancing")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingClassifier

pipeline = Pipeline(steps=[('imputation', SimpleImputer(strategy='median')), ('scaling', StandardScaler())])
preprocessor = ColumnTransformer(transformers=[('num_pipeline', pipeline, X.columns)])  # as all cols are numerical
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.metrics import accuracy_score,classification_report

models = {'Logistic Regression': LogisticRegression(),
          'Random Forest Classifier': RandomForestClassifier(),
          'Decision Tree Classifier': DecisionTreeClassifier(),
          'GaussianNB': GaussianNB(),
          'MultinomialNB ': MultinomialNB(),
          'support vector machine': SVC(),
          'Decision Tree Classifier': DecisionTreeClassifier(),
          }

models

def model_train_eval(X_train, y_train, X_test, y_test, models):
    evaluation = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            evaluation[name] = accuracy_score(y_test, y_pred)
        except ValueError as e:          # catches the NB error (or any other)
            print(f"⚠️  Skipped {name}: {e}")
            evaluation[name] = None
    return evaluation

results = model_train_eval(X_train, y_train, X_test, y_test, models)
results

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(class_weight='balanced', random_state=42)
# using gridsearchCV doing Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV

params = {'max_depth': [1, 2, 3, 5, 10, None],
          'n_estimators': [50, 100, 200, 300],
          'criterion': ['gini', 'entropy']}

clf = RandomizedSearchCV(rf, param_distributions=params, cv=5, n_iter=10, verbose=3)

clf.fit(X_train, y_train)

clf.best_params_
clf.best_score_
clf=clf.best_estimator_
clf
pickle.dump(clf, open('model.pkl', 'wb'))

df.Gender
df.Age

df.Systolic.sample(15)

df.Stages.sample(10)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Example: assume X and y are already defined
# X = your features (e.g., DataFrame or numpy array)
# y = your labels

# 1. Split the dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train the model
logistic_regression = LogisticRegression()
logistic_regression.fit(x_train, y_train)

# 3. Make predictions
y_pred = logistic_regression.predict(x_test)

# 4. Evaluate performance
acc_lr = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy:", acc_lr)
print("Classification Report:\n", classification_report(y_test, y_pred))
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Example dataset (replace with your actual data)
# X = your feature matrix
# y = your labels

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
mNB = MultinomialNB()
mNB.fit(x_train, y_train)

# Predict
y_predNB = mNB.predict(x_test)

# Accuracy
acc_mnb = accuracy_score(y_test, y_predNB)
print("MultinomialNB Accuracy:", acc_mnb)
from sklearn.naive_bayes import GaussianNB

NB = GaussianNB()

NB.fit(x_train, y_train)

y_pred = NB.predict(x_test)

acc_nb = accuracy_score(y_test,y_pred)

c_nb = classification_report(y_test,y_pred)

print('Accuracy Score: ',acc_nb)

print(c_nb)
from sklearn.naive_bayes import MultinomialNB

MNB = MultinomialNB()

MNB.fit(x_train, y_train)

y_pred = NB.predict(x_test)

acc_mnb = accuracy_score(y_test,y_pred)

c_mnb = classification_report(y_test,y_pred)

print('Accuracy Score: ',acc_mnb)

print(c_mnb)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load example data
data = load_iris()
X = data.data
y = data.target

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train decision tree model
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(x_train, y_train)

# Predict and evaluate
y_pred = decision_tree_model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
# 1. Import required libraries
from sklearn.datasets import load_iris  # You can replace this with your own dataset
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 2. Load example dataset (Iris dataset used here)
data = load_iris()
X = data.data         # Features
y = data.target       # Labels

# 3. Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create and train the Decision Tree Classifier
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(x_train, y_train)

# 5. Make predictions on the test set
y_pred = decision_tree_model.predict(x_test)

# 6. Evaluate the model
acc = accuracy_score(y_test, y_pred)
print("Decision Tree Accuracy:", acc)

# 7. Display classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# 1. Import libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# 2. Example data — replace this with your dataset
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target

# 3. Train/test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Define and train the Random Forest model
random_forest = RandomForestClassifier()
random_forest.fit(x_train, y_train)

# 5. Make a prediction with sample input (must match number of features)
sample_input = [[5.1, 3.5, 1.4, 0.2]]  # change this to your input format
prediction = random_forest.predict(sample_input)

# 6. Show the prediction
print("Prediction:", prediction[0])

import pandas as pd

# Assuming these accuracy scores are already computed
# Replace these with your actual accuracy values
acc_lr = 0.835616
acc_dt = 1.0
acc_rf = 1.0
acc_nb = 0.676712
acc_mnb = 0.676712

# Create DataFrame for comparison
model = pd.DataFrame({
    'Model': [
        'Linear Regression',
        'Decision Tree Classifier',
        'RandomForest Classifier',
        'Gaussian Naive Bayes',
        'Multinomial Naive Bayes'
    ],
    'Score': [acc_lr, acc_dt, acc_rf, acc_nb, acc_mnb]
})

# Display the comparison
print(model)
pickle.dump(clf,open('model.pkl','wb'))
categorical_columns = [col for col in df.columns if df[col].dtype == 'object' and col != 'Stages']

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

with open('encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

