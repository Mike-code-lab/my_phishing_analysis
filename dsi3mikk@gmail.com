     import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
data = pd.read_csv('https://raw.githubusercontent.com/kdemertzis/EKPA/main/Data/phishing_dataset.csv', delimiter=',')

data = data.sample(frac=1).reset_index(drop=True)
data.head()

X = data.drop(['Domain', 'Label'], axis=1)

X.head()

#y=np.ravel(data['Label'])
y=np.ravel(data['Outcome'])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=43)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define classifiers
classifiers = [
    LogisticRegression(),
    SVC(),
    RandomForestClassifier(),
    MLPClassifier(),
    KNeighborsClassifier(),
    DecisionTreeClassifier()
]

# Evaluate and compare classifiers
results = {'Classifier': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': []}

for clf in classifiers:
    clf_name = clf.__class__.__name__

    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Calculate evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Display results
    print(f"\n{clf_name} Metrics:")
    print(f"Accuracy: {acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print("Confusion Matrix:")
    print(cm)

    # Store results for comparison
    results['Classifier'].append(clf_name)
    results['Accuracy'].append(acc)
    results['Precision'].append(precision)
    results['Recall'].append(recall)
    results['F1 Score'].append(f1)

# Display comparison results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='Accuracy', ascending=False)
print("\nComparison of Classifiers:")
print(results_df)
