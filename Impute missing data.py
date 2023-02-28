from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()

# Simulate missing data by setting some values to NaN
X = iris.data
missing_mask = np.random.random(X.shape) < 0.1  # 10% missing values
X[missing_mask] = np.nan

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, iris.target, test_size=0.3, random_state=42)

# Impute missing values using mean imputation
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Train a KNN classifier on the original dataset and test on the imputed dataset
knn_orig = KNeighborsClassifier()
knn_orig.fit(X_train, y_train)
y_pred_orig = knn_orig.predict(X_test_imputed)
acc_orig = accuracy_score(y_test, y_pred_orig)

# Train a KNN classifier on the imputed dataset and test on the imputed dataset
knn_imputed = KNeighborsClassifier()
knn_imputed.fit(X_train_imputed, y_train)
y_pred_imputed = knn_imputed.predict(X_test_imputed)
acc_imputed = accuracy_score(y_test, y_pred_imputed)

print("Accuracy on original dataset: {:.3f}".format(acc_orig))
print("Accuracy on imputed dataset: {:.3f}".format(acc_imputed))

