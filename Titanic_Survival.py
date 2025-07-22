import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Data_PreProcessing import Processed_Data_test,Processed_Data_Train
from sklearn.metrics import accuracy_score
from Single_Tree import Tree

data_train = Processed_Data_Train()
data_test = Processed_Data_test()

X_train = data_train.drop('Survived', axis=1).values.astype(int)
y_train = data_train['Survived'].values.astype(int)
y_train = y_train.reshape(-1, 1)


X_test = data_test.values.astype(int)

print(f"Shape for X_train is {X_train.shape}")
print(f"Shape for X_test is {X_test.shape}")
print(f"Shape for y_train is {y_train.shape}")

print(y_train[0])


class RandomForestClassifier():
    def __init__(self, n_trees, max_depth, min_samples_split):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            X_sample, y_sample = self.bootstrap_sample(X, y)
            tree = Tree(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        print(f"Random Forest trained with {self.n_trees} trees")

    def predict(self, X):
        """Making predictions using majority voting from all trees"""

        tree_predictions = []
        for tree in self.trees:
            predictions = tree.predict(X)
            tree_predictions.append(predictions)
        
        tree_predictions = np.array(tree_predictions)
        
        final_predictions = []
        for i in range(X.shape[0]):
            sample_predictions = tree_predictions[:, i]
            unique, counts = np.unique(sample_predictions, return_counts=True)
            majority_prediction = unique[np.argmax(counts)]
            final_predictions.append(majority_prediction)
        
        return np.array(final_predictions)

    
print("Sample y_train values:", y_train[:10].flatten())
print("Unique y_train values:", np.unique(y_train.astype(str)))
print("Data types in y_train:", set(type(x).__name__ for x in y_train.flatten()))


n_tress = 10
max_depth = 4
min_samples_split = 10

Model = RandomForestClassifier(n_tress,max_depth,min_samples_split)
Model.fit(X_train, y_train)


Test_results = Model.predict(X_test)

print(f"Test Results Shape: {Test_results.shape}")

print(f"Number of trees in forest: {Model.n_trees}")
print(f"Max depth per tree: {Model.max_depth}")
print(f"Min samples per split: {Model.min_samples_split}")

submission = pd.DataFrame({
    'PassengerId': range(892, 892 + len(Test_results)), 
    'Survived': Test_results
})
submission.to_csv(f'Titanic_predictions_n_trees{n_tress}_Max_depth{max_depth}_min_split{min_samples_split}.csv', index=False)