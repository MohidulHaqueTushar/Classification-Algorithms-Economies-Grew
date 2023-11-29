# Classification-Algorithms-Economies-Grew

**Tasks:**

The python script load and analysis the given data points, and then run three different binary classification algorithms to explain whether regions economies grew by more than 5%. Goodness of the models evaluated at the end.

**Treatments:**

Chosen three classification algorithms are DecisionTree, RandomForest, and SupportVectorMachine. To evaluate the goodness of trained model, we need to split the given data in train and test set. In not rich data
situation 80% data used for train the model. After analysis, we can see that the given data is imbalanced: for this reason Stratified Sampling is used in train_test_split to split the data into train and test set. The three mentioned binary classification algorithms will be trained on train set, and the test set will be used to evaluate those trained models.

Please read the pdf `ClassificationAlgorithms_EconomiesGrew.pdf` for more information. 
