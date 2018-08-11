Title: Data  leakage pitfalls in data science
Slug: dataLeakage
Date: 2018-08-11 16:00
Category: Data Science
Tags: data science, intermediate, AI, ML, feature selection, validation, data mining, scikit-learn
Author: João Oda
Lang: en
<!-- Status: draft -->

![mario_see_data_leakage](/images/leakimg.jpg)


In an attempt to organize my projects, I have just refactored an old [data mining project](https://github.com/ojon/MD_Proj) done a few years ago. Originally I used Python for data preparation and feature selection and then with [Weka](https://www.cs.waikato.ac.nz/ml/weka/) I performed the training, validation and model selection. I reimplemented everything in Python with [sckit-learn](http://scikit-learn.org/stable/), making use of [_pipelines_](http://scikit-learn.org/stable/modules/pipeline .html # pipeline) and used the opportunity to fix a data leakage issue.

In the context of data science, in a broad sense we have two cases of **data leakage** in predictive models:

## Leaking Features

It occurs when the set of training features has information (typically from the variable we want to predict) that will not be present when we perform the prediction in production environment.

### Examples


Suppose you want to create a predictive model, which predicts whether a loan will be paid on time. Taking a real set of data as an example, on the [lendingclub website](https://www.lendingclub.com/info/download-data.action) we have a database available, where one of the columns "total_rec_late_fee", informs the late fees received so far. If the value of this column is different from zero, it is clear that the loan was not paid on time. We can not use this information, which will only be available after the loan is granted, since a model that makes a prediction before making a loan is what is expected. In this same data source, there are many other columns to be disregarded for similar reason.

In healthcare, a database can present a list of medicines that a person takes and you are trying to develop a model to make a diagnosis. You should take care to find out if the list in the database has been updated after the doctor's diagnosis due to a prescription of the doctor or if they were medicines that the person had already taken due to some condition they have.


Now a case I dealt with in the past when working with trading algorithms. Imagine your data is a time series, where $x_{t-1}$ and $x_{t+1}$ are known, but $x_t$ is a missing data. You decide, to work around this problem, do an interpolation where $x_{t} = \frac{x_{t-1} + x_{t+1}}{2}$ to fill the missing data. This approach compromises the creation of a model that uses data until time t to predict what happens t + 1, since there was a leakage of data from time t + 1 to the previous time t.

## Leakage in Training Examples

It occurs when information from the validation set is used to train the model. You may think that simply partitioning your data set into training and testing before performing the training is enough. However the situation may be a little more complicated and the leakage can occur previously, during data preparation, selection of attributes, reduction of dimensionality and being a purist even a wide data visualization.

Now I will illustrate a feature selection process that I used in the [project](https://github.com/ojon/MD_Proj) mentioned at the beginning of this post. This project deals with data from [gene expression](https://en.wikipedia.org/wiki/Gene_expression) ([_microarray_](https://en.wikipedia.org/wiki/DNA_microarray)) where the number of features is much larger than the number of samples. In this case using a very large number of attributes implies in a model which training takes a longer time, with less interpretability and more prone to overfitting.

### Features selection using the t test

A simple way to perform feature selection is through a t-test based filter. In this version of the filter, for each label to be predicted, the filter bipartises the samples and performs a t-test. The top `w` features, with the highest absolute t-value for each label, are selected


I've implemented the filter in python as a _transformer_, following the sckit-learn library standard. In order to do that we extend the classes `BaseEstimator` and `TransformerMixin` and implement the methods `fit` and` transform`. This makes it possible to use [_pipelines_](http://scikit-learn.org/stable/modules/pipeline.html#pipeline) from the same library. To apply the filter we call the two methods, `fit` and` transform`, in sequence.


```python
class TtestScoreSelection(BaseEstimator, TransformerMixin):
    def __init__(self, w=3):
        self.w = w

    def fit(self, X, y=None):
        X = check_array(X)
        self.input_shape_ = X.shape

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.labels_ = unique_labels(y)

        self.tValuesDF = pd.DataFrame(columns=self.labels_)
        self.sortedIndexes = pd.DataFrame(columns=self.labels_)

        for label in self.labels_:
            sample1 = X[y == label]
            sample2 = X[y != label]
            t = st.ttest_ind(sample1, sample2, equal_var=False)
            self.tValuesDF[label] = np.abs(t[0])                      
            self.sortedIndexes[label] = self.tValuesDF.sort_values(by=label,
                                                                   ascending=False).index

        # Return the transformer
        return self

    def transform(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['input_shape_'])

        # Input validation
        X = check_array(X)

        # union of indexes from the top w columns for each label
        self.selCols = np.unique(self.sortedIndexes[:][0:self.w].values.flatten())

        return X[:, self.selCols]
```  


### Predictions from random numbers

Let's now use our filter to select features in a wrong way. Consider the following didactic experiment:

1. Let's generate a random set of 100 samples of 100,000 Attributes.
2. Reduce the number of features through the `TtestScoreSelection` filter
3. Train a SVM using cross-validation of 5-folds


```Python
#random data generation
X = np.random.rand(100, 100000)
y = np.random.randint(2,size=100)

#feature selection
tscoreSel = TtestScoreSelection(w=30)
tscoreSel.fit(X,y)
selX = tscoreSel.transform(X)

#SVM training and acuracy estimation by cross-validation.
lr = svm.SVC()
k= 5
scores_leak = pd.DataFrame()
scores_leak['score'] = cross_val_score(lr, selX, y, cv=k)
```

As a result we have an accuracy close to 100% over all 5 folds.

![resultados_com_vazamento_de_dados](/images/scores_with_leak.png)

How is this possible ??? Make predictions with near 100% accuracy, from random numbers ?!

If we carefully analyze our procedures we can observe that during step 2, the selection of variables occurred using the whole dataset. Since our filter makes use of the class labels to partition the samples and perform the test, in this step we indirectly propagate information from the data set that will be used in the future to validate the training set.

The training depends on which features were selected and these were selected using even relevant information from the test set. So we have training with leaked examples.

### Fixing data lekage

![Mario](/images/mario_wanna_fix_leakage.jpg)

Someone tells Mario that to fix this data leak that "allowed" a prediction from random numbers we do not need a pipe wrench, but use the correct methodology and write good code. Therefore, we need to separate a dataset, without any intersection with the validation set, where feature selection and training occurs. As we are performing cross-validation of 5-folds, this will be repeated 5 times, the total data set will be partitioned into 5 folds and each time a fold will be used for validation and the complement for feature selection and training.

Scikit-learn allows us to use the [_pipelines_](http://scikit-learn.org/stable/modules/pipeline.html#pipeline) to implement the processing steps that the data is submitted. The `cross_val_score` function is capable of receiving a pipeline as an argument and performing training and cross-validation in k folds. No wonder I implemented the filter in a way compatible with pipelines. So we have the following code:

```python
pipe = Pipeline([
    ('featureSelection', TtestScoreSelection(w=30)),
    ('classify', svm.SVC())
])
k= 5
scores = pd.DataFrame()
scores['score'] = cross_val_score(pipe, X, y, cv=k)
```

The results are:

![resultados_sem_vazamento_de_dados](/images/scores_without_leak.png)

As expected, after all, one can expect nothing more than an accuracy of around 50% for a binary classification from random data.

## Final considerations

Data leaks can lead to unpleasant surprises, such as models that perform better in a development environment than in production. Always be careful, because the data leakage is not always explicitly exposed.

A similar post dealing with leaked training with an R approach can be found on [Johann de Jong's blog](https://johanndejong.wordpress.com/2017/08/06/feature-selection-cross-validation-and -data-leakage /)
