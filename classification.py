import os
import numpy as np
from datetime import datetime
from pathlib import Path
from joblib import Parallel, delayed

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier as gb, RandomForestClassifier as rf

def get_data(datafile='tictactoe64x64_V2_train.csv'):
    # open file to read data from
    file = open(datafile)
    tictactoe_array = np.loadtxt(file, delimiter=",")
    data = []
    tictactoe_classes = []
    for ds in tictactoe_array:
        dataset = ds
        data.append(ds[:-1])
        tictactoe_classes.append(dataset[-1])
    return data, tictactoe_classes

# train classifier and test
def classify(classes, data, clf, protocol_folder, output=True, testing_method='hold_out', test_classes=None, test_data=None):
    time_start = datetime.now().replace(microsecond=0)

    if testing_method=='hold_out':
        # Split data into train and test subsets
        X_train = data
        X_test = test_data
        y_train = classes
        y_test = test_classes
        if (test_data == None) and (test_classes == None):
            X_train, X_test, y_train, y_test = train_test_split(
                data, classes, test_size=0.1, shuffle=True, random_state=0
            )
        true_classes=y_test
        # Learn the digits on the train subset
        clf.fit(X_train, y_train)
        # Predict the value of the digit on the test subset with hold-out-validation
        predicted = clf.predict(X_test)

    if testing_method=='cross_validation':
        true_classes=classes
        # Predict the value of the digit on the test subset with cross-validation
        # spec_output = cross_validate(clf, data, tictactoe_classes, cv=5)
        predicted=cross_val_predict(clf, data, classes, cv=5)

    conf_matrix=confusion_matrix(true_classes, predicted)
    accuracy=accuracy_score(true_classes, predicted)

    # output-stuff (file and terminal)
    if output:
        output="""Classification report for classifier {clf} with {method}:\n
{report}\n
Confusion Matrix:\n
{confusion}\n
Accuracy={accuracy}\n
Time={time}\n""".format(
            method=testing_method,
            clf=clf, 
            report=classification_report(true_classes, predicted),
            accuracy=accuracy,
            confusion=conf_matrix,
            time=str(datetime.now().replace(microsecond=0)-time_start)
        )
        Path(protocol_folder).mkdir(parents=True, exist_ok=True)
        f = open(protocol_folder+'protocol_accuracy='+str(accuracy)[:5]+'_'+str(datetime.now().replace(microsecond=0))+'.txt', "w")
        f.write(output)
        f.close()
        print(output)


def gradient_boosting(datafile='tictactoe64x64_V2_train.csv', testdatafile=None, learning_rate=[0.1,0.5,1.0], n_estimators=[50,100,200], min_samples_split=[2,3,5], min_samples_leaf=[2,3,5], max_depth=[2,3,5], testing_method='cross_validation', n_jobs=1):
    """"
    Trains the GradientBoostingClassifier on the given TicTacToe-Data.

    datafile: string, default='tictactoe64x64_V2_train.csv'
        The file to get the data from. 
        Has to be an .csv-File with 64x64+1 column where the first 64x64 columns contain the pixel values of tictactoe images (in grayscales) and the last column contains the class lable (numeric is desireable but nominal should work well too).
    testing_method: {'hold_out' , 'cross_validation'}, default='cross_validation'
        The testing method to be used.
    n_jobs: int, default='1'
        The number of processes to run in parallel. 
        If 1 then (practically) no proccess run parallel. 
        If -1 the maximal number of proccesses will run in parallel (only recomended for dedicated devices).

    The following parameters are the same as documented for the Sklearn-GradientBoostingClassifier:
    learning_rate: list of float, default=[0.25,0.75]
        Learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.
    n_estimators: list of int, default=[50,100,200]
        The number of boosting stages to perform. 
    min_samples_split: list of int, default=[2,3,5]
        The min number of samples to split a node.
    min_samples_leaf: list of int, default=[2,3,5]
        The minimum number of samples required to be at a leaf node.
    max_depth: list of int, default=[2,3,5]
        The maximum depth of the individual regression estimators (limits the number of nodes in the tree).
    """
    # test gradient boosting with default settings.
    gb_clf = {
        'loss':'deviance',
        'learning_rate':0.1,
        'n_estimators':100,
        'subsample':1.0,
        'criterion':'friedman_mse',
        'min_samples_split':2,
        'min_samples_leaf':1,
        'min_weight_fraction_leaf':0.0,
        'max_depth':3,
        'min_impurity_decrease':0.0,
        'init':None,
        'random_state':0,
        'max_features':None,
        'verbose':1, #Für den Output / das Log
        'max_leaf_nodes':None,
        'warm_start':False,
        'validation_fraction':0.1,
        'n_iter_no_change':None,
        'tol':1e-4,
        'ccp_alpha':0.0
    }
    gb_classifier = []
    # generation all the different gb-configurations
    for lr in learning_rate:
        for ne in n_estimators:
            for mss in min_samples_split:
                for msl in min_samples_leaf:
                    for md in max_depth:
                        gb_clf['learning_rate'] = lr
                        gb_clf['n_estimators'] = ne
                        gb_clf['min_samples_split'] = mss
                        gb_clf['min_samples_leaf'] = msl
                        gb_clf['max_depth'] = md
                        gb_classifier.append(gb(**gb_clf))
    test_data=None
    test_tictactoe_classes=None
    protocol_folder='gradient_boosting/'
    data, tictactoe_classes = get_data(datafile=datafile)
    if testdatafile != None:
        test_data, test_tictactoe_classes = get_data(datafile=testdatafile)
        protocol_folder='gradient_boosting_generalization/'
    Parallel(n_jobs=n_jobs, prefer="threads")(delayed(
        classify)(classes=tictactoe_classes, data=data, test_classes=test_tictactoe_classes, test_data=test_data, clf=clf, protocol_folder=protocol_folder, testing_method=testing_method) for clf in gb_classifier)

def random_forest(datafile='tictactoe64x64_V2_train.csv', testdatafile=None, n_estimators=[50,100,200,300,400], max_depth=[None,10,20,50,100,200,1000], min_samples_split=[2,3,5,10,20,50,100,500], min_samples_leaf=[2,3,4,5,10,20,50,100], testing_method='cross_validation', n_jobs=1):
    """"
    Trains the RandomForestClassifier on the given TicTacToe-Data.

    datafile: string, default='tictactoe64x64_V2_train.csv'
        The file to get the data from. 
        Has to be an .csv-File with 64x64+1 column where the first 64x64 columns contain the pixel values of tictactoe images (in grayscales) and the last column contains the class lable (numeric is desireable but nominal should work well too).
    testing_method: {'hold_out' , 'cross_validation'}, default='cross_validation'
        The testing method to be used.
    n_jobs: int, default='1'
        The number of processes to run in parallel. 
        If 1 then (practically) no proccess run parallel. 
        If -1 the maximal number of proccesses will run in parallel (only recomended for dedicated devices).

    The following parameters are the same as documented for the Sklearn-RandomForestClassifier:

    n_estimators: list of int, default=[50,100,200]
        the number of trees in the forest.
    max_depth: list of int, default=[None,100,500]
        The maximum deepth of trees. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
    min_samples_split: list of int, default=[2,3,5]
        The min number of samples to split a node.
    min_samples_leaf: list of int, default=[2,3,5]
        The minimum number of samples required to be at a leaf node.
    """
    # test random forest with default setting.
    rf_clf = {
        'n_estimators':100,
        'criterion':'gini',
        'max_depth':None,
        'min_samples_split':2,
        'min_samples_leaf':1,
        'min_weight_fraction_leaf':0.0,
        'max_features':'auto',
        'max_leaf_nodes':None,
        'min_impurity_decrease':0.0,
        'bootstrap':True,
        'oob_score':False,
        'n_jobs':None,
        'random_state':None,
        'verbose':1,
        'warm_start':False,
        'class_weight':None,
        'ccp_alpha':0.0,
        'max_samples':None
    }
    rf_classifier = []
    #protocol_folder = 'gradient_boosting'+str(datetime.now())+'/'
    for ne in n_estimators:
        for md in max_depth:
            for mss in min_samples_split:
                for msl in min_samples_leaf:
                    rf_clf['n_estimators'] = ne
                    rf_clf['max_depth'] = md
                    rf_clf['min_samples_split'] = mss
                    rf_clf['min_samples_leaf'] = msl
                    rf_classifier.append(rf(**rf_clf))
    test_data=None
    test_tictactoe_classes=None
    protocol_folder='random_forest/'
    data, tictactoe_classes = get_data(datafile=datafile)
    if testdatafile != None:
        test_data, test_tictactoe_classes = get_data(datafile=testdatafile)
        protocol_folder='random_forest_generalization/'
    # train on every gb-conf and estimate test-error
    Parallel(n_jobs=n_jobs, prefer="threads")(delayed(
        classify)(classes=tictactoe_classes, data=data, test_classes=test_tictactoe_classes, test_data=test_data, clf=clf, protocol_folder=protocol_folder, testing_method=testing_method) for clf in rf_classifier)

# Create Test-Protocolls
gradient_boosting(datafile='tictactoe64x64_V2_train.csv', testing_method='cross_validation', n_jobs=-1)
random_forest(datafile='tictactoe64x64_V2_train.csv', testing_method='cross_validation', n_jobs=-1)

# Evaluate Generalisation Error
random_forest(datafile='tictactoe64x64_V2_train.csv', testdatafile='tictactoe64x64_V2_test.csv', testing_method='hold_out', n_estimators=[400], max_depth=[50], min_samples_split=[2], min_samples_leaf=[2])
gradient_boosting(datafile='tictactoe64x64_V2_train.csv', testdatafile='tictactoe64x64_V2_test.csv', testing_method='hold_out',  learning_rate=[0.1], n_estimators=[200], min_samples_split=[5], min_samples_leaf=[3], max_depth=[5])