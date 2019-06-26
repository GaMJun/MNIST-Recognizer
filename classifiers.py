from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd


def load_features(features_path):
    try:
        data_frame = pd.read_csv(features_path)
        x = data_frame.iloc[:, 0:12].values
        y = data_frame.iloc[:, 12].values
        return x, y
    except FileNotFoundError:
        print("Extracted features files not found!")
        exit(2)


def redo_ds_partitions(x, y):
    global train_x
    global train_y
    global validation_x
    global test_x
    global validation_y
    global test_y

    train_x, aux_x, train_y, aux_y = train_test_split(x, y, test_size=0.2)
    validation_x, test_x, validation_y, test_y = train_test_split(aux_x, aux_y, test_size=0.5)


def accuracy(y_true: object, y_pred: object) -> object:
    return np.mean(np.equal(y_true, y_pred))


def dt(bp_dt):
    decision_tree = DecisionTreeClassifier(max_depth=bp_dt)
    decision_tree = decision_tree.fit(train_x, train_y)
    return accuracy(test_y, decision_tree.predict(test_x)), decision_tree.predict_proba(test_x), test_y


def knn(bp_knn):
    k_nearest_neighbors = KNeighborsClassifier(n_neighbors=bp_knn[1], algorithm='auto', weights=bp_knn[0])
    k_nearest_neighbors.fit(train_x, train_y)
    return accuracy(test_y, k_nearest_neighbors.predict(test_x)), k_nearest_neighbors.predict_proba(test_x), test_y


def nb():
    naive_bayes = BernoulliNB()
    naive_bayes.fit(train_x, train_y)
    return accuracy(test_y, naive_bayes.predict(test_x)), naive_bayes.predict_proba(test_x), test_y


def mlp(bp_mlp):
    multi_layer_perceptron = MLPClassifier(hidden_layer_sizes=bp_mlp[1], activation='relu', max_iter=bp_mlp[0],
                                           learning_rate=bp_mlp[2], learning_rate_init=bp_mlp[3])
    multi_layer_perceptron.fit(train_x, train_y)
    return accuracy(test_y, multi_layer_perceptron.predict(test_x)), multi_layer_perceptron.predict_proba(
        test_x), test_y


def svm(bp_svm):
    sup_vector_machine = SVC(C=bp_svm[1], kernel=bp_svm[0], probability=True)
    sup_vector_machine.fit(train_x, train_y)
    return accuracy(test_y, sup_vector_machine.predict(test_x)), sup_vector_machine.predict_proba(test_x), test_y


def rf(bp_rf):
    random_forest = RandomForestClassifier(n_estimators=bp_rf[0], criterion=bp_rf[1], max_depth=bp_rf[2])
    random_forest.fit(train_x, train_y)
    return accuracy(test_y, random_forest.predict(test_x)), random_forest.predict_proba(test_x), test_y


# Finding best parameters
def fbp_dt():
    print('Decision Tree\nFinding best depth\t')
    aux_depth = 0
    aux_accuracy = 0.00

    decision_tree = DecisionTreeClassifier(max_depth=None)
    decision_tree = decision_tree.fit(train_x, train_y)
    if (accuracy(validation_y, decision_tree.predict(validation_x)) > aux_accuracy):
        aux_accuracy = accuracy(validation_y, decision_tree.predict(validation_x))
        aux_depth = None
    for depth in range(1, 26, 2):
        print(depth)
        decision_tree = DecisionTreeClassifier(max_depth=depth)
        decision_tree = decision_tree.fit(train_x, train_y)
        if (accuracy(validation_y, decision_tree.predict(validation_x)) > aux_accuracy):
            aux_accuracy = accuracy(validation_y, decision_tree.predict(validation_x))
            aux_depth = depth
    print('Best depth = ', aux_depth)
    return aux_depth


def fbp_knn():
    print('\nK Nearest Neighbors\nFinding best K and distance metric\t')
    aux_accuracy = 0.00
    aux_weights = ''
    aux_k = 0

    for k in range(3, 20, 1):
        print(k)
        k_nearest_neighbors_uniform = KNeighborsClassifier(n_neighbors=k, algorithm='auto', weights='uniform')
        k_nearest_neighbors_uniform.fit(train_x, train_y)
        if (accuracy(validation_y, k_nearest_neighbors_uniform.predict(validation_x)) > aux_accuracy):
            aux_accuracy = accuracy(validation_y, k_nearest_neighbors_uniform.predict(validation_x))
            aux_weights = 'uniform'
            aux_k = k

        k_nearest_neighbors_distance = KNeighborsClassifier(n_neighbors=k, algorithm='auto', weights='distance')
        k_nearest_neighbors_distance.fit(train_x, train_y)
        if accuracy(validation_y, k_nearest_neighbors_distance.predict(validation_x)) > aux_accuracy:
            aux_accuracy = accuracy(validation_y, k_nearest_neighbors_distance.predict(validation_x))
            aux_weights = 'distance'
            aux_k = k

    print('Best K = ', aux_k, ' and distance metric = ', aux_weights)
    return aux_weights, aux_k


def fbp_svm():
    print('\nSupport Vector Machine\nFinding best kernel and penalty\t')
    aux_accuracy = 0.00
    aux_penalty = 0.00
    aux_kernel = ''

    for penalty in range(1, 125, 10):
        print(penalty / 10)
        sup_vector_machine_poly = SVC(C=penalty / 10, kernel='poly')
        sup_vector_machine_poly.fit(train_x, train_y)

        sup_vector_machine_rbf = SVC(C=penalty / 10, kernel='rbf')
        sup_vector_machine_rbf.fit(train_x, train_y)

        if (accuracy(validation_y, sup_vector_machine_poly.predict(validation_x)) > aux_accuracy):
            aux_accuracy = accuracy(validation_y, sup_vector_machine_poly.predict(validation_x))
            aux_kernel = 'poly'
            aux_penalty = penalty / 10

        if (accuracy(validation_y, sup_vector_machine_rbf.predict(validation_x)) > aux_accuracy):
            aux_accuracy = accuracy(validation_y, sup_vector_machine_rbf.predict(validation_x))
            aux_kernel = 'rbf'
            aux_penalty = penalty / 10

    print('Best Kernel = ', aux_kernel, 'and penalty = ', aux_penalty)
    return aux_kernel, aux_penalty


def fbp_rf():
    print("\nRandom Forest\nFinding best n_estimators, criterion and max_depth\t")
    aux_accuracy = 0.00
    aux_n_estimators = 0
    aux_max_depth = 0
    aux_criterion = ''

    for n_estimators in range(150, 351, 10):
        random_forest_classifier_gini = RandomForestClassifier(n_estimators=n_estimators, criterion="gini",
                                                               max_depth=None)
        random_forest_classifier_gini.fit(test_x, test_y)

        random_forest_classifier_entropy = RandomForestClassifier(n_estimators=n_estimators, criterion="entropy",
                                                                  max_depth=None)
        random_forest_classifier_entropy.fit(test_x, test_y)

        if (accuracy(validation_y, random_forest_classifier_gini.predict(validation_x)) > aux_accuracy):
            aux_accuracy = accuracy(validation_y, random_forest_classifier_gini.predict(validation_x))
            aux_n_estimators = n_estimators
            aux_criterion = 'gini'
            aux_max_depth = None

        if (accuracy(validation_y, random_forest_classifier_entropy.predict(validation_x)) > aux_accuracy):
            aux_accuracy = accuracy(validation_y, random_forest_classifier_entropy.predict(validation_x))
            aux_n_estimators = n_estimators
            aux_criterion = 'entropy'
            aux_max_depth = None

        for max_depth in range(1, 46, 2):
            print(n_estimators, max_depth)
            random_forest_classifier_gini = RandomForestClassifier(n_estimators=n_estimators, criterion="gini",
                                                                   max_depth=max_depth)
            random_forest_classifier_gini.fit(test_x, test_y)

            random_forest_classifier_entropy = RandomForestClassifier(n_estimators=n_estimators, criterion="entropy",
                                                                      max_depth=max_depth)
            random_forest_classifier_entropy.fit(test_x, test_y)

            if (accuracy(validation_y, random_forest_classifier_gini.predict(validation_x)) > aux_accuracy):
                aux_accuracy = accuracy(validation_y, random_forest_classifier_gini.predict(validation_x))
                aux_n_estimators = n_estimators
                aux_criterion = 'gini'
                aux_max_depth = max_depth

            if (accuracy(validation_y, random_forest_classifier_entropy.predict(validation_x)) > aux_accuracy):
                aux_accuracy = accuracy(validation_y, random_forest_classifier_entropy.predict(validation_x))
                aux_n_estimators = n_estimators
                aux_criterion = 'entropy'
                aux_max_depth = max_depth

    print('Best n_estimators = ', aux_n_estimators, 'criterion = ', aux_criterion,
          'and max_depth = ', aux_max_depth)
    return aux_n_estimators, aux_criterion, aux_max_depth


def fbp_mlp():
    print("\nMulti Layer Perceptron\nFinding best max_iter, hidden_layer_sizes, learning_rate, learning_rate_init\t")
    aux_accuracy = 0.00
    aux_hls = 0
    aux_mi = 0
    aux_learning_rate = ''
    aux_lri = 0.00

    for mi in range(200, 320, 20):
        for hls in range(2, 6):
            for lri in range(1, 4, 1):
                print(mi, hls, (1 / pow(10, lri)))
                multi_layer_perceptron_constant = MLPClassifier(hidden_layer_sizes=hls, activation='relu', max_iter=mi,
                                                                learning_rate='constant',
                                                                learning_rate_init=(1 / pow(10, lri)))
                multi_layer_perceptron_constant.fit(train_x, train_y)

                multi_layer_perceptron_invscaling = MLPClassifier(hidden_layer_sizes=hls, activation='relu',
                                                                  max_iter=mi,
                                                                  learning_rate='invscaling',
                                                                  learning_rate_init=(1 / pow(10, lri)))
                multi_layer_perceptron_invscaling.fit(train_x, train_y)

                multi_layer_perceptron_adaptive = MLPClassifier(hidden_layer_sizes=hls, activation='relu', max_iter=mi,
                                                                learning_rate='adaptive',
                                                                learning_rate_init=(1 / pow(10, lri)))
                multi_layer_perceptron_adaptive.fit(train_x, train_y)

                if (accuracy(validation_y, multi_layer_perceptron_constant.predict(validation_x)) > aux_accuracy):
                    aux_accuracy = accuracy(validation_y, multi_layer_perceptron_constant.predict(validation_x))
                    aux_learning_rate = 'constant'
                    aux_hls = hls
                    aux_mi = mi
                    aux_lri = (1 / pow(10, lri))

                if (accuracy(validation_y, multi_layer_perceptron_invscaling.predict(validation_x)) > aux_accuracy):
                    aux_accuracy = accuracy(validation_y, multi_layer_perceptron_invscaling.predict(validation_x))
                    aux_learning_rate = 'invscaling'
                    aux_hls = hls
                    aux_mi = mi
                    aux_lri = (1 / pow(10, lri))

                if (accuracy(validation_y, multi_layer_perceptron_adaptive.predict(validation_x)) > aux_accuracy):
                    aux_accuracy = accuracy(validation_y, multi_layer_perceptron_adaptive.predict(validation_x))
                    aux_learning_rate = 'adaptive'
                    aux_hls = hls
                    aux_mi = mi
                    aux_lri = (1 / pow(10, lri))

    print('Best max_iter = ', aux_mi, ' hidden_layer_sizes = ', aux_hls, 'learning_rate = ', aux_learning_rate,
          'learning_rate_init = ', aux_lri)
    return aux_mi, aux_hls, aux_learning_rate, aux_lri
