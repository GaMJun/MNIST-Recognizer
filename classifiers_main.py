import statistics
import classifiers
import statistical_main
import numpy as np
from scipy import stats
import operator


def classifiers_chooser(features_path):
    iterations = 11

    x, y = classifiers.load_features(features_path)

    dt = []
    knn = []
    nb = []
    svm = []
    rf = []
    mlp = []
    params_dt = []
    params_knn = []
    params_svm = []
    params_rf = []
    params_mlp = []

    for index in range(1, iterations):
        print('\nIteration ', index, '\n')
        classifiers.redo_ds_partitions(x, y)

        print('Finding best parameters\n')
        bp_dt = classifiers.fbp_dt()
        params_dt.append(bp_dt)
        bp_knn = classifiers.fbp_knn()
        params_knn.append(bp_knn)
        print('Naive Bayes\nUsing default configuration')
        bp_svm = classifiers.fbp_svm()
        params_svm.append(bp_svm)
        bp_rf = classifiers.fbp_rf()
        params_rf.append(bp_rf)
        bp_mlp = classifiers.fbp_mlp()
        params_mlp.append(bp_mlp)

        print("Executing Decision Tree with optimized parameters")
        dt_acc, dt_proba, dt_true = classifiers.dt(bp_dt)
        dt_proba_and_true = np.c_[dt_proba, dt_true]
        dt.append(dt_acc)

        print("Executing KNN with optimized parameters")
        knn_acc, knn_proba, knn_true = classifiers.knn(bp_knn)
        knn_proba_and_true = np.c_[knn_proba, knn_true]
        knn.append(knn_acc)

        print("Executing Naive Bayes with no optimized parameters")
        nb_acc, nb_proba, nb_true = classifiers.nb()
        nb_proba_and_true = np.c_[nb_proba, nb_true]
        nb.append(nb_acc)

        print("Executing Support Vector Machine with optimized parameters")
        svm_acc, svm_proba, svm_true = classifiers.svm(bp_svm)
        svm_proba_and_true = np.c_[svm_proba, svm_true]
        svm.append(svm_acc)

        print("Executing Random Forest with optimized parameters")
        rf_acc, rf_proba, rf_true = classifiers.rf(bp_rf)
        rf_proba_and_true = np.c_[rf_proba, rf_true]
        rf.append(rf_acc)

        print("Executing Multi Layer Perceptron with optimized parameters")
        mlp_acc, mlp_proba, mlp_true = classifiers.mlp(bp_mlp)
        mlp_proba_and_true = np.c_[mlp_proba, mlp_true]
        mlp.append(mlp_acc)

        with open(
                "./Classifiers_Statistical_Results/" + features_path.rsplit('/', 1)[-1][
                                                       :-4] + "/" + "prob_" + str(
                    index) + "/dt.csv", "w") as fp:
            fp.write("%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %s" % (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "True"))
            fp.write("\n")
            for n in range(len(dt_proba_and_true)):
                for _ in range(10):
                    fp.write("%f," % dt_proba_and_true[n][_])
                fp.write("%d," % dt_proba_and_true[n][10])
                fp.write("\n")

        with open(
                "./Classifiers_Statistical_Results/" + features_path.rsplit('/', 1)[-1][
                                                       :-4] + "/" + "prob_" + str(
                    index) + "/knn.csv", "w") as fp:
            fp.write("%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %s" % (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "True"))
            fp.write("\n")
            for n in range(len(knn_proba_and_true)):
                for _ in range(10):
                    fp.write("%f," % knn_proba_and_true[n][_])
                fp.write("%d," % knn_proba_and_true[n][10])
                fp.write("\n")

        with open(
                "./Classifiers_Statistical_Results/" + features_path.rsplit('/', 1)[-1][
                                                       :-4] + "/" + "prob_" + str(
                    index) + "/nb.csv", "w") as fp:
            fp.write("%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %s" % (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "True"))
            fp.write("\n")
            for n in range(len(nb_proba_and_true)):
                for _ in range(10):
                    fp.write("%f," % nb_proba_and_true[n][_])
                fp.write("%d," % nb_proba_and_true[n][10])
                fp.write("\n")

        with open(
                "./Classifiers_Statistical_Results/" + features_path.rsplit('/', 1)[-1][
                                                       :-4] + "/" + "prob_" + str(
                    index) + "/svm.csv", "w") as fp:
            fp.write("%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %s" % (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "True"))
            fp.write("\n")
            for n in range(len(svm_proba_and_true)):
                for _ in range(10):
                    fp.write("%f," % svm_proba_and_true[n][_])
                fp.write("%d," % svm_proba_and_true[n][10])
                fp.write("\n")

        with open(
                "./Classifiers_Statistical_Results/" + features_path.rsplit('/', 1)[-1][
                                                       :-4] + "/" + "prob_" + str(
                    index) + "/rf.csv", "w") as fp:
            fp.write("%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %s" % (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "True"))
            fp.write("\n")
            for n in range(len(rf_proba_and_true)):
                for _ in range(10):
                    fp.write("%f," % rf_proba_and_true[n][_])
                fp.write("%d," % rf_proba_and_true[n][10])
                fp.write("\n")

        with open(
                "./Classifiers_Statistical_Results/" + features_path.rsplit('/', 1)[-1][
                                                       :-4] + "/" + "prob_" + str(
                    index) + "/mlp.csv", "w") as fp:
            fp.write("%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %s" % (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "True"))
            fp.write("\n")
            for n in range(len(mlp_proba_and_true)):
                for _ in range(10):
                    fp.write("%f," % mlp_proba_and_true[n][_])
                fp.write("%d," % mlp_proba_and_true[n][10])
                fp.write("\n")

    with open(
            "./Classifiers_Statistical_Results/Classifiers_Statistical_Results_" + features_path.rsplit('/', 1)[-1],
            "w") as fp:
        fp.write(
            "Dt, depth, Knn, weights, k, Nb, No Params, Svm, kernel, penalty, Rf, n_estimators, criterion, max_depth, Mlp, max_iter, hidder_layers, learning_rate, learning_rate_init\n")
        for index in range(iterations-1):
            fp.write("%f, %s, %f, %s, %s, %f, %s, %f, %s, %s, %f, %s, %s, %s, %f, %s, %s, %s, %s\n" % (
                dt[index], params_dt[index], knn[index], params_knn[index][0], params_knn[index][1], nb[index],
                "Default", svm[index], params_svm[index][0], params_svm[index][1], rf[index], params_rf[index][0],
                params_rf[index][1], params_rf[index][2], mlp[index], params_mlp[index][0], params_mlp[index][1],
                params_mlp[index][2], params_mlp[index][3]))
        fp.write("\n%f, %s, %f, %s, %s, %f, %s, %f, %s, %s, %f, %s, %s, %s, %f, %s, %s, %s, %s\n" % (
            float(np.mean(dt)), '', float(np.mean(knn)), '', '', float(np.mean(nb)), '',
            float(np.mean(svm)), '', '', float(np.mean(rf)), '', '', '', float(np.mean(mlp)), '', '', '', ''))
        fp.write("%f, %s, %f, %s, %s, %f, %s, %f, %s, %s, %f, %s, %s, %s, %f, %s, %s, %s, %s\n" % (
            statistics.pstdev(dt), '', statistics.pstdev(knn), '', '',
            statistics.pstdev(nb), '', statistics.pstdev(svm), '', '', statistics.pstdev(rf), '', '', '',
            statistics.pstdev(mlp), '', '', '', ''))


def statistical_tests(features_path, approach):
    dt, knn, nb, svm, rf, mlp = statistical_main.load_files(features_path)
    h_statistical, p_val = stats.kruskal(dt, knn, nb, svm, rf, mlp)

    if p_val < 0.05:
        print("There is a significant difference between the classifiers!")
        print("Calculating Mann-Whitney!")

        all_classifiers = [dt, knn, nb, svm, rf, mlp]
        to_function_all_classifiers = [dt, knn, nb, svm, rf, mlp]

        return_all_classifiers = statistical_main.mann_whitney(to_function_all_classifiers, 0, 1)
        best_classifiers = []
        if len(all_classifiers) > 1:
            for i in range(len(return_all_classifiers)):
                best_classifiers.append(statistical_main.find_classifier(return_all_classifiers[i], all_classifiers))

            print(
                'For this database and for ' + approach + ' approach there are 2 or more equivalent classifiers, they are: ')
            for i in range(len(best_classifiers)):
                if best_classifiers[i] == 'dt':
                    print('Decision Tree')
                elif best_classifiers[i] == 'knn':
                    print('K Nearest Neighbor')
                elif best_classifiers[i] == 'nb':
                    print('Naive Bayes')
                elif best_classifiers[i] == 'svm':
                    print('Support Vector Machine')
                elif best_classifiers[i] == 'rf':
                    print('Randon Forest')
                elif best_classifiers[i] == 'mlp':
                    print('Multi Layer Perceptron')

            print('Since these are equivalent we chose the first one')
            return best_classifiers[0]

        else:
            best_classifiers.append(statistical_main.find_classifier(return_all_classifiers[0], all_classifiers))
            print('For this database and for ' + approach + ' approach is: ')
            if best_classifiers[0] == 'dt':
                print('Decision Tree')
            elif best_classifiers[0] == 'knn':
                print('K Nearest Neighbor')
            elif best_classifiers[0] == 'nb':
                print('Naive Bayes')
            elif best_classifiers[0] == 'svm':
                print('Support Vector Machine')
            elif best_classifiers[0] == 'rf':
                print('Randon Forest')
            elif best_classifiers[0] == 'mlp':
                print('Multi Layer Perceptron')
            return best_classifiers[0]

    if p_val > 0.05:
        print("There is no significant difference between the classifiers!")
        print("Therefore, any database can be analyzed, using Random Forest")
        return "rf"


def data_preparation_for_combination(features_path, classifier_first_feature, classifier_second_feature):
    x_first_feature, y_first_feature = statistical_main.load_files_to_combination(
        features_path + 'first_features/prob_',
        classifier_first_feature)
    x_second_feature, y_second_feature = statistical_main.load_files_to_combination(
        features_path + 'second_features/prob_',
        classifier_second_feature)

    return x_first_feature, x_second_feature, y_first_feature


def classifiers_combiner_borda_count(x_first_feature, x_second_feature, y_first_feature):
    x_combination_first = [0] * 10
    x_combination_second = [0] * 10
    x_combination = [0] * 10

    final_votes = []
    votes_per_file = []

    for i in range(len(x_first_feature)):
        for j in range(len(x_first_feature[i])):
            position_array = sorted(range(len(x_first_feature[i][j])), reverse=True,
                                    key=lambda k: x_first_feature[i][j][k])

            x_combination_first[position_array[0]] = 4
            x_combination_first[position_array[1]] = 3
            x_combination_first[position_array[2]] = 2
            x_combination_first[position_array[3]] = 1

            position_array = sorted(range(len(x_second_feature[i][j])), reverse=True,
                                    key=lambda k: x_second_feature[i][j][k])

            x_combination_second[position_array[0]] = 4
            x_combination_second[position_array[1]] = 3
            x_combination_second[position_array[2]] = 2
            x_combination_second[position_array[3]] = 1

            for l in range(len(x_combination_first)):
                x_combination[l] = x_combination_first[l] + x_combination_second[l]

            position_array = sorted(range(len(x_combination)), reverse=True, key=lambda k: x_combination[k])
            votes_per_file.append(position_array[0])
            x_combination_first = [0] * 10
            x_combination_second = [0] * 10
        final_votes.append(votes_per_file)
        votes_per_file = []

    count_hits = 0
    count_miss = 0
    avg_hits = []
    avg_miss = []

    for i in range(len(final_votes)):
        for j in range(len(final_votes[i])):
            if final_votes[i][j] == y_first_feature[i][j]:
                count_hits += 1
            else:
                count_miss += 1
        avg_hits.append(count_hits / 4200)
        avg_miss.append(count_miss / 4200)
        count_hits = 0
        count_miss = 0
    return avg_hits


def classifiers_combiner_sum_rule(x_first_feature, x_second_feature, y_first_feature):
    final_votes = []
    votes_per_file = []

    for i in range(len(x_first_feature)):
        for j in range(len(x_first_feature[i])):
            x_combination = list(map(operator.add, x_first_feature[i][j], x_second_feature[i][j]))
            position_array = sorted(range(len(x_combination)), reverse=True, key=lambda k: x_combination[k])
            votes_per_file.append(position_array[0])
        final_votes.append(votes_per_file)
        votes_per_file = []

    count_hits = 0
    count_miss = 0
    avg_hits = []
    avg_miss = []

    for i in range(len(final_votes)):
        for j in range(len(final_votes[i])):
            if final_votes[i][j] == y_first_feature[i][j]:
                count_hits += 1
            else:
                count_miss += 1
        avg_hits.append(count_hits / 4200)
        avg_miss.append(count_miss / 4200)
        count_hits = 0
        count_miss = 0
    return avg_hits


def record_avg_file(avg_borda_count, avg_sum_rule):
    iterations = len(avg_borda_count)
    with open(
            "./Classifiers_Statistical_Results/Classifiers_Statical_Results_Combination.csv", "w") as fp:
        fp.write("Borda Count, Sum Rule\n")
        for index in range(iterations):
            fp.write("%f, %f\n" % (avg_borda_count[index], avg_sum_rule[index]))
        fp.write("\n%f, %f\n" % (float(np.mean(avg_borda_count)), float(np.mean(avg_sum_rule))))
        fp.write("%f, %f\n" % (statistics.pstdev(avg_borda_count), statistics.pstdev(avg_sum_rule)))
