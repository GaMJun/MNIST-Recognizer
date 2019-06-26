import pandas as pd
from scipy import stats


def mann_whitney(all_classifiers, first, second):
    if second >= len(all_classifiers):
        return all_classifiers

    u, pvalue = stats.mannwhitneyu(all_classifiers[first], all_classifiers[second], alternative='two-sided')
    if pvalue < 0.05:
        if u > 50:
            all_classifiers.pop(second)
            mann_whitney(all_classifiers, 0, 1)
        else:
            all_classifiers.pop(first)
            mann_whitney(all_classifiers, 0, 1)
    else:
        mann_whitney(all_classifiers, first, second + 1)

    return all_classifiers


def find_classifier(find_classifier, all_classifiers):
    if find_classifier[0] == all_classifiers[0][0] and find_classifier[1] == all_classifiers[0][1] and find_classifier[
        2] == all_classifiers[0][2] and find_classifier[3] == all_classifiers[0][3] and find_classifier[4] == \
            all_classifiers[0][4] and find_classifier[5] == all_classifiers[0][5] and find_classifier[6] == \
            all_classifiers[0][6] and find_classifier[7] == all_classifiers[0][7] and find_classifier[8] == \
            all_classifiers[0][8] and find_classifier[9] == all_classifiers[0][9]:
        return 'dt'
    elif find_classifier[0] == all_classifiers[1][0] and find_classifier[1] == all_classifiers[1][1] and \
            find_classifier[
                2] == all_classifiers[1][2] and find_classifier[3] == all_classifiers[1][3] and find_classifier[4] == \
            all_classifiers[1][4] and find_classifier[5] == all_classifiers[1][5] and find_classifier[6] == \
            all_classifiers[1][6] and find_classifier[7] == all_classifiers[1][7] and find_classifier[8] == \
            all_classifiers[1][8] and find_classifier[9] == all_classifiers[1][9]:
        return 'knn'
    elif find_classifier[0] == all_classifiers[2][0] and find_classifier[1] == all_classifiers[2][1] and \
            find_classifier[
                2] == all_classifiers[2][2] and find_classifier[3] == all_classifiers[2][3] and find_classifier[4] == \
            all_classifiers[2][4] and find_classifier[5] == all_classifiers[2][5] and find_classifier[6] == \
            all_classifiers[2][6] and find_classifier[7] == all_classifiers[2][7] and find_classifier[8] == \
            all_classifiers[2][8] and find_classifier[9] == all_classifiers[2][9]:
        return 'nb'
    elif find_classifier[0] == all_classifiers[3][0] and find_classifier[1] == all_classifiers[3][1] and \
            find_classifier[
                2] == all_classifiers[3][2] and find_classifier[3] == all_classifiers[3][3] and find_classifier[4] == \
            all_classifiers[3][4] and find_classifier[5] == all_classifiers[3][5] and find_classifier[6] == \
            all_classifiers[3][6] and find_classifier[7] == all_classifiers[3][7] and find_classifier[8] == \
            all_classifiers[3][8] and find_classifier[9] == all_classifiers[3][9]:
        return 'svm'
    elif find_classifier[0] == all_classifiers[4][0] and find_classifier[1] == all_classifiers[4][1] and \
            find_classifier[
                2] == all_classifiers[4][2] and find_classifier[3] == all_classifiers[4][3] and find_classifier[4] == \
            all_classifiers[4][4] and find_classifier[5] == all_classifiers[4][5] and find_classifier[6] == \
            all_classifiers[4][6] and find_classifier[7] == all_classifiers[4][7] and find_classifier[8] == \
            all_classifiers[4][8] and find_classifier[9] == all_classifiers[4][9]:
        return 'rf'
    elif find_classifier[0] == all_classifiers[5][0] and find_classifier[1] == all_classifiers[5][1] and \
            find_classifier[
                2] == all_classifiers[5][2] and find_classifier[3] == all_classifiers[5][3] and find_classifier[4] == \
            all_classifiers[5][4] and find_classifier[5] == all_classifiers[5][5] and find_classifier[6] == \
            all_classifiers[5][6] and find_classifier[7] == all_classifiers[5][7] and find_classifier[8] == \
            all_classifiers[5][8] and find_classifier[9] == all_classifiers[5][9]:
        return 'mlp'


def load_files(features_path):
    try:
        data_frame = pd.read_csv(features_path)
        dt = data_frame.iloc[:11, 0].values
        knn = data_frame.iloc[:11, 2].values
        nb = data_frame.iloc[:11, 5].values
        svm = data_frame.iloc[:11, 7].values
        rf = data_frame.iloc[:11, 10].values
        mlp = data_frame.iloc[:11, 14].values
        return dt, knn, nb, svm, rf, mlp
    except FileNotFoundError:
        print("Extracted features files not found!")
        exit(2)


def load_files_to_combination(features_path, classifier):
    return_x = []
    return_y = []

    for i in range(1, 11):
        try:
            data_frame = pd.read_csv(features_path + str(i) + '/' + str(classifier) + '.csv', index_col=False)
            x = data_frame.iloc[:, 0:10].values
            y = data_frame.iloc[:, 10].values

            return_x.append(x)
            return_y.append(y)
        except FileNotFoundError:
            print("Extracted features files not found!")
            return return_x, return_y

    return return_x, return_y
