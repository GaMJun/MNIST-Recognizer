#!/usr/bin/env python3
import feature_extraction as fe
import classifiers_main as cl
from warnings import filterwarnings
import threading

filterwarnings('ignore')


class ThreadFirstApproach(threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name

    def run(self):
        print("Initializing " + self.name)
        fe.first_feature_extraction(fe.read_all_images())
        cl.classifiers_chooser("./Extracted_Features/first_features.csv")
        print("Ending " + self.name)


class ThreadSecondApproach(threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name

    def run(self):
        print("Initializing " + self.name)
        fe.second_feature_extraction(fe.read_all_images())
        cl.classifiers_chooser("./Extracted_Features/second_features.csv")
        print("Ending " + self.name)


thread0 = ThreadFirstApproach("Thread - First Approach")
thread1 = ThreadSecondApproach("Thread - Second Approach")

thread0.start()
thread1.start()

thread0.join()
thread1.join()

classifier_first_feature = cl.statistical_tests(
    "./Classifiers_Statistical_Results/Classifiers_Statistical_Results_first_features.csv", 'primeira')
classifier_second_feature = cl.statistical_tests(
    "./Classifiers_Statistical_Results/Classifiers_Statistical_Results_second_features.csv", 'segunda')

x_first_feature, x_second_feature, y_first_feature = cl.data_preparation_for_combination(
    "./Classifiers_Statistical_Results/", classifier_first_feature, classifier_second_feature)

avg_hits_borda_count = cl.classifiers_combiner_borda_count(x_first_feature, x_second_feature, y_first_feature)
avg_hits_sum_rule = cl.classifiers_combiner_sum_rule(x_first_feature, x_second_feature, y_first_feature)
cl.record_avg_file(avg_hits_borda_count, avg_hits_sum_rule)
