from cmath import nan
from msilib.schema import Feature
import numpy as np
from scipy.stats import skew, kurtosis
import math
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import os
import pickle
from librosa import lpc

class SGT_Dataset:
    def __init__(self, window_size=250, window_inc=100, frequency=1259):
        self.frequency = frequency
        self.window_size = window_size
        self.window_inc = window_inc
        self.dataset = {}
    
    def import_data(self, data_file):
        self.filename = data_file
        if not os.path.exists(self.filename):
            print("Error has occurred in loading file")
        
        raw_data = np.genfromtxt(self.filename, delimiter=",")
        rep = raw_data[:,-1]
        gesture = raw_data[:,-2]
        # remove the counters & labels
        raw_data = raw_data[:,2:-2]
        # remove the zero columns
        # store active channels (useful for classifier later)
        self.active_channels = raw_data.sum(axis=0) != 0
        raw_data = raw_data[:,self.active_channels]

        # remove zero rows - delsys hasn't started yet
        zero_row_ids = [i for i, row in enumerate(raw_data.tolist()) if row == [0]*sum(self.active_channels)]
        raw_data = np.delete(raw_data, zero_row_ids, axis=0)
        gesture  = np.delete(gesture, zero_row_ids, axis=0)
        rep      = np.delete(rep, zero_row_ids, axis=0)

        win_size_s = int(self.window_size*self.frequency/1000)
        win_inc_s  = int(self.window_inc *self.frequency/1000)
        # windowing
        num_windows = (raw_data.shape[0] - win_size_s) // win_inc_s
        st_idx = 0
        ed_idx = win_size_s
        data = []
        classes  = []
        reps     = []
        for w in range(num_windows):
            # DELSYS STARTS RECORDING BEFORE TRANSMITTING -- ALL ZEROS LEADS TO FEATURE EXTRACTION ERRORS.
            if not (raw_data[st_idx:ed_idx,:] == 0).all():
                class_mode, _ = np.unique(gesture[st_idx:ed_idx], return_counts=True)
                rep_mode, _   = np.unique(rep[st_idx:ed_idx], return_counts=True)
                if class_mode[0] != 1000:
                    
                    classes.append(class_mode[0])
                    reps.append(rep_mode[0])
                    data.append(raw_data[st_idx:ed_idx, :].transpose())
                
            st_idx += win_inc_s
            ed_idx += win_inc_s

        self.dataset['data'] = np.array(data)
        self.dataset['rep']  = np.array(reps, dtype=np.int32)
        self.dataset['class']= np.array(classes, dtype=np.int32)
    
    def active_threshold(self):
        nm_windows = self.dataset['class']==0
        # get mav
        feature_extractor = Feature_Extractor(num_channels = self.active_channels.shape[0])
        feature_list = ['MAV']
        mav = feature_extractor.extract(feature_list, self.dataset['data'])['MAV']
        # mean and std across channels
        channel_mean = np.mean(mav, axis=1)
        nm_channel_mean = channel_mean[nm_windows]
        nm_mean_MAV = np.mean(nm_channel_mean)
        nm_std_MAV  = np.std(nm_channel_mean)
        # check that windows are not outside 3std from mean if active class
        thresholded_windows = channel_mean < (nm_mean_MAV + 3*nm_std_MAV)
        self.dataset['class'][thresholded_windows] = 0





# this is a horrific mess. why do we need to load the sgt dataset when we may already have the features extracted? Why must the classifier store the dataset? 
# There is a lot fo change here
class EMGClassifier:
    def __init__(self, model_type=None, arguments=None, data_file=None, window_params = [250, 50, 1259], active_threshold=False, rejection_threshold=False):
        if data_file:
            self.arguments = arguments
            # final column = rep
            # second last column = class
            #   class of 100 = rest
            self.window_size = window_params[0]
            self.window_increment = window_params[1]
            self.frequency = window_params[2]
            dataset = SGT_Dataset(window_size=self.window_size, window_inc = self.window_increment, frequency = self.frequency)
            dataset.import_data(data_file)
            if active_threshold:
                dataset.active_threshold()
            self.active_channels = dataset.active_channels
            self.feature_extractor = Feature_Extractor(num_channels = self.active_channels.shape[0])

        if model_type == "manual":
            # install features - TODO: break this out to own function later
            if arguments[0] == 'TD':
                self.feature_list = ['MAV','ZC','SSC','WL']
            elif arguments[0] == 'TDPSD':
                self.feature_list = ['M0','M2','M4','SPARSI','IRF','WLF']
            # TODO: add a condition for every feature set

            # install classifier
            features = self.feature_extractor.extract_for_classifier(self.feature_list, dataset.dataset['data'])
            if arguments[1] == "LDA":
                self.classifier = LinearDiscriminantAnalysis()
                self.classifier.fit(features, dataset.dataset['class'])
                
            
            elif arguments[1] == "SVM":
                pass
                # TODO: add common classifiers here
            self.setup_rejection(rejection_threshold, features=features, class_labels=dataset.dataset['class'])

        elif model_type == "selection":

            # get features for selection
            if os.path.exists("tmp/"+data_file[:-4] + '.pkl'):
                 with open("tmp/"+data_file[:-4] + '.pkl', 'rb') as f:
                    prepared_dataset = pickle.load(f)
            else:
                prepared_dataset = self.feature_extractor.extract(self.feature_extractor.get_feature_list(), dataset.dataset['data'])
                self.feature_extractor.check_features(prepared_dataset)
                prepared_dataset['class'] = dataset.dataset['class']
                prepared_dataset['rep'] = dataset.dataset['rep']
                prepared_dataset['feature_list'] = self.feature_extractor.get_feature_list()
                with open("tmp/"+data_file[:-4] + '.pkl', 'wb') as f:
                    pickle.dump(prepared_dataset, f)

            # setup selection metric
            if arguments[0] == "accuracy": # This matches the combofield
                arguments.append("argmax") # this is the "good"/goal side of the spectrum for the metric
            elif arguments[0] == "activeaccuracy":
                arguments.append("argmax")
            elif arguments[0] == "msa":
                arguments.append("argmax")
            elif arguments[0] == "fe":
                arguments.append("argmax")


            # intiialize selector
            feature_selector = Feature_Selector(arguments, len(self.feature_extractor.get_feature_list()))
            

            # determine feature order according to criterion
            feature_selector.run_selection(prepared_dataset)
            feature_selector.print_results()

            # determine feature set from order.
            self.feature_list = feature_selector.get_feature_set(num=5)
            self.classifier   = LinearDiscriminantAnalysis()
            features = self.feature_extractor.extract_for_classifier(self.feature_list, dataset.dataset['data'])
            self.classifier.fit(features, dataset.dataset['class'])
            
            self.setup_rejection(rejection_threshold, features=features, class_labels=dataset.dataset['class'])
    
    def setup_rejection(self, rejection_threshold, features, class_labels):
        if isinstance(rejection_threshold, bool):
            if rejection_threshold:
                # do ROC
                rejection_search = 1-np.logspace(-2,0, 20)
                # not all of these are used for selection, but it is nice seeing the values w/ breakpoints
                active_error     = np.zeros_like(rejection_search)
                rejection_rate   = np.zeros_like(rejection_search)
                false_rejections = np.zeros_like(rejection_rate)
                accuracy         = np.zeros_like(rejection_search)
                criterion = np.zeros_like(rejection_search)
                #using the same dataset the classifier was trained for... get probabilities for all samples
                probabilties = self.classifier.predict_proba(features)
                predictions = np.argmax(probabilties, axis=1)
                active_predictions_pre_rejection = predictions != 0
                max_proba    = np.max(probabilties,axis=1)

                for t in range(rejection_search.shape[0]):
                    rejections = max_proba < rejection_search[t]
                    threshold_predictions = predictions.copy()
                    threshold_predictions[rejections] = 0

                    nm_predictions = threshold_predictions == 0
                    active_predictions_post_rejection = threshold_predictions != 0
                    # TODO: refactor these metrics into functions

                    # accuracy - post rejection, what is accuracy (we don't consider a rejection to be an error)
                    if sum(np.invert(rejections)) == 0:
                        accuracy[t] = 0 # when everything is rejected, it is not viable
                    else:
                        accuracy[t] = sum(threshold_predictions[np.invert(rejections)] == class_labels[np.invert(rejections)]) / class_labels[np.invert(rejections)].shape[0]

                    # active error - when should you have rejected but did not.
                    # sum of all the misclassifications that are not predicting no motion
                    # if sum(active_predictions_post_rejection) == 0:
                    #     active_error[t] = 1 # worst case, every active sample has been rejected
                    # else:
                    #     misclassifications = class_labels != threshold_predictions
                    #     active_misclassifications = (misclassifications.astype(int) + active_predictions_post_rejection.astype(int)) == 2
                    #     active_error[t] = sum(active_misclassifications) / sum(active_predictions_post_rejection)
                    
                    # false rejections - when you rejected but should not have
                    # sum of when it rejects but would have been correct
                    # active_rejections = (rejections.astype(int)+active_predictions_pre_rejection.astype(int)) == 2
                    # if sum(active_rejections) == 0:
                    #     false_rejections[t] = 0 # ideal case for this metric, no rejections of active class
                    # else:
                    #     false_rejections[t]  = sum(predictions[active_rejections] == class_labels[active_rejections])/sum(active_rejections)
                    
                    # we want a minimum rate of rejection
                    rejection_rate[t] = sum(rejections)/class_labels.shape[0]


                    # the optimum for this is at coordinates [0, 1]: no misclassifications to nm, perfect accuracy
                    criterion[t] = (rejection_rate[t])**2 + (accuracy[t]-1)**2

                    
                self.rejection = True
                self.rejection_threshold = rejection_search[np.argmin(criterion)]
            else:
                self.rejection = False
                self.rejection_threshold = np.nan
        elif isinstance(rejection_threshold, float):
            # if a float is given here instead
            self.rejection = True
            self.rejection_threshold = rejection_threshold

    def run(self, window, maskon=False):
        if maskon:
            # Make sure it goes channel, samples in pygame also
            window = window[self.active_channels,:]
        if len(window.shape) == 2:
            window = np.expand_dims(window, axis=0)
            # feature extraction library expects (batch, channels, samples)
            # often we will have only a single observation in the form (channels, smaples), so we can add a trivial dimension at the start in this case.
        features = self.feature_extractor.extract_for_classifier(self.feature_list, window)
        features[np.isnan(features)]=0
        features[np.isinf(features)]=0
        features[np.isneginf(features)]=0
        probabilities = self.classifier.predict_proba(features)
        if self.rejection:
            for w in range(probabilities.shape[0]):
                if np.max(probabilities[w,1:]) < self.rejection_threshold:
                    probabilities[w,0] = 1
                    probabilities[w,1:] = 0
        return probabilities

    def save(self, filename):
        classifier_dictionary = {
            'active_channels': self.active_channels,
            'feature_list': self.feature_list,
            'classifier': self.classifier,
            'window_params': [self.window_size, self.window_increment, self.frequency],
            'rejection_params': [self.rejection, self.rejection_threshold]}
        with open(filename, 'wb') as f:
            pickle.dump(classifier_dictionary, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            classifier_dictionary = pickle.load(f)
        
        self.active_channels = classifier_dictionary['active_channels']
        self.feature_extractor = Feature_Extractor(num_channels = self.active_channels.shape[0])
        self.feature_list = classifier_dictionary['feature_list']
        self.classifier = classifier_dictionary['classifier']
        self.window_size = classifier_dictionary['window_params'][0]
        self.window_inc  = classifier_dictionary['window_params'][1]
        self.frequency   = classifier_dictionary['window_params'][2]
        self.rejection   = classifier_dictionary['rejection_params'][0]
        self.rejection_threshold = classifier_dictionary['rejection_params'][1]



# This will have methods for the evaluation of 33 metrics... eventually 
#------|-----------------------------------------------------------
# RI   | Repeatability index
# mwRI | Mean Within-Repetition Repeatability Index
# swRI | Std. Dev. Within-Repetition Repeatability Index
# swSI | Std. Dev. Within-Trial Separability Index
# MSA  | Mean Semi-Principal Axes
# CD   | Centroid Drift
# MAV  | Mean Absolute Value
# SI   | Separability Index
# mSI  | Modified Separability Index
# mwSI | Mean Within-Trial Separability Index
# BD   | Bhattacharyya Distance
# KLD  | Kullback-Leibler Divergence
# HD   | Hellinger Distance Squared
# FDR  | Fisher's Discriminant Analysis
# VOR  | Volume of Overlap Region
# FE   | Feature Efficiency
# TSM  | Trace of Scatter Matrices
# DS   | Desirability Score
# CDM  | Class Discriminability Measure
# PU   | Purity
# rPU  | Rescaled Purity
# NS   | Neighborhood Separability
# rNS  | Rescaled Neighborhood Separability
# CE   | Collective Entropy
# rCE  | Rescaled Collective Entropy
# C    | Compactness
# rC   | Rescaled Compactness
# CA   | Classification Accuracy
# ACA  | Active Classification Accuracy
# UD   | Usable Data
# ICF  | Inter-Class Fraction
# IIF  | Intra-Inter Fraction
# TT   | Training time

# Implemented:
# CA   | Classification Accuracy
# ACA  | Active Classification Accuracy
# MSA  | Mean Semi-Principal Axes
# FE   | Feature Efficiency

# To Implement:
# PU   | Purity
# MAV  | Mean Absolute Value
# RI   | Repeatability index
# SI   | Separability Index

class Feature_Selector:
    def __init__(self, metric, num_features):
        self.metric = metric[0]
        self.metric_callback = getattr(self, self.metric)
        self.print_callback  = getattr(self, "print_" + self.metric + "_results")
        self.optimal_condition = getattr(np, metric[1])
        self.num_features = num_features
        self.feature_order = [] # This is the feature order by selection
        self.feature_order_id = [] # This is purely used in the display function -- 
    
    def get_feature_set(self, tol=None, num=None):
        # if there is not tol %dif in iteration, continue to add features, else stop
        feature_set = []
        

        if self.metric == 'accuracy':
            metric = self.mean_accuracy
        elif self.metric == 'activeaccuracy':
            metric = self.mean_accuracy
        elif self.metric == 'msa':
            metric = self.mean_msa
        elif self.metric == 'fe':
            metric = self.mean_fe
        
        # tolerence based selection
        if tol:
            change = tol+1e-5
            last_crit = -np.inf
            feature_id = 0
            while (100*change) > tol:
                # add feature to list
                feature_set.append(self.feature_order[feature_id])
                
                # get criterion for best feature
                cur_crit = metric[feature_id][self.feature_order_id[feature_id]]
                # check to see that it improves criterion by 1/2 percent
                change = (cur_crit - last_crit)/(cur_crit)
                last_crit = cur_crit
                feature_id += 1
        elif num:
            feature_set = self.feature_order[:num]
        return feature_set[:-1]




    def run_selection(self, data):
        features = data["feature_list"]
        features_included = []
        features_remaining = features
        for fi in range(0, self.num_features):
            criterion = self.metric_callback(data, features_included, features_remaining)
            chosen_feature = self.optimal_condition(criterion)
            self.feature_order.append(features_remaining[chosen_feature])
            self.feature_order_id.append(chosen_feature)
            features_included = self.feature_order
            features_remaining.remove(features_remaining[chosen_feature])

    def accuracy(self, data, features_included, features_remaining):

        if not hasattr(self, "mean_accuracy"):
            self.feature_list = data["feature_list"]
            self.mean_accuracy = {}
            self.std_accuracy = {}

        num_subjects = 1
        num_reps     = len(np.unique(data["rep"]))
        
        mean_accuracy = np.zeros(len(features_remaining))
        std_accuracy  = np.zeros(len(features_remaining))

        prior_features = np.array([])
        for feature in features_included:
            prior_features = np.hstack([prior_features, data[feature]]) if prior_features.size else data[feature]
        
        for i,feature in enumerate(features_remaining):
            iteration_features = np.hstack([prior_features, data[feature]])  if prior_features.size else data[feature]
            feature_accuracy = np.zeros((num_subjects, num_reps))
            # do a k-fold cross-validation within subject
            for rep in range(num_reps):
                train_class    = data["class"][data["rep"] != rep]
                train_features = iteration_features[data["rep"] != rep, :]
                test_class     = data["class"][data["rep"] == rep]
                test_features  = iteration_features[data["rep"] == rep, :]

                mdl = LinearDiscriminantAnalysis()
                mdl.fit(train_features, train_class)
                predictions = mdl.predict(test_features)
                feature_accuracy[0, rep] = sum(predictions == test_class) / test_class.shape[0]
            mean_accuracy[i] = np.mean(np.mean(feature_accuracy, axis=1))
            std_accuracy[i]  = np.std(np.mean(feature_accuracy,axis=1))

        self.mean_accuracy[len(self.mean_accuracy.keys())] = mean_accuracy
        self.std_accuracy[len(self.std_accuracy.keys())] = std_accuracy

        return mean_accuracy

    def activeaccuracy(self, data, features_included, features_remaining):

        if not hasattr(self, "mean_accuracy"):
            self.feature_list = data["feature_list"]
            self.mean_accuracy = {}
            self.std_accuracy = {}

        num_subjects = 1
        num_reps     = len(np.unique(data["rep"]))
        
        mean_accuracy = np.zeros(len(features_remaining))
        std_accuracy  = np.zeros(len(features_remaining))

        prior_features = np.array([])
        for feature in features_included:
            prior_features = np.hstack([prior_features, data[feature]]) if prior_features.size else data[feature]
        
        for i,feature in enumerate(features_remaining):
            iteration_features = np.hstack([prior_features, data[feature]])  if prior_features.size else data[feature]
            feature_accuracy = np.zeros((num_subjects, num_reps))
            # do a k-fold cross-validation within subject
            for rep in range(num_reps):
                train_class    = data["class"][data["rep"] != rep]
                train_features = iteration_features[data["rep"] != rep, :]
                test_class     = data["class"][data["rep"] == rep]
                test_features  = iteration_features[data["rep"] == rep, :]

                mdl = LinearDiscriminantAnalysis()
                mdl.fit(train_features, train_class)
                predictions = mdl.predict(test_features)
                # active accuracy doesn't account for misclassifications that predicted no motion (class 0)
                misclassification_locations    = predictions != test_class
                # TODO: MAKE SURE THIS INDEX IS ACTUALLY THE NM INDEX
                active_errors                  = sum(misclassification_locations) - sum(predictions[misclassification_locations] == 0)
                feature_accuracy[0, rep] = 1 - active_errors / test_class.shape[0]
            
            mean_accuracy[i] = np.mean(np.mean(feature_accuracy, axis=1))
            std_accuracy[i]  = np.std(np.mean(feature_accuracy,axis=1))

        self.mean_accuracy[len(self.mean_accuracy.keys())] = mean_accuracy
        self.std_accuracy[len(self.std_accuracy.keys())] = std_accuracy

        return mean_accuracy

    def msa(self, data, features_included, features_remaining):

        if not hasattr(self, "mean_msa"):
            self.feature_list = data["feature_list"]
            self.mean_msa = {}
            self.std_msa = {}

        num_subjects = 1
        num_classes  = len(np.unique(data["class"]))
        class_list   =  np.unique(data["class"])

        mean_msa = np.zeros(len(features_remaining))
        std_msa  = np.zeros(len(features_remaining))

        prior_features = np.array([])
        for feature in features_included:
            prior_features = np.hstack([prior_features, data[feature]]) if prior_features.size else data[feature]

        for i,feature in enumerate(features_remaining):
            iteration_features = np.hstack([prior_features, data[feature]])  if prior_features.size else data[feature]
            msa = np.zeros((num_subjects, num_classes))
            norm_iteration_features = (iteration_features - iteration_features.mean()) / iteration_features.std()
            for c in range(num_classes):
                class_features = norm_iteration_features[data["class"] == class_list[c], :]

                pca = PCA()
                pca.fit(class_features)
                for comp in pca.components_:
                    msa[0, c] += np.abs(comp).prod() ** (1.0 / len(comp))
            msa = np.mean(msa,1)
            mean_msa[i] = np.mean(msa)
            std_msa[i]  = np.std(msa)
        
        self.mean_msa[len(self.mean_msa.keys())] = mean_msa
        self.std_msa[len(self.std_msa.keys())] = std_msa

        return mean_msa

    def fe(self, data, features_included, features_remaining):
        # FE is not actually a sequential metric...
        # we could do it all in one go and have the complexity be N
        # but to keep the format of the selection library, we will do it sequentially
        # calls for a refactor later.

        if not hasattr(self, "mean_fe"):
            self.feature_list = data["feature_list"]
            self.mean_fe = {}

        num_subjects = 1
        class_list   =  np.unique(data["class"])

        # N are active channels, we don't use NM here (as described in Nawfel paper)
        class_list = class_list[class_list != 0]
        num_classes  = len(class_list)

        mean_fe = np.zeros(len(features_remaining))

        # we don't need to do anything with the features already included
        for i,feature in enumerate(features_remaining):
            iteration_features = data[feature]
            # determine overlap between class j and class i
            sum_class_efficiency = 0
            for cj in class_list:
                cj_ids = data["class"] == cj
                cardinality_j = sum(cj_ids)
                
                for ci in class_list:
                    max_class_efficiency = 0
                    if ci == cj:
                        continue
                    else:
                        ci_ids = data["class"] == ci
                        cardinality_i = sum(ci_ids)
                        # Sk is the number of included points, point is included if:
                        max_feature_class_efficiency = 0
                        for k in range(iteration_features.shape[1]):
                            min_j = min(iteration_features[cj_ids,k])
                            min_i = min(iteration_features[ci_ids,k])
                            max_of_mins = max([min_j, min_i])

                            max_j = max(iteration_features[cj_ids,k])
                            max_i = max(iteration_features[ci_ids==False,k])
                            min_of_maxes = min([max_j, max_i])

                            included_points = (iteration_features[ci_ids+cj_ids,k] < min_of_maxes).astype(np.int32) + (iteration_features[ci_ids+cj_ids,k] > max_of_mins).astype(np.int32)
                            Sk = sum(included_points==2)
                            class_efficiency = (cardinality_i + cardinality_j - Sk) / (cardinality_i + cardinality_j)
                            if class_efficiency > max_feature_class_efficiency:
                                max_feature_class_efficiency = class_efficiency
                    if max_feature_class_efficiency > max_class_efficiency:
                        max_class_efficiency = max_feature_class_efficiency
                    
                sum_class_efficiency += max_class_efficiency
            mean_fe[i] = sum_class_efficiency / (num_classes-1)
        self.mean_fe[len(self.mean_fe.keys())] = mean_fe
        return mean_fe
                    










    def print_results(self):
        self.print_callback()
    
    def print_accuracy_results(self):
        # longest feature name
        longest = 11
        for f in self.feature_order:
            if longest < len(f):
                longest = len(f)
        header_row = "iter".center(longest)
        for f in range(len(self.feature_order)):
            header_row += "|" + self.feature_order[f].center(longest)
        print(header_row)
        print('='*((longest+1)*(len(self.feature_order)+1)))


        for i in range(len(self.feature_order)):
            row = str(i).center(longest)
            mean_acc = self.mean_accuracy[i][self.feature_order_id[i:]] * 100
            std_acc  = self.std_accuracy[i][self.feature_order_id[i:]] * 100

            for j in range(i):
                row += "|" + " ".center(longest)
            for ii,f in enumerate(range(i, len(self.feature_order))):
                row += "|" + ("{:.1f}+{:.1f}".format(mean_acc[ii],std_acc[ii])).center(longest)
            print(row)

    def print_activeaccuracy_results(self):
        self.print_accuracy_results()

    def print_msa_results(self):
        # longest feature name
        longest = 11
        for f in self.feature_order:
            if longest < len(f):
                longest = len(f)
        header_row = "iter".center(longest)
        for f in range(len(self.feature_order)):
            header_row += "|" + self.feature_order[f].center(longest)
        print(header_row)
        print('='*((longest+1)*(len(self.feature_order)+1)))


        for i in range(len(self.feature_order)):
            row = str(i).center(longest)
            mean_msa = self.mean_msa[i][self.feature_order_id[i:]]
            std_msa  = self.std_msa[i][self.feature_order_id[i:]]

            for j in range(i):
                row += "|" + " ".center(longest)
            for ii,f in enumerate(range(i, len(self.feature_order))):
                row += "|" + ("{:.2f}+{:.2f}".format(mean_msa[ii],std_msa[ii])).center(longest)
            print(row)

    def print_fe_results(self):
        longest = 11
        for f in self.feature_order:
            if longest < len(f):
                longest = len(f)
        header_row = "iter".center(longest)
        for f in range(len(self.feature_order)):
            header_row += "|" + self.feature_order[f].center(longest)
        print(header_row)
        print('='*((longest+1)*(len(self.feature_order)+1)))


        for i in range(len(self.feature_order)):
            row = str(i).center(longest)
            mean_fe = self.mean_fe[i][self.feature_order_id[i:]]

            for j in range(i):
                row += "|" + " ".center(longest)
            for ii,f in enumerate(range(i, len(self.feature_order))):
                row += "|" + ("{:.2f}".format(mean_fe[ii])).center(longest)
            print(row)

class Feature_Extractor:
    """
    Feature extraction class including feature groups, feature list, and feature extraction code.
    Parameters
    ----------
    num_channels: int > 0
        The number of EMG channels. 
    feature_list: array_like (optional)
        A list of features to remember within the object. Used for when this object is passed to an EMGClassifier.
    feature_group: string (optional)
         A feature group to remember. Used for when this object is passed to an EMGClassifier.
    """
    def __init__(self, num_channels, feature_list=[], feature_group=None):
        self.num_channels = num_channels

    def get_feature_groups(self):
        """Gets a list of all available feature groups.
        Returns
        ----------
        dictionary
            A dictionary with the all available feature groups.
        """
        feature_groups = {'HTD': ['MAV', 'ZC', 'SSC', 'WL'],
                          'TD4': ['LS', 'MFL', 'MSR', 'WAMP'],
                          'TD9': ['LS', 'MFL', 'MSR', 'WAMP', 'ZC', 'RMS', 'IAV', 'DASDV', 'VAR'],
                          'TDPSD': ['M0','M2','M4','SPARSI','IRF','WLF']}
        return feature_groups

    def get_feature_list(self):
        """Gets a list of all available features.
        Returns
        ----------
        array_like
            A list of all available features.
        """
        feature_list = ['MAV',
                        'ZC',
                        'SSC',
                        'WL',
                        'LS',
                        'MFL',
                        'MSR',
                        'WAMP',
                        'RMS',
                        'IAV',
                        'DASDV',
                        'VAR',
                        'M0',
                        'M2',
                        'M4',
                        'SPARSI',
                        'IRF',
                        'WLF',
                        'AR', 
                        'CC',
                        'LD',
                        'MAVFD',
                        'MAVSLP',
                        'MDF',
                        'MNF',
                        'MNP',
                        'MPK',
                        'SKEW',
                        'KURT']
        return feature_list
    
    def extract_feature_group(self, feature_group, windows):
        """Extracts a group of features.
        Parameters
        ----------
        feature_group: string
            The group of features to extract. Valid options include: 'HTD', 'TD4', 'TD9' and 'TDPSD'.
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        dictionary
            A dictionary where each key is a specific feature and its value is a list of the computed 
            features for each window.
        """
        features = {}
        if not feature_group in self.get_feature_groups():
            return features
        return self.extract_features(self.get_feature_groups()[feature_group], windows)

    def extract(self, feature_list, windows):
        """Extracts a list of features.
        Parameters
        ----------
        feature_list: list
            The group of features to extract. Run get_feature_list() or checkout the github documentation 
            to find an up-to-date feature list.  
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        dictionary
            A dictionary where each key is a specific feature and its value is a list of the computed 
            features for each window.
        """
        features = {}
        for feature in feature_list:
            if feature in self.get_feature_list():
                print(feature)
                method_to_call = getattr(self, 'get' + feature + 'feat')
                features[feature] = method_to_call(windows)
            
        return features

    def extract_for_classifier(self, feature_list, windows):
        for i, feature in enumerate(feature_list):
            method_to_call = getattr(self, 'get' + feature + 'feat')
            if i == 0:
                features = method_to_call(windows)
            else:
                features = np.column_stack((features, method_to_call(windows)))
        return features

    def check_features(self, features):
        violations=0
        feature_list = list(features.keys())
        for fk in feature_list:
            if (features[fk] == np.nan).any():
                violations += 1
            if (features[fk] == np.inf).any():
                violations += 1
            if (features[fk] == -1*np.inf).any():
                violations += 1
            for ch in range(features[fk].shape[1]):
                if (features[fk][:,ch] == features[fk][0,ch]).all():
                    violations += 1
        print(f"violations: {violations}")

    '''
    -----------------------------------------------------------------------------------------
    The following methods are all feature extraction methods. They should follow the same
    format that already exists (i.e., get<FEAT_ABBREVIATION>feat). The feature abbreviation
    should be added to the get feature_list function. 
    '''

    def getMAVfeat(self, windows):
        """Extract Mean Absolute Value (MAV) feature.
        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        feat = np.mean(np.abs(windows),2)
        return feat
    
    # TODO: Add threshold
    def getZCfeat(self, windows):
        """Extract Zero Crossings (ZC) feature.
        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        sgn_change = np.diff(np.sign(windows),axis=2)
        neg_change = sgn_change == -2
        pos_change = sgn_change ==  2
        feat_a = np.sum(neg_change,2)
        feat_b = np.sum(pos_change,2)
        return feat_a+feat_b
    

    def getSSCfeat(self, windows,threshold=0):
        """Extract Slope Sign Change (SSC) feature.
        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        w_2 = windows[:,:,2:]
        w_1 = windows[:,:,1:-1]
        w_0 = windows[:,:,:-2]
        con = (((w_1-w_0)*(w_1-w_2)) >= threshold)
        return np.sum(con,axis=2)

    def getWLfeat(self, windows):
        """Extract Waveform Length (WL) feature.
        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        feat = np.sum(np.abs(np.diff(windows,axis=2)),2)
        return feat

    def getLSfeat(self, windows):
        """Extract L-Score (LS) feature.
        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        feat = np.zeros((windows.shape[0],windows.shape[1]))
        for w in range(0, windows.shape[0],1):
            for c in range(0, windows.shape[1],1):
                tmp = self.__lmom(np.reshape(windows[w,c,:],(1,windows.shape[2])),2)
                feat[w,c] = tmp[0,1]
        return feat

    def __lmom(self, signal, nL):
        # same output to matlab when ones vector of various sizes are input
        b = np.zeros((1,nL-1))
        l = np.zeros((1,nL-1))
        b0 = np.zeros((1,1))
        b0[0,0] = np.mean(signal)
        n = signal.shape[1]
        signal = np.sort(signal, axis=1)
        for r in range(1,nL,1):
            num = np.tile(np.asarray(range(r+1,n+1)),(r,1))  - np.tile(np.asarray(range(1,r+1)),(1,n-r))
            num = np.prod(num,axis=0)
            den = np.tile(np.asarray(n),(1,r)) - np.asarray(range(1,r+1))
            den = np.prod(den)
            b[r-1] = 1/n * np.sum(num / den * signal[0,r:n])
        tB = np.concatenate((b0,b))
        B = np.flip(tB,0)
        for i in range(1, nL, 1):
            Spc = np.zeros((B.shape[0]-(i+1),1))
            Coeff = np.concatenate((Spc, self.__LegendreShiftPoly(i)))
            l[0,i-1] = np.sum(Coeff * B)
        L = np.concatenate((b0, l),1)

        return L

    def __LegendreShiftPoly(self, n):
        # Verified: this has identical function to MATLAB function for n = 2:10 (only 2 is used to compute LS feature)
        pk = np.zeros((n+1,1))
        if n == 0:
            pk = 1
        elif n == 1:
            pk[0,0] = 2
            pk[1,0] = -1
        else:
            pkm2 = np.zeros(n+1)
            pkm2[n] = 1
            pkm1 = np.zeros(n+1)
            pkm1[n] = -1
            pkm1[n-1] = 2

            for k in range(2,n+1,1):
                pk = np.zeros((n+1,1))
                for e in range(n-k+1,n+1,1):
                    pk[e-1] = (4*k-2)*pkm1[e]+ (1-2*k)*pkm1[e-1] + (1-k) * pkm2[e-1]
                pk[n,0] = (1-2*k)*pkm1[n] + (1-k)*pkm2[n]
                pk = pk/k

                if k < n:
                    pkm2 = pkm1
                    pkm1 = pk

        return pk

    def getMFLfeat(self, windows):
        """Extract Maximum Fractal Length (MFL) feature.
        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        feat = np.log10(np.sum(np.abs(np.diff(windows, axis=2)),axis=2))
        return feat

    def getMSRfeat(self, windows):
        """Extract Mean Squared Ratio (MSR) feature.
        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        feat = np.abs(np.mean(np.sqrt(windows.astype('complex')),axis=2))
        return feat

    def getWAMPfeat(self, windows, threshold=5e-6): # TODO: add optimization if threshold not passed, need class labels
        """Extract Willison Amplitude (WAMP) feature.
        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        feat = np.sum(np.abs(np.diff(windows, axis=2)) > threshold, axis=2)
        return feat

    def getRMSfeat(self, windows):
        """Extract Root Mean Square (RMS) feature.
        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        feat = np.sqrt(np.mean(np.square(windows),2))
        return feat

    def getIAVfeat(self, windows):
        """Extract Integral of Absolute Value (IAV) feature.
        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        feat = np.sum(np.abs(windows),axis=2)
        return feat

    def getDASDVfeat(self, windows):
        """Difference Absolute Standard Deviation Value (DASDV) feature.
        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        feat = np.sqrt(np.mean(np.diff(windows,axis=2)**2,axis=2))
        return feat

    def getVARfeat(self, windows):
        """Extract Variance (VAR) feature.
        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        feat = np.var(windows,axis=2)
        return feat

    def getM0feat(self, windows):
        """Extract First Temporal Moment (M0) feature.
        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        
        def closure(w):
            m0 = np.sqrt(np.sum(w**2,axis=2))/(w.shape[2]-1)
            m0 = m0 ** 0.1 / 0.1
            return np.log(np.abs(m0))
        m0_ebp=closure(windows)
        m0_efp=closure(np.log(windows**2+np.spacing(1)))

        num=-2*np.multiply(m0_efp,m0_ebp)
        den=np.multiply(m0_efp, m0_efp) + np.multiply(m0_ebp, m0_ebp)

        #Feature extraction goes here
        return num/den
    
    def getM2feat(self, windows):
        """Extract Second Temporal Moment (M2) feature.
        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        def closure(w):
            m0 = np.sqrt(np.sum(w**2,axis=2))/(w.shape[2]-1)
            m0 = m0 ** 0.1 / 0.1
            d1 = np.diff(w, n=1, axis=2)
            m2 = np.sqrt(np.sum(d1 **2, axis=2)/ (w.shape[2]-1))
            m2 = m2 ** 0.1 / 0.1
            return np.log(np.abs(m0-m2))
        m2_ebp=closure(windows)
        m2_efp=closure(np.log(windows**2+np.spacing(1)))

        num=-2*np.multiply(m2_efp,m2_ebp)
        den=np.multiply(m2_efp, m2_efp) + np.multiply(m2_ebp, m2_ebp)
        
        return num/den

    def getM4feat(self, windows):
        """Extract Fourth Temporal Moment (M4) feature.
        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        def closure(w):
            m0 = np.sqrt(np.sum(w**2,axis=2))/(w.shape[2]-1)
            m0 = m0 ** 0.1 / 0.1
            d1 = np.diff(w, n=1, axis=2)
            d2 = np.diff(d1, n=1, axis=2)
            m4 = np.sqrt(np.sum(d2 **2, axis=2)/ (w.shape[2]-1))
            m4 = m4 ** 0.1 / 0.1
            return np.log(np.abs(m0-m4))
        m4_ebp=closure(windows)
        m4_efp=closure(np.log(windows**2+np.spacing(1)))
        
        num=-2*np.multiply(m4_efp,m4_ebp)
        den=np.multiply(m4_efp, m4_efp) + np.multiply(m4_ebp, m4_ebp)
        
        return num/den
    
    def getSPARSIfeat(self, windows):
        """Extract Sparsness (SPARSI) feature.
        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        def closure(w):
            m0 = np.sqrt(np.sum(w**2,axis=2))/(w.shape[2]-1)
            m0 = m0 ** 0.1 / 0.1
            d1 = np.diff(w, n=1, axis=2)
            m2 = np.sqrt(np.sum(d1 **2, axis=2)/ (w.shape[2]-1))
            m2 = m2 ** 0.1 / 0.1
            d2 = np.diff(d1, n=1, axis=2)
            m4 = np.sqrt(np.sum(d2 **2, axis=2)/ (w.shape[2]-1))
            m4 = m4 ** 0.1 / 0.1
            sparsi = np.sqrt(np.abs((m0-m2)*(m0-m4)))/m0
            return np.log(np.abs(sparsi))
        sparsi_ebp=closure(windows)
        sparsi_efp=closure(np.log(windows**2+np.spacing(1)))
        
        num=-2*np.multiply(sparsi_efp,sparsi_ebp)
        den=np.multiply(sparsi_efp, sparsi_efp) + np.multiply(sparsi_ebp, sparsi_ebp)

        return num/den

    def getIRFfeat(self, windows):
        """Extract Irregularity Factor (IRF) feature.
        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        def closure(w):
            m0 = np.sqrt(np.sum(w**2,axis=2))/(w.shape[2]-1)
            m0 = m0 ** 0.1 / 0.1
            d1 = np.diff(w, n=1, axis=2)
            m2 = np.sqrt(np.sum(d1 **2, axis=2)/ (w.shape[2]-1))
            m2 = m2 ** 0.1 / 0.1
            d2 = np.diff(d1, n=1, axis=2)
            m4 = np.sqrt(np.sum(d2 **2, axis=2)/ (w.shape[2]-1))
            m4 = m4 ** 0.1 / 0.1
            irf = m2/np.sqrt(m0*m4)
            return np.log(np.abs(irf))
        irf_ebp=closure(windows)
        irf_efp=closure(np.log(windows**2+np.spacing(1)))
        
        num=-2*np.multiply(irf_efp,irf_ebp)
        den=np.multiply(irf_efp, irf_efp) + np.multiply(irf_ebp, irf_ebp)

        return num/den

    def getWLFfeat(self, windows):
        """Waveform Length Factor (WLF) feature.
        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        def closure(w):
            d1 = np.diff(w, n=1, axis=2)
            
            d2 = np.diff(d1, n=1, axis=2)
            
            wlf = np.sqrt(np.sum(np.abs(d1),axis=2)/np.sum(np.abs(d2),axis=2))
            return np.log(np.abs(wlf))
        wlf_ebp=closure(windows)
        wlf_efp=closure(np.log(windows**2+np.spacing(1)))
        
        num=-2*np.multiply(wlf_efp,wlf_ebp)
        den=np.multiply(wlf_efp, wlf_efp) + np.multiply(wlf_ebp, wlf_ebp)

        return num/den


    def getARfeat(self, windows, order=4):
        """Extract Autoregressive Coefficients (AR) feature.
        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """


        # the way they are computed via the matlab script: linear predictive filter

        feature = np.reshape(lpc(windows, order=order,axis=2)[:,:,1:],(windows.shape[0],order*windows.shape[1]),order="C")
        return feature

    def getCCfeat(self, windows, order =4):
        """Extract Cepstral Coefficient (CC) feature.
        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        AR = self.getARfeat(windows, order)
        cc = np.zeros_like(AR)
        cc[:,::order] = -1*AR[:,::order]
        if order > 2:
            for p in range(1,order):
                for l in range(1, p):
                    cc[:,p::order] = cc[:,p::order]+(AR[:,p::order] * cc[:,p-l::order] * (1-(l/p)))
                cc[:,p::order] = -1*AR[:,p::order]-cc[:,p::order]
        return cc
    
    def getLDfeat(self, windows):
        """Extract Log Detector (LD) feature.
        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        return np.exp(np.mean(np.log(np.abs(windows)+1), 2))

    def getMAVFDfeat(self, windows):
        """Extract Mean Absolute Value First Difference (MAVFD) feature.
        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        mavfd = np.mean(np.abs(np.diff(windows,axis=2)),axis=2)
        return mavfd

    def getMAVSLPfeat(self, windows, segment=2):
        """Extract Mean Absolute Value Slope (MAVSLP) feature.
        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        m = int(round(windows.shape[2]/segment))
        mav = []
        mavslp = []
        for i in range(0,segment):
            mav.append(np.mean(np.abs(windows[:,:,i*m:(i+1)*m]), axis=2))
        for i in range (0, segment-1):
            mavslp.append(mav[i+1]- mav[i])
        return np.asarray(mavslp).squeeze()

    def getMDFfeat(self, windows,fs=1000):
        """Extract Median Frequency (MDF) feature.
        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        def closure(winsize):
            return 1 if winsize==0 else 2**math.ceil(math.log2(winsize))
        nextpow2 = closure(windows.shape[2])
        spec = np.fft.fft(windows,nextpow2, axis=2)/windows.shape[2]
        spec = spec[:,:,0:int(nextpow2/2)]
        POW = np.real(spec * np.conj(spec))
        totalPOW = np.sum(POW, axis=2)
        cumPOW   = np.cumsum(POW, axis=2)
        medfreq = np.zeros((windows.shape[0], windows.shape[1]))
        for i in range(0, windows.shape[0]):
            for j in range(0, windows.shape[1]):
                medfreq[i,j] = (fs/2)*np.argwhere(cumPOW[i,j,:] > totalPOW[i,j] /2)[0]/(nextpow2/2)
        return medfreq

    def getMNFfeat(self, windows, fs=1000):
        """Extract Mean Frequency (MNF) feature.
        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        def closure(winsize):
            return 1 if winsize==0 else 2**math.ceil(math.log2(winsize))
        nextpow2 = closure(windows.shape[2])
        spec = np.fft.fft(windows, n=nextpow2,axis=2)/windows.shape[2]
        f = np.fft.fftfreq(nextpow2)*fs
        spec = spec[:,:,0:int(round(spec.shape[2]/2))]
        f = f[0:int(round(nextpow2/2))]
        f = np.repeat(f[np.newaxis, :], spec.shape[0], axis=0)
        f = np.repeat(f[:, np.newaxis,:], spec.shape[1], axis=1)
        POW = spec * np.conj(spec)
        return np.real(np.sum(POW*f,axis=2)/np.sum(POW,axis=2))

    def getMNPfeat(self, windows):
        """Extract Mean Power (MNP) feature.
        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        def closure(winsize):
            return 1 if winsize==0 else 2**math.ceil(math.log2(winsize))
        nextpow2 = closure(windows.shape[2])
        spec = np.fft.fft(windows,n=nextpow2,axis=2)/windows.shape[2]
        spec = spec[:,:,0:int(round(nextpow2/2))]
        POW  = np.real(spec[:,:,:int(nextpow2)]*np.conj(spec[:,:,:int(nextpow2)]))
        return np.sum(POW, axis=2)/POW.shape[2]

    def getMPKfeat(self, windows):
        """Extract (MPK) feature.
        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        return windows.max(axis=2)

    def getSKEWfeat(self, windows):
        """Extract Skewness (SKEW) feature.
        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        return skew(windows, axis=2)

    def getKURTfeat(self, windows):
        """Extract Kurtosis (KURT) feature.
        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        return kurtosis(windows, axis=2, fisher=False)
