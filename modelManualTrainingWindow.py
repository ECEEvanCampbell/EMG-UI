from utils import PageWindow
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QComboBox, QMessageBox, QLabel, QLineEdit
from pr_utils import EMGClassifier
import os

class ManualModelTrainingWindow(PageWindow):
    def __init__(self, basewindow):
        super().__init__()

        self.basewindow = basewindow 

        self.setObjectName("manualmodeltraining_window")
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Manual Model Training")
        self.UiComponents()


    def UiComponents(self):
        ## CENTRAL WIDGET - this is a parent widget for formatting
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.setCentralWidget(self.centralwidget)

        ## MANUAL MODEL STUFF

        self.manual_data_filename_label = QLabel(self.centralwidget)
        self.manual_data_filename_label.setGeometry(QtCore.QRect(5,40,190,30))
        self.manual_data_filename_label.setText("Data for Training")

        self.manual_data_filename_input = QLineEdit(self.centralwidget)
        self.manual_data_filename_input.setGeometry(QtCore.QRect(5,75,190,30))
        self.manual_data_filename_input.setText("SGT_SXX.csv")

        self.manual_window_size_label = QLabel(self.centralwidget)
        self.manual_window_size_label.setGeometry(QtCore.QRect(5,110,190,30))
        self.manual_window_size_label.setText("Window Size (ms)")

        self.manual_window_size_input = QLineEdit(self.centralwidget)
        self.manual_window_size_input.setGeometry(QtCore.QRect(5,145, 190, 30))
        self.manual_window_size_input.setText("0.150")

        self.manual_window_inc_label = QLabel(self.centralwidget)
        self.manual_window_inc_label.setGeometry(QtCore.QRect(5,180,190,30))
        self.manual_window_inc_label.setText("Window Increment (ms)")

        self.manual_window_inc_input = QLineEdit(self.centralwidget)
        self.manual_window_inc_input.setGeometry(QtCore.QRect(5,215,190,30))
        self.manual_window_inc_input.setText("0.050")

        self.manual_model_feature_label = QLabel(self.centralwidget)
        self.manual_model_feature_label.setGeometry(QtCore.QRect(5, 250, 190, 30))
        self.manual_model_feature_label.setText("Feature Set")

        self.manual_model_feature_box = QComboBox(self.centralwidget)
        self.manual_model_feature_box.setGeometry(QtCore.QRect(5, 285, 190, 30))
        self.manual_model_feature_box.addItems(["TD","TDPSD","LSF4"])

        self.manual_model_classifier_label = QLabel(self.centralwidget)
        self.manual_model_classifier_label.setGeometry(QtCore.QRect(5, 320, 190, 30))
        self.manual_model_classifier_label.setText("Classifier")

        self.manual_model_classifier_box = QComboBox(self.centralwidget)
        self.manual_model_classifier_box.setGeometry(QtCore.QRect(5, 355, 190, 30))
        self.manual_model_classifier_box.addItems(["LDA", "SVM"])

        self.manual_button = QtWidgets.QPushButton(self.centralwidget)
        self.manual_button.setGeometry(QtCore.QRect(5, 390, 190, 30))
        self.manual_button.setText("Make Model")
        self.manual_button.clicked.connect(self.make_manual_model)

    def make_manual_model(self):
        features = self.manual_model_feature_box.currentText()
        classifier = self.manual_model_classifier_box.currentText()
        data_filename = self.manual_data_filename_input.text()
        window_size = self.manual_window_size_label.text()
        window_increment = self.manual_window_inc_input.text()

        frequency = 1259 # TODO: dont't just hardcode this for delsys

        window_parameters = [window_size, window_increment, frequency]
        self.basewindow.model = EMGClassifier("manual", [features, classifier], data_filename, window_parameters)
        # POPUP saying the classifier is trained
        msg = QMessageBox()
        msg.setWindowTitle("Classifier Trained")
        msg.setText("Classifier is trained")
        msg.setIcon(QMessageBox.Critical)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_() # shows the message box