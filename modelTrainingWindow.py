from utils import PageWindow
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QComboBox, QMessageBox, QLabel, QLineEdit
from pr_utils import EMGClassifier
import os

class ModelTrainingWindow(PageWindow):
    def __init__(self, basewindow):
        super().__init__()

        self.basewindow = basewindow 

        self.setObjectName("modeltraining_window")
        self.initUI()

    def initUI(self):
        self.setWindowTitle("ModelTraining")
        self.UiComponents()

    def goToMain(self):
        self.goto("main")

    def UiComponents(self):
        ## CENTRAL WIDGET - this is a parent widget for formatting
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.setCentralWidget(self.centralwidget)

        ## MANUAL MODEL STUFF
        self.manual_model_label = QLabel(self.centralwidget)
        self.manual_model_label.setGeometry(QtCore.QRect(5, 5, 190, 30))
        self.manual_model_label.setText("Manual Model Construction")

        self.manual_data_filename_label = QLabel(self.centralwidget)
        self.manual_data_filename_label.setGeometry(QtCore.QRect(5,40,190,30))
        self.manual_data_filename_label.setText("Data for Training")

        self.manual_data_filename_input = QLineEdit(self.centralwidget)
        self.manual_data_filename_input.setGeometry(QtCore.QRect(5,75,190,30))
        self.manual_data_filename_input.setText("SGT_SXX.csv")


        self.manual_model_feature_label = QLabel(self.centralwidget)
        self.manual_model_feature_label.setGeometry(QtCore.QRect(5, 110, 190, 30))
        self.manual_model_feature_label.setText("Feature Set")

        self.manual_model_feature_box = QComboBox(self.centralwidget)
        self.manual_model_feature_box.setGeometry(QtCore.QRect(5, 145, 190, 30))
        self.manual_model_feature_box.addItems(["TD","TDPSD","LSF4"])

        self.manual_model_classifier_label = QLabel(self.centralwidget)
        self.manual_model_classifier_label.setGeometry(QtCore.QRect(5, 180, 190, 30))
        self.manual_model_classifier_label.setText("Classifier")

        self.manual_model_classifier_box = QComboBox(self.centralwidget)
        self.manual_model_classifier_box.setGeometry(QtCore.QRect(5, 215, 190, 30))
        self.manual_model_classifier_box.addItems(["LDA", "SVM"])

        self.manual_button = QtWidgets.QPushButton(self.centralwidget)
        self.manual_button.setGeometry(QtCore.QRect(5, 250, 190, 30))
        self.manual_button.setText("Make Model")
        self.manual_button.clicked.connect(self.make_manual_model)

        ## MODEL SELECTION STUFF

        self.select_model_label = QLabel(self.centralwidget)
        self.select_model_label.setGeometry(QtCore.QRect(300, 5, 190, 30))
        self.select_model_label.setText("Automatic Model Selection")

        self.select_data_filename_label = QLabel(self.centralwidget)
        self.select_data_filename_label.setGeometry(QtCore.QRect(300,40,190,30))
        self.select_data_filename_label.setText("Data for Training")

        self.select_data_filename_input = QLineEdit(self.centralwidget)
        self.select_data_filename_input.setGeometry(QtCore.QRect(300,75,190,30))
        self.select_data_filename_input.setText("SGT_SXX.csv")

        self.select_model_metric_label = QLabel(self.centralwidget)
        self.select_model_metric_label.setGeometry(QtCore.QRect(300, 110, 190,30))
        self.select_model_metric_label.setText("Selection Metric")

        self.select_model_metric_box = QComboBox(self.centralwidget)
        self.select_model_metric_box.setGeometry(QtCore.QRect(300, 145, 190,30))
        self.select_model_metric_box.addItems(["accuracy","activeaccuracy", "MSA", "FE"])

        self.select_button = QtWidgets.QPushButton(self.centralwidget)
        self.select_button.setGeometry(QtCore.QRect(300,180, 190, 30))
        self.select_button.setText("Make Model")
        self.select_button.clicked.connect(self.make_automatic_model)
        

        ## Model SAVE/LOAD stuff
        self.model_label = QLabel(self.centralwidget)
        self.model_label.setText("Save/Load Model")
        self.model_label.setGeometry(QtCore.QRect(600,5,190,30))

        self.model_filename = QLineEdit(self.centralwidget)
        self.model_filename.setGeometry(QtCore.QRect(600, 40, 190, 30))
        self.model_filename.setText("MDL_SXX_AXX.pkl")

        self.model_save_button = QtWidgets.QPushButton(self.centralwidget)
        self.model_save_button.setGeometry(600, 75, 90, 30)
        self.model_save_button.setText("Save Model")
        self.model_save_button.clicked.connect(self.save_model)

        self.model_load_button = QtWidgets.QPushButton(self.centralwidget)
        self.model_load_button.setGeometry(QtCore.QRect(700, 75, 90, 30))
        self.model_load_button.setText("Load Model")
        self.model_load_button.clicked.connect(self.load_model)

    def make_manual_model(self):
        features = self.manual_model_feature_box.currentText()
        classifier = self.manual_model_classifier_box.currentText()
        data_filename = self.manual_data_filename_input.text()
        self.basewindow.model = EMGClassifier("manual", [features, classifier], data_filename)
        # POPUP saying the classifier is trained

    def make_automatic_model(self):
        selection_metric = self.select_model_metric_box.currentText()
        data_filename = self.select_data_filename_input.text()
        self.basewindow.model = EMGClassifier("selection", [selection_metric], data_filename)
        # popup saying the classifier was trained

    def save_model(self):
        if hasattr(self.basewindow.model,"run"):
            filename = self.model_filename.text()
            self.basewindow.model.save(filename)
        else:
            # popup saying it failed  
            
            msg = QMessageBox()
            msg.setWindowTitle("Classifier Saving")
            msg.setText("Classifier save failed")
            msg.setIcon(QMessageBox.Critical)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_() # shows the message box

    def load_model(self):
        filename = self.model_filename.text()
        if os.path.isfile(filename):
            self.basewindow.model = EMGClassifier()
            self.basewindow.model.load(filename)
        else:
            # popup saying it failed  
            
            msg = QMessageBox()
            msg.setWindowTitle("Classifier Loading")
            msg.setText("Classifier load failed")
            msg.setIcon(QMessageBox.Critical)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_() # shows the message box