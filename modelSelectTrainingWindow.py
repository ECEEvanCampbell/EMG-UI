from utils import PageWindow
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QComboBox, QMessageBox, QLabel, QLineEdit
from pr_utils import EMGClassifier
import os

class SelectModelTrainingWindow(PageWindow):
    def __init__(self, basewindow):
        super().__init__()

        self.basewindow = basewindow 

        self.setObjectName("selectmodeltraining_window")
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Automatic Model Training")
        self.UiComponents()


    def UiComponents(self):
        ## CENTRAL WIDGET - this is a parent widget for formatting
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.setCentralWidget(self.centralwidget)

        ## MODEL SELECTION STUFF
        self.select_data_filename_label = QLabel(self.centralwidget)
        self.select_data_filename_label.setGeometry(QtCore.QRect(5,40,190,30))
        self.select_data_filename_label.setText("Data for Training")


        self.select_data_filename_input = QLineEdit(self.centralwidget)
        self.select_data_filename_input.setGeometry(QtCore.QRect(5,75,190,30))
        self.select_data_filename_input.setText("SGT_SXX.csv")


        self.select_window_size_label = QLabel(self.centralwidget)
        self.select_window_size_label.setGeometry(QtCore.QRect(5,110,190,30))
        self.select_window_size_label.setText("Window Size (ms)")

        self.select_window_size_input = QLineEdit(self.centralwidget)
        self.select_window_size_input.setGeometry(QtCore.QRect(5,145, 190, 30))
        self.select_window_size_input.setText("0.150")

        self.select_window_inc_label = QLabel(self.centralwidget)
        self.select_window_inc_label.setGeometry(QtCore.QRect(5,180,190,30))
        self.select_window_inc_label.setText("Window Increment (ms)")

        self.select_window_inc_input = QLineEdit(self.centralwidget)
        self.select_window_inc_input.setGeometry(QtCore.QRect(5,215,190,30))
        self.select_window_inc_input.setText("0.050")


        self.select_data_filename_input = QLineEdit(self.centralwidget)
        self.select_data_filename_input.setGeometry(QtCore.QRect(5,250,190,30))
        self.select_data_filename_input.setText("SGT_SXX.csv")

        self.select_model_metric_label = QLabel(self.centralwidget)
        self.select_model_metric_label.setGeometry(QtCore.QRect(5, 285, 190,30))
        self.select_model_metric_label.setText("Selection Metric")

        self.select_model_metric_box = QComboBox(self.centralwidget)
        self.select_model_metric_box.setGeometry(QtCore.QRect(5, 320, 190,30))
        self.select_model_metric_box.addItems(["accuracy","activeaccuracy", "MSA", "FE"])

        self.select_button = QtWidgets.QPushButton(self.centralwidget)
        self.select_button.setGeometry(QtCore.QRect(5,355, 190, 30))
        self.select_button.setText("Make Model")
        self.select_button.clicked.connect(self.make_automatic_model)


       

    def make_automatic_model(self):
        window_size = self.select_window_size_label.text()
        window_increment = self.select_window_inc_input.text()
        frequency = 1259 # TODO: dont't just hardcode this for delsys

        window_parameters = [window_size, window_increment, frequency]
        selection_metric = self.select_model_metric_box.currentText()
        data_filename = self.select_data_filename_input.text()
        self.basewindow.model = EMGClassifier("selection", [selection_metric], data_filename, window_parameters)

        msg = QMessageBox()
        msg.setWindowTitle("Classifier Trained")
        msg.setText("Classifier is trained")
        msg.setIcon(QMessageBox.Critical)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_() # shows the message box