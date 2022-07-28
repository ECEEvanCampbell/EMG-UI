from utils import PageWindow
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QComboBox, QMessageBox, QLabel, QLineEdit
from pr_utils import EMGClassifier
import os

class ManageModelTrainingWindow(PageWindow):
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
        
        ## Model SAVE/LOAD stuff
        self.model_label = QLabel(self.centralwidget)
        self.model_label.setText("Save/Load Model")
        self.model_label.setGeometry(QtCore.QRect(5,5,190,30))

        self.model_filename = QLineEdit(self.centralwidget)
        self.model_filename.setGeometry(QtCore.QRect(5, 40, 190, 30))
        self.model_filename.setText("MDL_SXX_AXX.pkl")

        self.model_save_button = QtWidgets.QPushButton(self.centralwidget)
        self.model_save_button.setGeometry(5, 75, 90, 30)
        self.model_save_button.setText("Save Model")
        self.model_save_button.clicked.connect(self.save_model)

        self.model_load_button = QtWidgets.QPushButton(self.centralwidget)
        self.model_load_button.setGeometry(QtCore.QRect(105, 75, 90, 30))
        self.model_load_button.setText("Load Model")
        self.model_load_button.clicked.connect(self.load_model)



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