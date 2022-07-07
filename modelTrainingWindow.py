from utils import PageWindow
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QComboBox, QMessageBox

class ModelTrainingWindow(PageWindow):
    def __init__(self, basewindow):
        super().__init__()
        self.setObjectName("settings_window")
        self.resize(514, 466)
        self.massive_font = QtGui.QFont()
        self.massive_font.setPointSize(22)
        self.title_font = QtGui.QFont()
        self.title_font.setPointSize(16)
        self.subtitle_font = QtGui.QFont()
        self.subtitle_font.setPointSize(14)
        self.text_font  = QtGui.QFont()
        self.text_font.setPointSize(11)
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

        