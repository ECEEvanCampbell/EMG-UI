from utils import PageWindow
from PyQt5 import QtCore, QtWidgets

class ModelTrainingWindow(PageWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("ModelTraining")
        self.UiComponents()

    def goToMain(self):
        self.goto("main")

    def UiComponents(self):
        self.backButton = QtWidgets.QPushButton("BackButton", self)
        self.backButton.setGeometry(QtCore.QRect(5, 5, 100, 20))
        self.backButton.clicked.connect(self.goToMain)
