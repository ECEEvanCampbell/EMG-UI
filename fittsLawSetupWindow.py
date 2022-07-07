from utils import PageWindow
from PyQt5 import QtCore, QtWidgets

class FittsLawSetupWindow(PageWindow):
    def __init__(self, basewindow):
        super().__init__()
        self.basewindow = basewindow
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Fitts Law Setup")
        self.UiComponents()

    def spawnFittsLaw(self):
        # TODO: very important, actually spawn the pygame interface and give link to this interface
        pass

    def UiComponents(self):
        self.backButton = QtWidgets.QPushButton("Fitts Law Setup", self)
        self.backButton.setGeometry(QtCore.QRect(5, 5, 100, 20))
        self.backButton.clicked.connect(self.spawnFittsLaw)
