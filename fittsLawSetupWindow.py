from utils import PageWindow
from PyQt5 import QtCore, QtWidgets
from fittsLawWindow import Game

class FittsLawSetupWindow(PageWindow):
    def __init__(self, basewindow):
        super().__init__()
        self.basewindow = basewindow
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Fitts Law Setup")
        self.UiComponents()


    def UiComponents(self):

        ## CENTRAL WIDGET - this is a parent widget for formatting
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.setCentralWidget(self.centralwidget)


    def onRender(self):
        # display whether there is a connected device or not
        self.device_label = QtWidgets.QLabel(self.centralwidget)
        self.device_label.setGeometry(QtCore.QRect(5,10,300,30))
        if  ("name" in self.basewindow.device):
            self.device_label.setText(self.basewindow.device['name'] +' is connected.')
        else:
            self.device_label.setText("No device connected.")

        # display whether there is a trained classifier or not
        self.classifier_label = QtWidgets.QLabel(self.centralwidget)
        self.classifier_label.setGeometry(QtCore.QRect(5,50, 300, 30))
        if (hasattr(self.basewindow.model, "classifier")):
            self.classifier_label.setText('Model is prepared')
        else:
            self.classifier_label.setText('No model prepared.')

        # if the model has been trained, we need to map all trained classes to directions
        current_height = 5
        pad_height   = 15
        row_height   = 30
        self.class_labels = []
        if (hasattr(self.basewindow.model, "classifier")):
            # for every class that has been trained
            for class_ in self.basewindow.model.classifier.classes_:
                self.class_labels.append([QtWidgets.QLabel(self.centralwidget), QtWidgets.QComboBox(self.centralwidget)])
                # label identifier
                self.class_labels[-1][0].setGeometry(QtCore.QRect(400, current_height, 145, row_height))
                self.class_labels[-1][0].setText(str(class_))
                # combobox for that label
                self.class_labels[-1][1].setGeometry(QtCore.QRect(555, current_height, 145, row_height))
                self.class_labels[-1][1].addItems(["Left", "Right","Up","Down", "NM"])
                current_height += pad_height + row_height

        # actual fitts law settings
        self.num_circle_label = QtWidgets.QLabel(self.centralwidget)
        self.num_circle_label.setGeometry(QtCore.QRect(800, 5, 300, 30))
        self.num_circle_label.setText("Number of Circles")

        self.num_circle_input = QtWidgets.QLineEdit(self.centralwidget)
        self.num_circle_input.setGeometry(QtCore.QRect(800, 50, 300, 30))
        self.num_circle_input.setText("8")

        # TODO: add more settings like speed, maybe controller type, etc.

        # TODO: add filename for trajectory info
        # add begin button
        self.begin_button = QtWidgets.QPushButton(self.centralwidget)
        self.begin_button.setGeometry(QtCore.QRect(800, 500, 100, 30))
        self.begin_button.setText("begin")
        self.begin_button.clicked.connect(self.spawn_fitts_law)




    def spawn_fitts_law(self):

        # check that device is connected
        if not "name" in self.basewindow.device:
            print("no sensor connected, error")
            return
        # check that classifier is trained
        if not hasattr(self.basewindow.model, "classifier"):
            print("no classifier trained")
            return
        # get all the relevant settings
        
        class_mappings = {}
        for i in range(len(self.class_labels)):
            class_mappings[self.class_labels[i][0].text()] = self.class_labels[i][1].currentText()

        num_circles = int(self.num_circle_input.text())

        device = self.basewindow.device
        model = self.basewindow.model

        game = Game( num_circles, device, model, class_mappings)
        game.run()
        



                
