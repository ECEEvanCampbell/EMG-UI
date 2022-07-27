from utils import PageWindow


# Need to do this on a thread

class VisualizationWindow(PageWindow):
    def __init__(self, basewindow):
        super().__init__()
        # Lets store the handle to the settings window as a 
        self.basewindow = basewindow
