import pygame
import math
import time
from time import sleep
import random
import sys
import pickle
import os
from biomedical_hardware_readers import DelsysTrignoReader
import datetime
import numpy as np
from PyQt5.QtCore import QThread, QObject, pyqtSignal

from screenGuidedTrainingWindow import CollectionWorker

class DataWorker(QObject):
    finished = pyqtSignal()
    direction_ready = pyqtSignal()
    # emg = None
    # imu = None
    def __init__(self, fl):
        super(DataWorker, self).__init__()
        self.fl = fl
        self.fl.game.device['reader'].start() # start streaming loop
        self.fl.game.device['reader'].wait_for_reading_loop()
        sleep(2)

    def stream(self):
        while self.fl.trial < self.fl.max_trial:
            tic = time.perf_counter()
            emg = self.fl.game.device['emg_buf'].read_matrix()
            #imu = self.fl.game.device['aux_buf'].read_matrix()
            emg = emg[:,2:-2]
            emg = emg[:,self.fl.game.model.active_channels]
            prob = self.fl.game.model.run(emg.transpose())
            # in the event something weird happens, use the last window
            # alternatively this could be the average prob
            # eventually this logic should be build into a 'controller' class
            prob = prob[-1,:]
            #direction = [xvel,yvel]
            direction = [0,0]

            # code for regression type controllers
            # for c in range(len(prob)):
            #     class_id = self.fl.game.model.classifier.classes_[c]
            #     class_direction = self.fl.game.class_mappings[str(class_id)]
            #     if class_direction == "Left":
            #         direction[0] += -1 * self.fl.VEL * prob[c]
            #     elif class_direction == "Right":
            #         direction[0] += self.fl.VEL*prob[c]
            #     elif class_direction == "Up":
            #         direction[1] += -1* self.fl.VEL *prob[c]
            #     elif class_direction == "Down":
            #         direction[1] +=  self.fl.VEL * prob[c]
            #     # there is also a no movement case (NM)

            # code for classification based controllers
            decision = np.argmax(prob)
            class_direction = self.fl.game.class_mappings[str(decision)]
            if class_direction == "Left":
                direction[0] += -1 * self.fl.VEL
            elif class_direction == "Right":
                direction[0] += self.fl.VEL
            elif class_direction == "Up":
                direction[1] += -1* self.fl.VEL
            elif class_direction == "Down":
                direction[1] +=  self.fl.VEL
            # there is also a no movement case (NM)

            self.fl.current_direction = direction
            # classifier stuff goes here
            if self.fl.LOGGING:
                self.fl.log(emg, prob, direction)
            toc = time.perf_counter()
            duration = toc - tic
            self.direction_ready.emit()
            sleep((self.fl.window_increment/1000 - duration))
        
        self.finished.emit()


class FittsLawTest:
    def __init__(self, game):
        self.game = game
        self.font = pygame.font.SysFont('helvetica', 40)

        # gameplay parameters
        self.BLACK = (0,0,0)
        self.RED   = (255,0,0)
        self.YELLOW = (255,255,0)
        self.BLUE   = (0,102,204)
        self.small_rad = 40
        self.big_rad   = 275
        self.pos_factor1 = self.big_rad/2
        self.pos_factor2 = (self.big_rad * math.sqrt(3))//2

        self.VEL = 1.5
        self.dwell_time = 3
        self.num_of_circles = self.game.num_circles # this value will be a user input


        # interface objects
        self.circles = []
        self.cursor  = self.cursor = pygame.Rect(self.game.width//2 - 7, self.game.height//2 - 7, 14, 14)
        self.goal_circle = -1
        self.get_new_goal_circle()

        self.current_direction = [0,0]
        self.window_increment = 50 # in ms

        ## Track if connected to Delsys
        self.sensor_connected = True
        
        ## Save or don't save
        self.LOGGING = True
        self.trial = 0
        self.max_trial = self.game.num_trials

        self.main_clock = time.perf_counter()

    

    def collection_worker(self):
        self.thread = QThread() # define a thread to stream data on
        self.worker = DataWorker(self) 
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.stream)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start() # start the thread

    def draw(self):
        self.game.screen.fill(self.BLACK)
        self.draw_circles()
        self.draw_cursor()
        self.draw_timer()
    
    def draw_circles(self):
        if not len(self.circles):
            self.angle = 0
            self.angle_increment = 360 // self.num_of_circles
            while self.angle < 360:
                self.circles.append(pygame.Rect((self.game.width//2 - self.small_rad) + math.cos(math.radians(self.angle)) * self.big_rad, (self.game.height//2 - self.small_rad) + math.sin(math.radians(self.angle)) * self.big_rad, self.small_rad * 2, self.small_rad * 2))
                self.angle += self.angle_increment

        for circle in self.circles:
            pygame.draw.circle(self.game.screen, self.RED, (circle.x + self.small_rad, circle.y + self.small_rad), self.small_rad, 2)
        
        
        goal_circle = self.circles[self.goal_circle]
        pygame.draw.circle(self.game.screen, self.RED, (goal_circle.x + self.small_rad, goal_circle.y + self.small_rad), self.small_rad)
            
    
    def draw_cursor(self):
        pygame.draw.circle(self.game.screen, self.YELLOW, (self.cursor.x + 7, self.cursor.y + 7), 7)

    def draw_timer(self):
        if hasattr(self, 'dwell_timer'):
            if self.dwell_timer is not None:
                toc = time.perf_counter()
                duration = round((toc-self.dwell_timer),2)
                time_str = str(duration)
                draw_text = self.font.render(time_str, 1, self.BLUE)
                self.game.screen.blit(draw_text, (10, 10))

    def run(self):
        if self.sensor_connected:
            self.print_caption()
            self.draw()
            self.run_game_process()
            self.move()
    
    def run_game_process(self):
        self.check_collisions()
        self.check_events()
        # comment this if not using keyboard
        #self.check_keys()

    def check_collisions(self):
        circle = self.circles[self.goal_circle]
        if math.sqrt((circle.centerx - self.cursor.centerx)**2 + (circle.centery - self.cursor.centery)**2) < 47:
            pygame.event.post(pygame.event.Event(pygame.USEREVENT + self.goal_circle))
            self.Event_Flag = True
        else:
            pygame.event.post(pygame.event.Event(pygame.USEREVENT + self.num_of_circles))
            self.Event_Flag = False

    def check_events(self):
        # closing window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if self.LOGGING:
                    self.save_log()
                pygame.quit()
                #sys.exit()

            ## CHECKING FOR COLLISION BETWEEN CURSOR AND RECTANGLES
            if event.type >= pygame.USEREVENT and event.type < pygame.USEREVENT + self.num_of_circles:
                if self.dwell_timer is None:
                    self.dwell_timer = time.perf_counter()
                else:
                    toc = time.perf_counter()
                    self.duration = round((toc - self.dwell_timer), 2)
                if self.duration >= self.dwell_time:
                    self.get_new_goal_circle()
                    self.dwell_timer = None
                    if self.trial < self.max_trial-1: # -1 because max_trial is 1 indexed
                        self.trial += 1
                    else:
                        if self.LOGGING:
                            self.save_log()
                        pygame.quit()
                        #sys.exit()
                        if self.game.device['name'] == "Delsys":
                            self.game.device['reader'].shutdown()
                            self.game.device['reader'].join()
                            self.game.device['emg_buf'].close()
                            self.game.device['aux_buf'].close()


            elif event.type == pygame.USEREVENT + self.num_of_circles:
                if self.Event_Flag == False:
                    self.dwell_timer = None
                    self.duration = 0

           
    def move(self):
         #               HO
            ##           ^
            #            l 
            #            l
            # WF < ---   x   --- > WE
            #            l
            #            l 
            #            v 
            #           PoG
        if self.cursor.x + self.current_direction[0] > 0 and self.cursor.x + self.current_direction[0] < self.game.width:
            self.cursor.x += self.current_direction[0]
        if self.cursor.y + self.current_direction[1] > 0 and self.cursor.y + self.current_direction[1] < self.game.height:
            self.cursor.y += self.current_direction[1]
    
    
    def get_new_goal_circle(self):
        # # base case: no checking for same goal circle
        # if self.goal_circle == -1:
        #     self.goal_circle = random.randint(0, self.num_of_circles - 1)
        # else:
        #     # check for getting the same goal
        #     new_goal_circle = random.randint(0, self.num_of_circles - 1)
        #     while new_goal_circle == self.goal_circle:
        #         new_goal_circle = random.randint(0, self.num_of_circles - 1)
        #     self.goal_circle = new_goal_circle
        if self.goal_circle == -1:
            self.goal_circle = 0
            self.next_circle_in = self.num_of_circles//2
            self.circle_jump = 0
        else:
            self.goal_circle =  (self.goal_circle + self.next_circle_in )% self.num_of_circles
            if self.circle_jump == 0:
                self.next_circle_in = self.num_of_circles//2 + 1
                self.circle_jump = 1
            else:
                self.next_circle_in = self.num_of_circles // 2
                self.circle_jump = 0
            


    def print_caption(self):
        pygame.display.set_caption(str(self.game.clock.get_fps()))

    def log(self,emg, prob, direction):
        # [(trial_number) (goal_circle) (global_clock) (cursor_position) <EMG> <current direction> ]
        if not hasattr(self, 'log_dictionary'):
            self.log_dictionary = {
                'trial_number':      [],
                'goal_circle' :      [],
                'global_clock' :     [],
                'cursor_position':   [],
                'EMG':               [],
                'current_prob':      [],
                'current_direction': []
            }
        # add time since start
        
        toc = time.perf_counter()
        self.log_dictionary['trial_number'].append(self.trial)
        self.log_dictionary['goal_circle'].append(self.goal_circle)
        self.log_dictionary['global_clock'].append(time.perf_counter())
        self.log_dictionary['cursor_position'].append((self.cursor.x, self.cursor.y))
        self.log_dictionary['EMG'].append(emg) 
        self.log_dictionary['current_prob'].append(prob)
        self.log_dictionary['current_direction'].append(direction)

    def save_log(self):
        if not os.path.exists('results'):
            os.mkdir('results')
        with open('results/'+self.game.savefile, 'wb') as f:
            pickle.dump(self.log_dictionary, f)

    def closeEvent(self, event):
        self.game.device['reader'].shutdown()
        self.game.device['reader'].join()
        self.game.device['emg_buf'].close()
        self.game.device['aux_buf'].close()
        self.done = True
        event.accept()
        



class Game:
    # just for interfacing with pygame
    def __init__(self, num_circles, device, model, class_mappings, num_trials=15, savefile="tmp.pkl", fps=60, width=1250, height=750):
        pygame.init()

        self.num_circles = num_circles
        self.device = device
        self.model = model
        self.class_mappings = class_mappings
        self.savefile = savefile
        self.num_trials = num_trials

        self.fps = fps
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode([width, height])
        self.clock = pygame.time.Clock()
        self.fitts_law = FittsLawTest(self)

    def run(self):
        self.fitts_law.collection_worker() # Make thread to get data on
        #collection_worker = DataWorker(self.fitts_law)
        self.fitts_law.done = False
        
        while not self.fitts_law.done:
            # updated frequently for graphics & gameplay
            self.fitts_law.run()
            pygame.display.update()
            self.clock.tick(self.fps)
            # remove this line when collection worker has been put back
            #collection_worker.stream()


if __name__ == "__main__":
    fps    = 60
    width  = 1250
    height = 750
    game = Game(fps, width, height)
    
    game.run()