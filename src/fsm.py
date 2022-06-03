from enum import Enum
from entities import Robot, Goal, Ball, Frame, GroundPoint
import communication_proto

class State(Enum):
    unkown = 0
    stop = 1
    search = 2
    drive = 3
    dock = 4
    rotate = 5

class FSM():
    def __init__(
                self,
                initial_state = 1,
                init_time = 0
                ):
        super(FSM, self).__init__()
        self.current_state = State(initial_state)
        self.last_state = State(initial_state)
        self.state_init_time = init_time
        self.state_time = 0

    def updateState1(self):
        final_state = self.current_state
        if self.current_state == State.stop:
            if self.condition1:
                final_state = State.search

        elif self.current_state == State.search:
            if self.condition1:
                final_state = State.drive

        elif self.current_state == State.drive:
            if not self.condition1:
                if self.condition2:
                    final_state = State.dock
                else:
                    final_state = State.search

        elif self.current_state == State.dock:
            if self.condition1:
                final_state = State.drive
        
        no_trasition = True
        if final_state != self.current_state:
            no_trasition = False
            self.last_state = self.current_state
            self.current_state = final_state

        return no_trasition
        