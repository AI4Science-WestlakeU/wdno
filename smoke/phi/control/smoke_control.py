from phi.control.sequences import *



class SmokeState(Frame):

    def __init__(self, index, density, velocity, type=TYPE_KEYFRAME):
        Frame.__init__(self, index, type=type)
        assert density is not None and velocity is not None
        self.density = density
        self.velocity = velocity


class UpdatableSmokeState(Frame):

    def __init__(self, ground_truth):
        Frame.__init__(self, ground_truth.index)
        self.ground_truth = ground_truth
        self.states = []

    def update(self, new_state, type):
        assert type >= self.type
        self.states.append(new_state)
        self.type = type