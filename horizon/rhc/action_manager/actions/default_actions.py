from dataclasses import dataclass, field

@dataclass
class Action:
    name: str


    @classmethod
    def from_dict(cls, task_dict):
        return cls(**task_dict)


    def __post_init__(self):
        # todo: this is for simplicity
        self.indices = np.array(self.indices) if self.indices is not None else None
        # self.nodes = list(range(self.prb.getNNodes()))

    def _reset(self):
        pass
    
    def getName(self):
        return self.name



class Step:
    def __init__(self, swing_frame):
        cycle_list = [True if contact_name != swing_contact else False for contact_name in self.contact_phases.keys()]
def step(self, swing_contact):
    cycle_list = [True if contact_name != swing_contact else False for contact_name in self.contact_phases.keys()]
    self.cycle(cycle_list)

def trot(self):
    cycle_list_1 = [0, 1, 1, 0]
    cycle_list_2 = [1, 0, 0, 1]
    self.cycle(cycle_list_1)
    self.cycle(cycle_list_2)


def crawl(self):
    cycle_list_1 = [0, 1, 1, 1]
    cycle_list_2 = [1, 1, 1, 0]
    cycle_list_3 = [1, 0, 1, 1]
    cycle_list_4 = [1, 1, 0, 1]
    self.cycle(cycle_list_1)
    self.cycle(cycle_list_2)
    self.cycle(cycle_list_3)
    self.cycle(cycle_list_4)


def leap(self):
    cycle_list_1 = [0, 0, 1, 1]
    cycle_list_2 = [1, 1, 0, 0]
    self.cycle(cycle_list_1)
    self.cycle(cycle_list_2)


def walk(self):
    cycle_list_1 = [1, 0, 1, 0]
    cycle_list_2 = [0, 1, 0, 1]
    self.cycle(cycle_list_1)
    self.cycle(cycle_list_2)


def jump(self):
    cycle_list = [0, 0, 0, 0]
    self.cycle(cycle_list)


def wheelie(self):
    cycle_list = [0, 0, 1, 1]
    self.cycle(cycle_list)


def stand(self):
    cycle_list = [1, 1, 1, 1]
    self.cycle(cycle_list)
