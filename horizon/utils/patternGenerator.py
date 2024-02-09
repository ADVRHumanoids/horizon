import warnings

class PatternGenerator():
    def __init__(self, ns, contacts):
        self.n_nodes = ns
        self.contacts = contacts

    def generateCycle(self, phi, cycle_nodes, duty_cycle):

        # phi: phase shift between each cycle of the gait
        # duty cycle: stance_phase/total_cycle_time

        if isinstance(duty_cycle, list):
            raise NotImplementedError("Independent duty cycle still to implement")


        stance_nodes = dict()
        swing_nodes = dict()

        # prepare stance and swing nodes
        for contact in self.contacts:
            stance_nodes[contact] = []
            swing_nodes[contact] = []

        i_contact = 0
        for contact in self.contacts:
            # the first node of the phase depends upon phi



            phase_start_node = int(cycle_nodes * phi[i_contact])
            stance_duration = int(cycle_nodes * duty_cycle)

            if stance_duration % duty_cycle != 0:
                warnings.warn(f'remainder detected for contact {contact}. Check twice.')

            # fill stance nodes. If it reaches the last node, it starts from the beginning
            # list(range(phase_start_node, phase_start_node + stance_duration))
            phase_end_node = phase_start_node + stance_duration
            if phase_end_node <= cycle_nodes:
                stance_nodes[contact] = list(range(phase_start_node, phase_start_node + stance_duration))
            else:
                stance_nodes[contact] = list(range(phase_start_node, cycle_nodes))
                stance_nodes[contact] = list(range(0, phase_end_node - cycle_nodes)) + stance_nodes[contact]

            swing_nodes[contact] = [n for n in range(cycle_nodes) if n not in stance_nodes[contact]]

            i_contact += 1

        return stance_nodes, swing_nodes

    def generateGait(self, gait_matrix, cycle_nodes):
        stance_nodes = dict()
        swing_nodes = dict()

        new_cycle_nodes = cycle_nodes - cycle_nodes % gait_matrix.shape[1]

        # prepare stance and swing nodes
        for contact in self.contacts:
            stance_nodes[contact] = []
            swing_nodes[contact] = []

        # compute nodes depending on gait type
        phase_nodes = int(cycle_nodes / gait_matrix.shape[1])

        for n_contact in range(gait_matrix.shape[0]):
            j = 0
            for n_cycle in range(gait_matrix.shape[1]):
                swing_chunk = [n + j for n in range(phase_nodes)]
                swing_nodes[self.contacts[n_contact]].extend(swing_chunk if gait_matrix[n_contact, n_cycle] else [])
                j += phase_nodes

        for name, value in stance_nodes.items():
            stance_nodes[name].extend([n for n in range(new_cycle_nodes) if n not in swing_nodes[name]])

        return stance_nodes, swing_nodes, new_cycle_nodes

    def generateCycle_old(self, gait_matrix, cycle_nodes, duty_cycle=1.):
        stance_nodes = dict()
        swing_nodes = dict()

        new_cycle_nodes = cycle_nodes - cycle_nodes % gait_matrix.shape[1]

        # prepare stance and swing nodes
        for contact in self.contacts:
            stance_nodes[contact] = []
            swing_nodes[contact] = []

        # compute nodes depending on gait type
        phase_nodes = int(cycle_nodes / gait_matrix.shape[1])
        flight_duration = int(phase_nodes * duty_cycle)

        for n_contact in range(gait_matrix.shape[0]):
            j = 0
            for n_cycle in range(gait_matrix.shape[1]):
                swing_chunk = [n + j for n in range(phase_nodes) if n < flight_duration]
                swing_nodes[self.contacts[n_contact]].extend(swing_chunk if gait_matrix[n_contact, n_cycle] else [])
                j += phase_nodes

        for name, value in stance_nodes.items():
            stance_nodes[name].extend([n for n in range(new_cycle_nodes) if n not in swing_nodes[name]])

        return stance_nodes, swing_nodes, new_cycle_nodes


    def generatePattern(self, gait_matrix, cycle_nodes, duty_cycle=1., opts=None):

        stance_nodes, swing_nodes, new_cycle_nodes = self.generateCycle_old(gait_matrix, cycle_nodes, duty_cycle)

        # repeat patter through nodes
        n_cycles = int(self.n_nodes / cycle_nodes) + 1

        # prepare the pattern containers
        stance_nodes_rep = dict()
        swing_nodes_rep = dict()
        for contact in self.contacts:
            stance_nodes_rep[contact] = []
            swing_nodes_rep[contact] = []

        # repeat pattern for n cycles
        for n in range(0, n_cycles):
            for name, value in swing_nodes.items():
                swing_nodes_rep[name].extend([elem + n * cycle_nodes for elem in value])

        for n in range(0, n_cycles):
            for name, value in stance_nodes.items():
                stance_nodes_rep[name].extend([elem + n * cycle_nodes for elem in value])

        # remove nodes if they are outside range
        for name, value in swing_nodes_rep.items():
            swing_nodes_rep[name] = [x for x in value if x < self.n_nodes]

        for name, value in stance_nodes_rep.items():
            stance_nodes_rep[name] = [x for x in value if x < self.n_nodes]

        return stance_nodes_rep, swing_nodes_rep

    def visualizer(self, cycle_nodes, stance_nodes: dict, swing_nodes: dict):
        import matplotlib.pyplot as plt

        columns = [str(elem) for elem in range(cycle_nodes)]
        rows = self.contacts.copy()

        stance_color = "g"
        swing_color = "w"

        stance_text = " "
        swing_text = " "

        table_text = []
        colors_text = []
        for contact in self.contacts:

            row_list = [None] * cycle_nodes
            color_list = [None] * cycle_nodes


            # fill table for each node depending on phase state (swing/stance)
            for n in range(cycle_nodes):
                flag_filled = False
                if n in stance_nodes[contact]:
                    row_list[n] = stance_text
                    color_list[n] = stance_color
                    flag_filled = True

                if n in swing_nodes[contact]:
                    if not flag_filled:
                        row_list[n] = swing_text
                        color_list[n] = swing_color
                    else:
                        raise Exception("Error: element both in stance and swing.")

            table_text.append(row_list)
            colors_text.append(color_list)


        for row in range(len(table_text)):
            table_text[row] = ['empty' if v is None else v for v in table_text[row]]


        # Add a table at the bottom of the axes
        fig, ax = plt.subplots()
        ax.axis('tight')
        ax.axis('off')
        tab = ax.table(cellText=table_text, cellColours=colors_text,
                 colLabels=columns, rowLabels=rows, loc='center')

        tab.set_fontsize(40)

        plt.show()



if __name__ == '__main__':

    import numpy as np
    pg = PatternGenerator(50, ['a', 'b', 'c', 'd'])

    # =============================================================================================
    # =============================================================================================
    cycle_nodes = 40
    gait_matrix = np.array([[0, 0, 0, 1],
                            [0, 1, 0, 0],
                            [1, 0, 0, 0],
                            [0, 0, 1, 0]]).astype(int)

    stance_old, swing_old, new_cycle_nodes = pg.generateGait(gait_matrix, cycle_nodes)

    print('stance')
    for name, elem in stance_old.items():
        print(name, elem)
    print('swing')
    for name, elem in swing_old.items():
        print(name, elem)

    pg.visualizer(cycle_nodes, stance_old, swing_old)
    exit()
    # =============================================================================================
    # =============================================================================================

    cycle_nodes = 10
    phi = [0., 0.25, 0.5, 0.75]
    duty_cycle = 0.75

    #  gait_matrix, contacts, cycle_nodes, duty_cycle=1., opts=None):
    stance, swing = pg.generateCycle(phi, cycle_nodes, duty_cycle)

    print('stance')
    for name, elem in stance.items():
        print(name, elem)
    print('swing')
    for name, elem in swing.items():
        print(name, elem)

    pg.visualizer(cycle_nodes, stance, swing)
# class PatternGenerator:
#     def __init__(self, pattern, duty_cycle, t_init, stride_time, step_n_min=3):
#
#         # t_start = t_init
#         # t_goal = t_start + stride_time * (1 - duty_cycle)
#         # s = Step(c, k_start, k_end)
#         # steps.append(s)

# steps = list()
# n_steps = 8
# pattern = [0, 3, 1, 2]
# stride_time = 6.0
# duty_cycle = 0.0
# tinit = 0.
# dt = 0.1
# nc = 4
# for k in range(2):
#     for i in range(n_steps):
#         l = pattern[i % nc]
#         t_start = tinit + i * stride_time / nc
#         t_goal = t_start + stride_time * (1 - duty_cycle)
#         s = dict(leg=l, k_start=k + int(t_start / dt), k_goal=k + int(t_goal / dt))
#         steps.append(s)
#
# for step in steps:
#     print(step)