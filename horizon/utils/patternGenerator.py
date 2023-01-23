class PatternGenerator():
    def __init__(self, ns):
        self.n_nodes = ns

    def generatePattern(self, gait_matrix, contacts, cycle_nodes, duty_cycle=1., opts=None):
        stance_nodes = dict()
        swing_nodes = dict()

        # prepare stance and swing nodes
        for contact in contacts:
            stance_nodes[contact] = []
            swing_nodes[contact] = []

        # compute nodes depending on gait type
        phase_nodes = int(cycle_nodes / gait_matrix.shape[1])
        flight_duration = int(phase_nodes * duty_cycle)

        for n_contact in range(gait_matrix.shape[0]):
            j = 0
            for n_cycle in range(gait_matrix.shape[1]):
                swing_chunk = [n + j for n in range(phase_nodes) if n < flight_duration]
                swing_nodes[contacts[n_contact]].extend(swing_chunk if gait_matrix[n_contact, n_cycle] else [])
                j += phase_nodes

        for name, value in stance_nodes.items():
            stance_nodes[name].extend([n for n in range(cycle_nodes) if n not in swing_nodes[name]])

        # repeat patter through nodes
        n_cycles = int(self.n_nodes / cycle_nodes) + 1

        stance_nodes_rep = dict()
        swing_nodes_rep = dict()
        for contact in contacts:
            stance_nodes_rep[contact] = []
            swing_nodes_rep[contact] = []

        for n in range(0, n_cycles):
            for name, value in swing_nodes.items():
                swing_nodes_rep[name].extend([elem + n * cycle_nodes for elem in value])

        for n in range(0, n_cycles):
            for name, value in stance_nodes.items():
                stance_nodes_rep[name].extend([elem + n * cycle_nodes for elem in value])

        for name, value in swing_nodes_rep.items():
            swing_nodes_rep[name] = [x for x in value if x < self.n_nodes]

        for name, value in stance_nodes_rep.items():
            stance_nodes_rep[name] = [x for x in value if x < self.n_nodes]

        return stance_nodes_rep, swing_nodes_rep



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