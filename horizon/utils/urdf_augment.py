import urdf_parser_py.urdf as urdfpy
import rospkg
from networkx.drawing.nx_agraph import graphviz_layout
import pygraphviz as pgv
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


class URDFAugment:
    def __init__(self, robot):
        self.robot = robot

    @classmethod
    def load_from_file(cls, urdf_path):
        robot = urdfpy.URDF.from_xml_file(urdf_path)
        return cls(robot)

    @classmethod
    def load_from_ros_param(cls, ros_param):
        robot = urdfpy.URDF.from_parameter_server(ros_param)
        return cls(robot)

    def __parse_urdf_to_graph(self):

        # Create a directed graph using NetworkX
        G = nx.DiGraph()

        for link in self.robot.links:
            G.add_node(link.name)

        # Add edges for each joint between the parent and child link, including joint names as edge labels
        edge_labels = {}
        for joint in self.robot.joints:
            parent_link = joint.parent
            child_link = joint.child
            G.add_edge(parent_link, child_link)
            edge_labels[(parent_link, child_link)] = joint.name  # Store the joint name as an edge label

        # Use the graphviz_layout for a hierarchical layout
        # pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Gnodesep=1.5 -Granksep=1.5 -Gmargin=0.2')
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')

        # Draw the graph
        plt.figure(figsize=(20, 20))
        nx.draw(G,
                pos,
                # bbox=dict(boxstyle='square'), #, ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)),
                with_labels=True,
                node_size=3000,
                node_shape='h',
                node_color='skyblue',
                font_size=7,
                font_weight='bold',
                edge_color='gray')

        # Draw the joint names on the edges, ensuring the labels are horizontal
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, rotate=False)

        # Show the plot
        plt.show()

    def graph(self):
        self.__parse_urdf_to_graph()


    def __get_elem(self, container, elem_name):

        for elem in container:
            if elem.name == elem_name:
                return elem
        return None

    def getJoint(self, joint_name):

        return self.__get_elem(self.robot.joints, joint_name)

    def getLink(self, link_name):

        return self.__get_elem(self.robot.links, link_name)

    def getLinkNames(self):

        return [link.name for link in self.robot.links]

    def getJointNames(self):

        return [joint.name for joint in self.robot.joints]

    def addReferenceFrame(self, link_name, parent_link_name, origin_xyz):


        if self.getLink(parent_link_name) is None:
            print(f"Parent link '{parent_link_name}' not found.")
            return False


        # Create a new link
        new_link = urdfpy.Link(name=link_name)
        self.robot.add_link(new_link)

        fixed_joint_name = f"{parent_link_name}_to_{link_name}"

        new_joint = urdfpy.Joint(name=fixed_joint_name,
                                 parent=parent_link_name,
                                 child=link_name,
                                 joint_type='fixed',
                                 origin=urdfpy.Pose(xyz=origin_xyz, rpy=[0., 0., 0.])
        )
        # Add child and parent elements to the joint
        # Append the new link to the URDF

        self.robot.add_joint(new_joint)

        # print(f"Added link '{link_name}' with joint '{new_joint.name}' connecting to parent '{parent_link_name}'.")
        return True

    def addRectangleReferenceFrame(self, parent_link_name, size, offset_x = 0.0):

        self.addReferenceFrame(f'{parent_link_name}_lower_right_link', parent_link_name, origin_xyz=[-size[0] / 2 + offset_x, -size[1] / 2, 0])
        self.addReferenceFrame(f'{parent_link_name}_lower_left_link',parent_link_name, origin_xyz=[-size[0] / 2 + offset_x, size[1] / 2, 0])
        self.addReferenceFrame(f'{parent_link_name}_upper_right_link', parent_link_name, origin_xyz=[size[0] / 2 + offset_x, -size[1] / 2, 0])
        self.addReferenceFrame(f'{parent_link_name}_upper_left_link',parent_link_name, origin_xyz=[size[0] / 2 + offset_x, size[1] / 2, 0])



    def writeToFile(self, file_name):

        with open(file_name, 'w') as f:
            f.write(self.robot.to_xml())

        print(f"URDF written to '{file_name}'")

    def getXml(self):

        return self.robot.to_xml_string()


# Usage
if __name__ == '__main__':
    from urdf_parser_py.urdf import URDF, Link, Joint

    urdf_path = rospkg.RosPack().get_path('g1_description') + '/urdf/g1_23dof.urdf'
    urdf_aug = URDFAugment(urdf_path)


    urdf_aug.addReferenceFrame('l_sole', 'left_ankle_roll_link', origin_xyz=[0, 0, 0])
    urdf_aug.addReferenceFrame('r_sole', 'right_ankle_roll_link', origin_xyz=[0, 0, 0])

    sole_xy = [0.2, 0.1]

    urdf_aug.addRectangleReferenceFrame('l_sole', size=sole_xy)
    urdf_aug.addRectangleReferenceFrame('r_sole', size=sole_xy)


    print(urdf_aug.getXml())
    # urdf_aug.graph()