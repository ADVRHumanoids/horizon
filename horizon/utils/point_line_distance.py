import numpy as np

def point_close_to_edges(point, polygon_vertices, threshold):
    # Calculate the distances between the point and each edge of the rectangle

    distances = []
    for vertex_i in range(len(polygon_vertices)):

        distances.append(distance_point_to_line(point,
                                                [polygon_vertices[vertex_i % len(polygon_vertices)],
                                                            polygon_vertices[(vertex_i + 1) % len(polygon_vertices)]]))


    # Check if any distance is within the threshold
    for distance in distances:
        if distance < threshold:
            return True

    return False


def distance_point_to_line(point, line_vertices):
    # Calculate the distance between a point (x, y) and a line defined by two points (x1, y1) and (x2, y2)
    distance = np.linalg.norm(np.cross(line_vertices[1] - line_vertices[0], line_vertices[0] - point)) / np.linalg.norm(line_vertices[1] - line_vertices[0])

    return distance


# Example usage
point = np.array([1, 0.5])

vertices = []
vertices.append(np.array([-1, -1]))
vertices.append(np.array([-1, 1]))
vertices.append(np.array([1, 1]))
vertices.append(np.array([1, -1]))

threshold = 1  # Adjust this threshold as needed

if point_close_to_edges(point, vertices, threshold):
    print("The point is close to the edges of the rectangle.")
else:
    print("The point is not close to the edges of the rectangle.")