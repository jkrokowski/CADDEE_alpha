import numpy as np
import matplotlib.pyplot as plt

def line_intersection(p1, p2, p3, p4):
    """Computes intersection point of line segments (p1,p2) and (p3,p4), if any."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if abs(denom) < 1e-10:
        return None  # Parallel lines

    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    point = np.array([px, py])

    # Check if intersection is within both segments
    if (min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2) and
        min(x3, x4) <= px <= max(x3, x4) and min(y3, y4) <= py <= max(y3, y4)):
        return point
    return None

def find_intersection(poly1, poly2):
    """Finds first intersection point and corresponding indices."""
    for i in range(len(poly1) - 1):
        p1, p2 = poly1[i], poly1[i + 1]
        for j in range(len(poly2) - 1):
            p3, p4 = poly2[j], poly2[j + 1]
            inter = line_intersection(p1, p2, p3, p4)
            if inter is not None:
                return (i, j, inter)
    return None

def join_polylines_at_intersection(poly1, poly2):
    """Trim both polylines to intersection point and join them."""
    result = find_intersection(poly1, poly2)
    if result is None:
        return None, None, None

    i, j, inter = result

    # Option: poly1 up to intersection, poly2 from intersection
    poly1_trimmed = np.vstack([poly1[:i + 1], inter])
    poly2_trimmed = np.vstack([inter, poly2[j + 1:]])

    joined = np.vstack([poly1_trimmed, poly2_trimmed])
    return joined, poly1_trimmed, poly2_trimmed

def plot_join(poly1, poly2, joined, inter):
    plt.figure(figsize=(8, 6))
    plt.plot(poly1[:, 0], poly1[:, 1], 'b--o', label='Polyline 1 (original)', alpha=0.5)
    plt.plot(poly2[:, 0], poly2[:, 1], 'g--o', label='Polyline 2 (original)', alpha=0.5)
    plt.plot(joined[:, 0], joined[:, 1], 'r-o', linewidth=2, label='Joined Polyline')
    plt.scatter(inter[0], inter[1], s=100, color='purple', label='Intersection')
    plt.legend()
    plt.title("Joined Polyline at Intersection")
    plt.grid(True)
    plt.show()

# Example polylines
poly1 = np.array([[0, 0], [2, 2], [4, 0]])
poly2 = np.array([[1, 3], [2, 1], [3, -1]])

joined, trimmed1, trimmed2 = join_polylines_at_intersection(poly1, poly2)
if joined is not None:
    _, _, intersection_point = find_intersection(poly1, poly2)
    plot_join(poly1, poly2, joined, intersection_point)
else:
    print("No intersection found between polylines.")
