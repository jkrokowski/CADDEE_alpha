import numpy as np
import matplotlib.pyplot as plt

def line_intersection(p1, p2, p3, p4):
    """Returns intersection point of segments (p1,p2) and (p3,p4), or None."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if abs(denom) < 1e-10:
        return None

    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    point = np.array([px, py])

    # Check if intersection point lies on both segments
    if (min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2) and
        min(x3, x4) <= px <= max(x3, x4) and min(y3, y4) <= py <= max(y3, y4)):
        return point
    return None

def find_last_intersection(poly1, poly2, start1="start", start2="start"):
    """Find the last intersection based on direction of traversal."""
    range1 = range(len(poly1) - 1)
    range2 = range(len(poly2) - 1)

    if start1 == "end":
        range1 = reversed(range1)
    if start2 == "end":
        range2 = reversed(range2)

    last = None
    for i in range1:
        p1, p2 = poly1[i], poly1[i + 1]
        for j in range2:
            q1, q2 = poly2[j], poly2[j + 1]
            inter = line_intersection(p1, p2, q1, q2)
            if inter is not None:
                last = (i, j, inter)
    return last

def join_polylines(poly1, poly2, start1="start", start2="start"):
    """Join polylines at last detected intersection in specified directions."""
    result = find_last_intersection(poly1, poly2, start1, start2)
    if result is None:
        return None, None, None, None

    i, j, inter = result

    if start1 == "start":
        poly1_trim = np.vstack([poly1[:i+1], inter])
    else:
        poly1_trim = np.vstack([inter, poly1[i+1:][::-1]])

    if start2 == "start":
        poly2_trim = np.vstack([inter, poly2[j+1:]])
    else:
        poly2_trim = np.vstack([poly2[:j+1][::-1], inter])

    joined = np.vstack([poly1_trim, poly2_trim[1:]])  # Avoid duplicating the intersection point
    return joined, poly1_trim, poly2_trim, inter

def plot_join(poly1, poly2, joined, inter):
    plt.figure(figsize=(8, 6))
    plt.plot(poly1[:, 0], poly1[:, 1], 'b--o', label='Polyline 1', alpha=0.4)
    plt.plot(poly2[:, 0], poly2[:, 1], 'g--o', label='Polyline 2', alpha=0.4)
    plt.plot(joined[:, 0], joined[:, 1], 'r-o', label='Joined Polyline', linewidth=2)
    plt.scatter(inter[0], inter[1], s=100, color='purple', label='Intersection')
    plt.title("Join at Last Intersection")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# Example: Two intersecting polylines
poly1 = np.array([[0, 0], [2, 2], [4, 0]])
poly2 = np.array([[1, 3], [2, 1], [3, -1]])

# Try different start directions
joined, t1, t2, inter = join_polylines(poly1, poly2, start1="start", start2="start")

if joined is not None:
    plot_join(poly1, poly2, joined, inter)
else:
    print("No intersection found.")
