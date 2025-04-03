import numpy as np
import matplotlib.pyplot as plt

def line_intersection(p1, p2, p3, p4):
    """Computes the intersection point of segments (p1, p2) and (p3, p4), if it exists."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:  # Parallel or coincident lines
        return None

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

    if (
        min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2)
        and min(x3, x4) <= px <= max(x3, x4) and min(y3, y4) <= py <= max(y3, y4)
    ):
        return np.array([px, py])
    return None

def find_self_intersections(points):
    """Finds exact intersection points and their indices."""
    intersections = []

    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        for j in range(i + 2, len(points) - 1):  # Avoid adjacent segments
            p3, p4 = points[j], points[j + 1]
            inter = line_intersection(p1, p2, p3, p4)
            if inter is not None:
                intersections.append((i + 1, j, inter))  # Store indices and intersection point

    if not intersections:
        return None

    first_idx, last_idx, first_pt = min(intersections, key=lambda x: x[0])
    _, _, last_pt = max(intersections, key=lambda x: x[1])

    return first_idx, last_idx, first_pt, last_pt

def trim_polyline(points):
    """Trims the polyline to start and end at exact intersection points."""
    result = find_self_intersections(points)
    if result is None:
        return points  # No trimming needed

    first_idx, last_idx, first_pt, last_pt = result
    return np.vstack([first_pt, points[first_idx:last_idx + 1], last_pt])

def plot_polylines(original, trimmed, first_pt, last_pt):
    """Plots the original and trimmed polylines for visualization."""
    plt.figure(figsize=(8, 6))

    # Plot original polyline
    plt.plot(original[:, 0], original[:, 1], 'b-o', label="Original Polyline", alpha=0.5)

    # Plot trimmed polyline
    plt.plot(trimmed[:, 0], trimmed[:, 1], 'r-o', label="Trimmed Polyline", linewidth=2)

    # Mark intersection points
    plt.scatter([first_pt[0], last_pt[0]], [first_pt[1], last_pt[1]], color='green', s=100, label="Intersections", zorder=3)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.title("Polyline Trimming with Exact Intersections")
    plt.show()

# Example polyline
polyline = np.array([
    [0, 0], [1, 2], [2, 4], [3, 2], [4, 0],  # Up and down
    [2, -2], [0, -4], [0, -2], [1, 0], [4, 1]  # Looping back
])

trimmed_polyline = trim_polyline(polyline)

# Get intersection points for plotting
result = find_self_intersections(polyline)
if result:
    _, _, first_pt, last_pt = result
    plot_polylines(polyline, trimmed_polyline, first_pt, last_pt)
else:
    print("No intersections found.")
