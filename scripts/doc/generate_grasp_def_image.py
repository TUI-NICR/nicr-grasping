import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

from nicr_grasping.datatypes.grasp import RectangleGrasp

img = np.ones((200, 200, 3)) * 255
grasp = RectangleGrasp(1, np.array([[75, 125]]), 100, np.pi / 5)

img = grasp.plot(img)

f, ax = plt.subplots(1, 1)
ax.imshow(img)
ax.scatter(grasp.points[:, 0], grasp.points[:, 1])


for i in range(4):
    ax.annotate(str(i), (grasp.points[i, 0] + 2, grasp.points[i, 1] + 2), color='green', fontweight='bold', fontsize='large')

# angle
ax.plot([grasp.center[:, 0], grasp.center[:, 0] + 50], [grasp.center[:, 1], grasp.center[:, 1]], color='black')
up = grasp.points[:2].mean(axis=0)
ax.plot([grasp.center[:, 0], up[0]], [grasp.center[:, 1], up[1]], color='black')

arc = Arc(grasp.center.squeeze(), 50, 50, 0, -grasp.angle / np.pi * 180, 0, color='black')
ax.add_patch(arc)

ax.annotate(f'{grasp.angle / np.pi * 180:.2f}', grasp.center.squeeze() + [10, -2])

# center
ax.scatter(grasp.center[:, 0], grasp.center[:, 1])
ax.annotate('center', (grasp.center[:, 0], grasp.center[:, 1] + 10), ha='center')

f.tight_layout()

f.savefig('grasp_def.png')
