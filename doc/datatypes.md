# Datatypes

## Overview

![](classes.png)

## 2D grasps
A 2d grasp ist defined by a set of points. 
In general the only type of 2d grasp ist the rectangluar representation of a grasp with a parallel gripper described below.


### Rectangle grasp
The following image shows the definition of a rectangle grasp.

![](grasp_def.png)

The displayed grasp has the following parameters supplied to the RectangleGrasp constructor:
* center = np.array([[75, 125]])
* width = 100
* angle = np.pi / 5

The red lines represent the gripper jaws whereas the blue lines are only connecting lines for visualization.

The internal representation of a rectangle grasp is a set of four points describing the edges of the rectangle in the shown order.
The coordinates of a point are given in image space.
Rotation direction is counterclockwise in image space.