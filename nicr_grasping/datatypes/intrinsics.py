import numpy as np


class CameraIntrinsic:
    def __init__(self) -> None:
        pass

class PinholeCameraIntrinsic(CameraIntrinsic):
    def __init__(self,
                 fx: float = 0,
                 fy: float = 0,
                 cx: float = 0,
                 cy: float = 0) -> None:
        super().__init__()

        self._fx = fx
        self._fy = fy

        self._cx = cx
        self._cy = cy

    @property
    def camera_matrix(self):
        res = np.zeros((3, 3))
        res[0, 0] = self._fx
        res[1, 1] = self._fy
        res[0, 2] = self._cx
        res[1, 2] = self._cy
        res[2, 2] = 1

        return res

    def pixel_to_point(self,
                        pixels: np.ndarray,
                        depth: float):
        """Converts pixel coordinates into 3d positions based on depth and intrinsics.

        Parameters
        ----------
        pixels : np.ndarray
            Matrix of pixel coordinates. Shape (NUM_PIXELS, 2)
        depth : float
            Depth used for all pixels

        Returns
        -------
        np.ndarray
            Matrix with 3d positions. Shape (NUM_PIXELS, 3)
        """
        res = np.zeros((len(pixels), 3))
        res[:, 2] = depth
        res[:, 0] = res[:, 2] / self._fx * (pixels[:, 0] - self._cx) # mm
        res[:, 1] = res[:, 2] / self._fy * (pixels[:, 1] - self._cy) # mm

        return res

    def point_to_pixel(self, points_3d):
        res = np.zeros((len(points_3d), 2))

        res[:, 0] = points_3d[:, 0] / points_3d[:, 2]
        res[:, 1] = points_3d[:, 1] / points_3d[:, 2]

        # r = np.sqrt(x1**2 + y1**2)
        # r4 = r * r * r * r

        # x11 = x1 * (1 + intr.k1 * r * r + intr.k2 * r4) + 2 * intr.p1 * x1 * y1 + intr.p2 * (r * r + 2 * x1 * x1)

        # y11 = y1 * (1 + intr.k1 * r * r + intr.k2 * r4) + intr.p1 * (r * r + 2 * y1 * y1) + 2 * intr.p2 * x1 * y1
        # x11 = x1
        # y11 = y1

        # res = np.array([self._fx * x1 + self._cx, self._fy * y1 + self._cy])
        res[:, 0] = res[:, 0] * self._fx + self._cx
        res[:, 1] = res[:, 1] * self._fy + self._cy

        return res
