import numpy as np

# covariance and eigenvalues
# http://www.cs.utah.edu/~tch/CS4640F2019/resources/A%20geometric%20interpretation%20of%20the%20covariance%20matrix.pdf

# multivariate_gaussian function from
# https://stackoverflow.com/questions/28342968/how-to-plot-a-2d-gaussian-with-different-sigma
MV_GAUSS_FACTOR = 2*np.pi
def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""

    n = mu.shape[1]
    assert n==2, mu.shape

    Sigma_det = Sigma[0,0] * Sigma[1,1] - (Sigma[0,1] * Sigma[1,0])
    assert Sigma_det > 0.0
    Sigma_inv = 1/Sigma_det * np.array([[Sigma[1,1], -Sigma[0,1]],[-Sigma[1,0], Sigma[0,0]]])

    N = MV_GAUSS_FACTOR * np.sqrt(Sigma_det)

    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    # same calc as np.einsum but slower
    # pos_mu = (pos-mu)[:,:,:, np.newaxis]
    # pos_mu_T = pos_mu.transpose(0,1,3,2)
    # Sigma_inv = Sigma_inv[np.newaxis, np.newaxis, :, :]
    # fac = np.squeeze(pos_mu_T @ Sigma_inv @ pos_mu)

    return np.exp(-fac / 2) / N

def draw_gauss(angle, width, length, center, shape, points=None, scaling=1.0, fast_mode=True):
    # eigen values of grasp rectangle
    e1 = width * scaling
    e2 = length * scaling

    if points:
        # normalized eigen vectors
        # TODO: validate order of v1/v2
        print("WARNING: draw gauss from points needs to be validated!")
        v1 = (points[1] - points[2]) / e1
        v2 = (points[0] - points[1]) / e2
    else:
        v1 = np.array([np.cos(angle),np.sin(angle)])
        # prevent over flow of range -pi..pi
        if angle < 0:
            v2 = np.array([np.cos(angle+np.pi/2),np.sin(angle+np.pi/2)])
        else:
            v2 = np.array([np.cos(angle-np.pi/2),np.sin(angle-np.pi/2)])

    # Sigma = V*L*V^-1 -> eigenvalue decomp
    # V: mat with eigen vectors as cols
    # L: diag mat with corresponding eigen values as elements
    V = np.array([v1, v2]).T
    V_inv = np.linalg.inv(V)
    L = np.eye(2)*np.array([e1, e2])
    Sigma = (V.dot(L)).dot(V_inv)

    if not fast_mode:
        # define variable space based on image shape
        X = np.linspace(0,shape[1], shape[1])
        Y = np.linspace(0, shape[0], shape[0])
    else:
        # define variable space based on rectangle around the grasp
        # compute xy rectangle around the grasp using eigen values and vectors
        V_dot_L = V.dot(L)
        grasp_xy = np.max(np.abs(V_dot_L),axis=0)
        # get top left and bottom right corner in image coordinates
        tl = (center - grasp_xy *  3 / 4).reshape(-1)
        br = (center + grasp_xy *  3 / 4).reshape(-1)
        # use this more efficient variable space
        # we need to sample twice as much pixels as the rectangle contains to respect the sampling theorem
        # max(, 1) prevents a 0 (just in case)
        X = np.linspace(tl[0], br[0], 2*max(int(br[0]-tl[0]), 1))
        Y = np.linspace(tl[1], br[1], 2*max(int(br[1]-tl[1]),1))

    X, Y = np.meshgrid(X, Y)
    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # grasp center as mu
    mu = center

    # compute gauss
    mv_gauss = multivariate_gaussian(pos, mu, Sigma)
    # normalize values
    mv_gauss -= mv_gauss.min()
    mv_gauss /= mv_gauss.max()

    if not fast_mode:
        return mv_gauss
    else:
        gauss_label = np.squeeze(np.zeros(shape))
        # cast float variable space to image coordinates
        X = np.ceil(X).astype(int)
        Y = np.ceil(Y).astype(int)
        # take care of image borders
        mask = ((X<gauss_label.shape[1]) & (X>=0) & (Y<gauss_label.shape[0]) & (Y>=0))
        gauss_label[Y[mask], X[mask]] = mv_gauss[mask]
        return gauss_label

# test
if __name__ == "__main__":
    from nicr_grasping.datatypes.grasp import RectangleGrasp, RectangleGraspDrawingMode
    from PIL import Image
    import time

    # dummy image and grasp
    np_image = np.zeros((400, 500))
    r = np.array([10, 10, 90, 90])+50
    c = np.array([10, 60, 60, 10])+50
    grasp_points = np.zeros((2, 4))
    grasp_points[0,:] = c
    grasp_points[1,:] = r

    # rotate the grasp a little bit
    gp_mean = grasp_points.mean(axis=1, keepdims=True)
    grasp_points -= gp_mean
    theta = np.radians(-30)
    r = np.array(( (np.cos(theta), -np.sin(theta)),
                (np.sin(theta),  np.cos(theta)) ))
    grasp_points = r.dot(grasp_points) + gp_mean

    # test with grasp
    grasp = RectangleGrasp.from_points(grasp_points.T)

    gauss_start = time.time()
    N = 10
    for _ in range(N):
        grasp_label_gauss = [np.zeros((400,500)), np.zeros((400,500)), np.zeros((400,500))]
        mode = RectangleGraspDrawingMode.GAUSS
        grasp_label_gauss = grasp.draw_label(grasp_label_gauss, mode=mode)
    gauss_time = (time.time()-gauss_start) / N

    rect_start = time.time()
    N = 10
    for _ in range(N):
        grasp_label = [np.zeros((400,500)), np.zeros((400,500)), np.zeros((400,500))]
        mode = RectangleGraspDrawingMode.INNER_RECTANGLE
        grasp_label = grasp.draw_label(grasp_label, mode=mode)
    rect_time = (time.time()-rect_start) / N

    print(f"gauss mode draw_label time: {gauss_time}")
    print(f"rect mode draw_label time: {rect_time}")

    # visualize
    im = Image.fromarray(grasp_label[0]*255.0)
    im = im.convert("L")
    #im.save("test_grasp.png")

    # quality
    im = Image.fromarray(grasp_label_gauss[0]*255.0)
    im = im.convert("L")
    #im.save("test_grasp_gauss.png")

    # angle
    ang = grasp_label_gauss[1]
    ang -= ang.min()
    ang /= ang.max()
    im = Image.fromarray(ang*255.0)
    im = im.convert("L")
    #im.save("test_grasp_gauss_ang.png")

    # width
    width = grasp_label_gauss[2]
    width -= width.min()
    width /= width.max()
    im = Image.fromarray(width*255.0)
    im = im.convert("L")
    #im.save("test_grasp_gauss_w.png")