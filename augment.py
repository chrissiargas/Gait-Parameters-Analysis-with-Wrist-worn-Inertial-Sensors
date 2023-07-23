import numpy as np
from scipy.interpolate import CubicSpline


def GenerateRandomCurves(N, sigma=0.2, knot=4, xyz=False):
    if not xyz:
        xx = (np.arange(0, N, (N - 1) / (knot + 1))).transpose()
        yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2))
        x_range = np.arange(N)
        cs_x = CubicSpline(xx[:], yy[:])
        return np.array([cs_x(x_range)]).transpose()

    else:
        xx = (np.ones((3, 1)) * (np.arange(0, N, (N - 1) / (knot + 1)))).transpose()
        yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, 3))
        x_range = np.arange(N)
        cs_x = CubicSpline(xx[:, 0], yy[:, 0])
        cs_y = CubicSpline(xx[:, 1], yy[:, 1])
        cs_z = CubicSpline(xx[:, 2], yy[:, 2])
        return np.array([cs_x(x_range), cs_y(x_range), cs_z(x_range)]).transpose()


def DistortTimesteps(N, sigma=0.2, xyz=False):
    if not xyz:
        tt = GenerateRandomCurves(N, sigma)
        tt_cum = np.cumsum(tt, axis=0)
        t_scale = [(N - 1) / tt_cum[-1]]
        tt_cum[:] = tt_cum[:] * t_scale
        return tt_cum

    else:
        tt = GenerateRandomCurves(N, sigma, xyz=xyz)
        tt_cum = np.cumsum(tt, axis=0)
        t_scale = [(N - 1) / tt_cum[-1, 0], (N - 1) / tt_cum[-1, 1], (N - 1) / tt_cum[-1, 2]]
        tt_cum[:, 0] = tt_cum[:, 0] * t_scale[0]
        tt_cum[:, 1] = tt_cum[:, 1] * t_scale[1]
        tt_cum[:, 2] = tt_cum[:, 2] * t_scale[2]
        return tt_cum


def DA_TimeWarp(N, sigma=0.2, xyz=False):
    if not xyz:
        tt_new = DistortTimesteps(N, sigma)
        tt_new = np.squeeze(tt_new)
        x_range = np.arange(N)
        return tt_new, x_range

    else:
        tt_new = DistortTimesteps(N, sigma, xyz)
        x_range = np.arange(N)
        return tt_new, x_range


def DA_Permutation(N, nPerm=4, minSegLength=10, xyz=False):
    segs = None
    if not xyz:
        bWhile = True
        while bWhile:
            segs = np.zeros(nPerm + 1, dtype=int)
            segs[1:-1] = np.sort(np.random.randint(minSegLength, N - minSegLength, nPerm - 1))
            segs[-1] = N
            if np.min(segs[1:] - segs[0:-1]) > minSegLength:
                bWhile = False

        return segs


def permutate(signal, N, segs, idx, nPerm=4):
    pp = 0
    X_new = np.zeros(N)

    for ii in range(nPerm):
        x_temp = signal[segs[idx[ii]]:segs[idx[ii] + 1]]
        X_new[pp:pp + len(x_temp)] = x_temp
        pp += len(x_temp)

    return X_new


def DA_Rotation(X):
    axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(X, axis_angle_to_rotation_matrix_3d_vectorized(axis, angle))


def axis_angle_to_rotation_matrix_3d_vectorized(axes, angles):
    x, y, z = axes

    n = np.sqrt(x * x + y * y + z * z)
    x = x / n
    y = y / n
    z = z / n

    c = np.cos(angles)
    s = np.sin(angles)
    C = 1 - c

    xs = x * s
    ys = y * s
    zs = z * s
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    return np.array([
        [x * xC + c, xyC - zs, zxC + ys],
        [xyC + zs, y * yC + c, yzC - xs],
        [zxC - ys, yzC + xs, z * zC + c]])

