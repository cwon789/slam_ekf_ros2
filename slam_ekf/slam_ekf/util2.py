#!/usr/bin/python3
import numpy as np
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation

def angle_between_yaw(yaw1, yaw2):
    """calculates the angle between two frames
    specified by their yaw angles. Avoid having
    to deal with wrapping the angles by expressing
    frame 2 under frame 1

    Args:
        yaw1 (_type_): yaw angle of the ref frame
        yaw2 (_type_): yaw angle of the query frame/vector

    Returns:
        theta: yaw2 minus yaw1 expressed in yaw1 frame
    """
    s = np.sin(yaw1)
    c = np.cos(yaw1)
    R = np.array([[c,-s],[s,c]])
    p = np.array([np.cos(yaw2),np.sin(yaw2)])[:,np.newaxis]
    p_ = R.T.dot(p) # expressed in the frame of yaw1
    theta = np.arctan2(p_[1,0],p_[0,0])
    return theta

def get_cov_ellipse_pts(mu, cov):
    """return the set of points on a ellipse that
      represents the mean and covariance of the gaussian.
      for plotting purposes.

    Args:
        mu (_type_): mean of the gaussian
        cov (_type_): covariance of the gaussian

    Returns:
        pts: 2 by n matrix of the points on the ellipse 
        representing the gaussian
    """
    x,y=mu
    # compute eig vector
    W,D = np.linalg.eig(cov)
    # set up the points
    ind = np.flip(np.argsort(W))
    W = np.sqrt(W[ind])
    D = D[:,ind]

    t = np.linspace(0,2*np.pi,30)
    xs = 2*W[0]*np.cos(t) # long axis
    ys = 2*W[1]*np.sin(t) # short axis
    pts = np.hstack((xs[:,np.newaxis],ys[:,np.newaxis]))
    pts = D.dot(pts.T) # rotate
    pts += np.array([[x,y]]).T
    return pts.T

def plot_cov(plot_handle, ax, mu, cov):
    """plot the gaussian as an ellipse

    Args:
        plot_handle (_type_): _description_
        ax : 
        mu (_type_): _description_
        cov (_type_): _description_
    """
    xys = get_cov_ellipse_pts(mu=mu,cov=cov)
    if plot_handle is None:
        plot_handle = \
            ax.plot(xys[:,0],xys[:,1],
                    linestyle='-.',
                    color=(0.2,0.2,0.2),
                    linewidth=0.5)[0]
    else:
        plot_handle.set_data(xys[:,0],xys[:,1])
    
    return plot_handle

def corner_loss(x, data):
    """calculate the loss for fitting a corner feature
    based on lidar reflection points

    Args:
        x (_type_): a vector containing the corner location
        data (_type_): the lidar points expressed as xy coordinates

    Returns:
        out: the loss to be minimized
    """
    # data is a 2D array
    distances = np.linalg.norm(data - x[np.newaxis, :], axis=1)
    angles = np.arctan2(data[:, 1] - x[1], data[:, 0] - x[0])
    angle_diffs = np.abs(np.diff(np.sort(angles)))
    out = np.sum(distances ** 2) + np.sum(angle_diffs ** 2)
    return out

def initial_corner_guess(data: np.ndarray):
    """calculates the initial guess of the corner location

    Args:
        data (np.ndarray): lidar points

    Returns:
        _type_: initial guess for the corner location
    """
    centroid = np.mean(data, axis=0)
    return centroid

def get_corner_feature(data: np.ndarray):
    """find the corner feature through minimizing the loss function

    Args:
        data (np.ndarray): lidar points

    Returns:
        corner: the location of the detected corner
    """
    if data.shape[0] < 3:
        return None

    # compute initial guess
    v0 = initial_corner_guess(data)
    out = minimize(corner_loss, x0=v0, args=(data,))
    if out.success:
        return out.x

    return None

def quaternion_to_yaw(msg: Odometry):
    """extract yaw info from odom messages

    Args:
        msg (Odometry): the odometry message

    Returns:
        yaw: the yaw angle extracted from the quaternion
    """
    r = Rotation.from_quat([msg.pose.pose.orientation.x,
                            msg.pose.pose.orientation.y,
                            msg.pose.pose.orientation.z,
                            msg.pose.pose.orientation.w])
    rpy = r.as_euler('xyz')
    return rpy[-1]
