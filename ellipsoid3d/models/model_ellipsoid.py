import numpy as np

def uniform_ellipsoid_shell(a,b,c,n_points=100000):

    # Generate uniformly distributed points on a unit sphere
    u = np.random.rand(n_points)
    v = np.random.rand(n_points)
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    # Scale the coordinates of the points by the semi-axes of the ellipsoid
    points = np.vstack((a*x, b*y, c*z)).T

    return points

    

def uniform_ellipsoid(a,b,c,n_points=int(1e6)):
    # Generate points in semi axis cube
    x = np.random.uniform(-a, a, n_points)
    y = np.random.uniform(-b, b, n_points)
    z = np.random.uniform(-c, c, n_points)
    
    mask = (x**2/a**2 + y**2/b**2 + z**2/c**2) < 1.0
    
    x,y,z = x[mask], y[mask], z[mask]

    return np.vstack((x,y,z)).T