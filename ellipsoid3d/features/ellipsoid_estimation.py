import numpy as np
"""
Updated April 16, 2018
Follow convention of Ken Osato: Use reduced quadropole moment to find axis ratio of ellipsoidal cluster
1. Project onto principle axes spitted out by quadropole tensor
2. Do not remove particles. Particles chosen for those inside Rvir
3. Use Reduced tensor
4. q, s refer to ratio of minor to major, and intermediate to major axis
Returns:
converge -- Boolean
[a,b,c] -- normalized major, intermediate, minor axes lengths (only ratio matters in reduced tensor)
[lx, ly, lz] -- direction of minor, intermediate, major in original (non-rotated) basis
"""

import logging

logging.basicConfig(level=logging.DEBUG)


def reduced_iterative_tensor_shell(particle_coordinates, center_coordinate, ellip_radius_min, ellipradius_max):
    
    center_x, center_y, center_z = center_coordinate
    particle_coordinates_x = particle_coordinates[:, 0]
    particle_coordinates_y = particle_coordinates[:, 1]
    particle_coordinates_z = particle_coordinates[:, 2]
    
    rx = (particle_coordinates_x - center_x)
    ry = (particle_coordinates_y - center_y)
    rz = (particle_coordinates_z - center_z)

    convergence = False
    iter = 0
    maxiter = 1000
    convergence_tolerance = 1e-6
    projection_matrix = np.identity(3)
    s = 1.0
    q = 1.0
    
    while (not convergence) & (iter < maxiter):
        # use reduced moment of inertia tensor to find axis ratio within shell
        rp = np.sqrt(rx**2 + (ry/s)**2 + (rz/q)**2)
        mask = (rp > ellip_radius_min) & (rp < ellipradius_max)

        rx_in_ellipsoid, ry_in_ellipsoid, rz_in_ellipsoid = rx[mask], ry[mask], rz[mask]
        assert np.all(rp > 0.0), "Rp is zero"
        r = np.array([rx_in_ellipsoid, ry_in_ellipsoid, rz_in_ellipsoid])
        r = np.matmul(projection_matrix, r)
        r_reduced = r / rp
        M_reduced = np.matmul(r_reduced, r_reduced.T)

        M_eigenvalues, M_eigenvectors = np.linalg.eig(M_reduced)
        sort_eigenvalues = np.argsort(M_eigenvalues)[::-1]
        a,b,c = np.sqrt(M_eigenvalues[sort_eigenvalues])
        lx,ly,lz = M_eigenvectors.T[sort_eigenvalues]
        lx, ly, lz = np.array(lx), np.array(ly), np.array(lz)
        
        projection_matrix *= np.array([lx, ly, lz])


        q_new, s_new = b/a, c/a

        convergence = (np.abs(q_new - q) < convergence_tolerance) & (np.abs(s_new - s) < convergence_tolerance)
        iter += 1
        
        

    inverse_projection = np.linalg.inv(projection_matrix)
    l_new_basis = np.array([lx, ly, lz])
    l_orig_basis = np.matmul(inverse_projection, l_new_basis)
    lx_orig = l_orig_basis[0]
    ly_orig = l_orig_basis[1]
    lz_orig = l_orig_basis[2]
    
    assert convergence, "Did not converge"

    return convergence, [a, b, c], [lx_orig, ly_orig, lz_orig]



def reduced_iterative_tensor(ptcl_coord, centr, dens):

    centr_x = centr[0]
    centr_y = centr[1]
    centr_z = centr[2]

    ptcl_coord_x = ptcl_coord[:, 0]
    ptcl_coord_y = ptcl_coord[:, 1]
    ptcl_coord_z = ptcl_coord[:, 2]

    rx = (ptcl_coord_x - centr_x) * dens
    ry = (ptcl_coord_y - centr_y) * dens
    rz = (ptcl_coord_z - centr_z) * dens

    # logging.debug(rx)
    logging.debug(np.mean(rx**2))
    logging.debug(np.mean(ry**2))
    logging.debug(np.mean(rz**2))

    # R_range = np.sqrt(rx**2. + ry**2. + rz**2.)
    # rmax = np.sqrt(np.max(r_mem_ptcl[:,3]))
    # logging.debug "Number of particles before selection is ", len(rx)

    # Choose particles inside Rvir
    # ptcl_range = np.where(R_range < rvir)
    # rx = rx[ptcl_range]; ry = ry[ptcl_range]; rz = rz[ptcl_range]

    num_mem_ptcl = len(rx)
    logging.debug(
        "Number of particles inside virial radius is {}".format(num_mem_ptcl))

    # Building quadrupole tensor.
    Rp = np.sqrt(rx**2.0 + ry**2.0 + rz**2.0)
    # logging.debug(Rp)
    assert np.all(Rp > 0.0), "Rp is zero"
    r = np.array([rx, ry, rz])
    # logging.debug(r[0])
    r_rdu = r / Rp
    # logging.debug(r_rdu[0])
    # logging.debug(len(Rp**2))
    M_rdu = np.matmul(r_rdu,
                      r_rdu.T)  # Initial quadrupole tensor before iteration
    logging.debug(M_rdu)

    # Finding eigvec, eigval
    M_eigval, M_eigvec = np.linalg.eig(M_rdu)
    sort_eigval = np.argsort(M_eigval)[::-1]  # from greater to smaller
    a, b, c = np.sqrt(
        M_eigval[sort_eigval])  # a, b, c major, intermediate, minor
    lx, ly, lz = M_eigvec.T[
        sort_eigval]  # lx, ly, lz major, intermediate, minor
    logging.debug("a,b,c in the first iteration: {} {} {}".format(a, b, c))
    lx = np.array(lx)
    ly = np.array(ly)
    lz = np.array(lz)
    logging.debug("lx in the first iteration {}".format(lx))
    logging.debug("ly in the first iteration {}".format(ly))
    logging.debug("lz in the first iteration {}".format(lz))

    # Sanity check
    """
    logging.debug "r_rdu", r_rdu
    check_eig = M_rdu.dot(lx) - num_mem_ptcl*c**2.*lx
    logging.debug "M_rdu.dot(lx) ", np.dot(np.array(M_rdu), lx)
    logging.debug "check_eig ", check_eig
    logging.debug "lx is ", lx
    logging.debug "M_eigvec.T[sort_eigval], ", M_eigvec.T[sort_eigval]
    logging.debug "M_eigvec[:,0] ", M_eigvec[:,0]
    logging.debug "M_eigvec[sort_eigval] ", M_eigvec[sort_eigval]
    logging.debug "M_eigvec", M_eigvec
    logging.debug "sort_eigval ", sort_eigval
    """

    # Initial conditions
    q_tot = 1.0
    s_tot = 1.0
    q_prev = 1.0
    s_prev = 1.0
    converge = False
    conv_iter = 0

    P_tot = np.eye(
        3
    )  # the multiplicative product of all projections done over each iteration
    while (not converge) & (conv_iter < 1000):
        # Change of basis
        P_axis = np.array([lx, ly, lz])
        logging.debug("P_axis: {}".format(P_axis))
        P_tot = np.matmul(P_axis, P_tot)
        r_proj = np.matmul(P_axis, r)
        rx = np.array(r_proj[0])
        ry = np.array(r_proj[1])
        rz = np.array(r_proj[2])

        # New iteration
        q_cur = c / a
        s_cur = b / a  # Osato conventaion

        Rp = np.sqrt((rx)**2.0 + (ry / s_cur)**2.0 + (rz / q_cur)**2.0)

        r = np.array([rx, ry, rz])

        r_rdu = r / Rp

        M_rdu = np.matmul(r_rdu, r_rdu.T)
        M_eigval, M_eigvec = np.linalg.eig(M_rdu)
        sort_eigval = np.argsort(M_eigval)[::-1]
        a, b, c = np.sqrt(M_eigval[sort_eigval])
        lx, ly, lz = M_eigvec.T[sort_eigval]
        lx = np.array(lx)
        ly = np.array(ly)
        lz = np.array(lz)

        # test converge
        conv_err = 1e-6
        conv_s = np.abs(1 - s_cur / s_prev)
        conv_q = np.abs(1 - q_cur / q_prev)
        converge = (conv_s < conv_err) & (conv_q < conv_err)
        # logging.debug "Conv_s, conv_q ", conv_s, conv_q
        # logging.debug "Number of particles ", len(rx)
        logging.debug("a,b,c in the {} iteration: {} {} {}".format(
            conv_iter, a, b, c))
        # logging.debug "q, s are ", q_cur, s_cur
        # logging.debug "lx", lx
        # logging.debug 'converge is ', converge
        # logging.debug '\n'
        conv_iter += 1
        q_prev = q_cur
        s_prev = s_cur
        q_tot *= q_cur
        s_tot *= s_cur

        logging.debug("q_cur, s_cur: {} {}".format(q_cur, s_cur))

    # find lx, ly, lz in original basis
    P_inv = np.linalg.inv(P_tot)
    l_new_basis = np.array([lx, ly, lz])
    l_orig_basis = np.matmul(P_inv, l_new_basis)
    lx_orig = l_orig_basis[0]
    ly_orig = l_orig_basis[1]
    lz_orig = l_orig_basis[2]

    assert converge

    return converge, [a, b, c], [lx_orig, ly_orig, lz_orig]
