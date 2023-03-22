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


def quad_moment(ptcl_coord, centr, dens):

    centr_x = centr[0]
    centr_y = centr[1]
    centr_z = centr[2]
    ptcl_coord_x = ptcl_coord[:, 0]
    ptcl_coord_y = ptcl_coord[:, 1]
    ptcl_coord_z = ptcl_coord[:, 2]

    rx = (ptcl_coord_x - centr_x) * dens
    ry = (ptcl_coord_y - centr_y) * dens
    rz = (ptcl_coord_z - centr_z) * dens

    # print(rx)
    print(np.mean(rx**2))
    print(np.mean(ry**2))
    print(np.mean(rz**2))

    # R_range = np.sqrt(rx**2. + ry**2. + rz**2.)
    # rmax = np.sqrt(np.max(r_mem_ptcl[:,3]))
    # print "Number of particles before selection is ", len(rx)

    # Choose particles inside Rvir
    # ptcl_range = np.where(R_range < rvir)
    # rx = rx[ptcl_range]; ry = ry[ptcl_range]; rz = rz[ptcl_range]

    num_mem_ptcl = len(rx)
    print("Number of particles inside virial radius is ", num_mem_ptcl)

    # Building quadrupole tensor.
    Rp = np.sqrt(rx**2.0 + ry**2.0 + rz**2.0)
    # print(Rp)
    assert np.all(Rp > 0.0), "Rp is zero"
    r = np.array([rx, ry, rz])
    # print(r[0])
    r_rdu = r / Rp
    # print(r_rdu[0])
    # print(len(Rp**2))
    M_rdu = np.matmul(r_rdu, r_rdu.T)  # Initial quadrupole tensor before iteration
    print(M_rdu)

    # Finding eigvec, eigval
    M_eigval, M_eigvec = np.linalg.eig(M_rdu)
    sort_eigval = np.argsort(M_eigval)[::-1]  # from greater to smaller
    a, b, c = np.sqrt(M_eigval[sort_eigval])  # a, b, c major, intermediate, minor
    lx, ly, lz = M_eigvec.T[sort_eigval]  # lx, ly, lz major, intermediate, minor
    print("a,b,c in the first iteration")
    print(a, b, c)
    lx = np.array(lx)
    ly = np.array(ly)
    lz = np.array(lz)
    print("lx in the first iteration", lx)
    print("ly in the first iteration", ly)
    print("lz in the first iteration", lz)

    # Sanity check
    """
    print "r_rdu", r_rdu
    check_eig = M_rdu.dot(lx) - num_mem_ptcl*c**2.*lx
    print "M_rdu.dot(lx) ", np.dot(np.array(M_rdu), lx)
    print "check_eig ", check_eig
    print "lx is ", lx
    print "M_eigvec.T[sort_eigval], ", M_eigvec.T[sort_eigval]
    print "M_eigvec[:,0] ", M_eigvec[:,0]
    print "M_eigvec[sort_eigval] ", M_eigvec[sort_eigval]
    print "M_eigvec", M_eigvec
    print "sort_eigval ", sort_eigval
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
        print("P_axis:", P_axis)
        P_tot = np.matmul(P_axis, P_tot)
        r_proj = np.matmul(P_axis, r)
        rx = np.array(r_proj[0])
        ry = np.array(r_proj[1])
        rz = np.array(r_proj[2])

        # New iteration
        q_cur = c / a
        s_cur = b / a  # Osato conventaion

        Rp = np.sqrt((rx) ** 2.0 + (ry / s_cur) ** 2.0 + (rz / q_cur) ** 2.0)

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
        # print "Conv_s, conv_q ", conv_s, conv_q
        # print "Number of particles ", len(rx)
        print("a, b, c ", a, b, c)
        # print "q, s are ", q_cur, s_cur
        # print "lx", lx
        # print 'converge is ', converge
        # print '\n'
        conv_iter += 1
        q_prev = q_cur
        s_prev = s_cur
        q_tot *= q_cur
        s_tot *= s_cur

        print("q_cur, s_cur", q_cur, s_cur)

    # find lx, ly, lz in original basis
    P_inv = np.linalg.inv(P_tot)
    l_new_basis = np.array([lx, ly, lz])
    l_orig_basis = np.matmul(P_inv, l_new_basis)
    lx_orig = l_orig_basis[0]
    ly_orig = l_orig_basis[1]
    lz_orig = l_orig_basis[2]

    return converge, [a, b, c], [lx_orig, ly_orig, lz_orig]