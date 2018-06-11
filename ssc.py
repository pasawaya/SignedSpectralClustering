import numpy as np
from sklearn.utils.extmath import randomized_svd


def norm_diff(X, Z, R):
    Q = Z.dot(R)
    return np.sqrt(np.trace((X - Q).H.dot(X - Q)))


# Given fixed X, Z, find R
# that minimizes phi(X, Z, R)
def find_r(x, z):
    u, _, v = np.linalg.svd(z.H.dot(x))
    return u.dot(v.H)


# Given fixed X, Z, find invertible Lambda
# that minimizes phi(X, Z, Lambda)
def find_a(x, z):
    y = z.H.dot(x)
    return np.lingalg.inv(z.h.dot(z)).dot(y)


def pick_orthogonal_rows(Z):
    zeroes = np.sum(np.abs(Z.H), axis=0)
    Z = Z[zeroes.nonzero()[1], :]
    n = Z.shape[0]
    k = Z.shape[1]
    R = np.eye(k)
    if n < k:
        return R

    Z = np.delete(Z, 0, axis=0)
    c = np.zeros((n-1, 1))
    for i in range(1, k):
        # Find index of row in Z most orthogonal to column i of R
        c += np.abs(Z.dot(R[:, i-1])).T
        min_c = np.argmin(c)

        # Add this row to R2
        R[:, i] = Z[min_c, :].conj()

        # Delete the row from Z
        Z = np.delete(Z, min_c, 0)
        c = np.delete(c, min_c, 0)

    nrc = np.zeros((k, k))
    rr = R.conj().T.dot(R)
    for i in range(0, k):
        nrc[i, i] = np.reciprocal(np.sqrt(rr[i, i]))
    return R.dot(nrc.conj().T)


# Z2 is identical to Z save for the columns
# in Z with negative averages. These columns
# have their signs flipped in Z2 to ensure
# non-negative averages.
def flip_negative_columns(z):
    n = z.shape[0]
    k = z.shape[1]
    rr = np.eye(k)
    col_sum = np.ones((1, n)).dot(z)
    z2 = z
    for i in range(0, k):
        if col_sum[0, i] < 0:
            z2[:, i] = -z[:, i]
            rr[i, i] = -1
    return [z2, rr]


# Computes discrete solution X using four different methods
# and returns method with minimum ||X - Z||_F. This is used
# to find an initial X*
def compute_initial_x_z_r(Z1, Z2, k, a, f):
    R2a = pick_orthogonal_rows(Z1)
    R2b = pick_orthogonal_rows(Z2)

    X1, RR1 = f(Z1, np.eye(k), a)
    R1 = RR1
    n1 = norm_diff(X1, Z1, R1)

    X2, RR2 = f(Z1, R2a, a)
    R2 = R2a.dot(RR2)
    n2 = norm_diff(X2, Z1, R2)

    X3, RR3 = f(Z2, np.eye(k), a)
    R3 = RR3
    n3 = norm_diff(X3, Z2, R3)

    X4, RR4 = f(Z2, R2b, a)
    R4 = R2b.dot(RR4)
    n4 = norm_diff(X4, Z2, R4)

    norms = [n1, n2, n3, n4]
    x_candidates = [X1, X2, X3, X4]
    z_candidates = [Z1, Z1, Z2, Z2]
    r_candidates = [R1, R2, R3, R4]
    X0 = x_candidates[np.argmin(norms)]
    Z0 = z_candidates[np.argmin(norms)]
    R0 = r_candidates[np.argmin(norms)]
    return [X0, Z0, R0]


def find_soft_clusters(Z, R, a):
    ZR1 = Z.dot(R)
    ZR2, Rn = flip_negative_columns(ZR1)
    for st in range(0, 2):
        ZR = ZR1 if st == 0 else ZR2
        ZR[ZR < 0] = 0
        n = ZR.shape[0]
        X1 = ZR
        for i in range(0, n):
            X1[i, :] = X1[i, :]/np.sum(X1[i, :])
        if st == 0:
            XX1 = a * X1
        else:
            XX2 = a * X1

    n1 = norm_diff(XX1, ZR1, np.eye(ZR1.shape[1]))
    n2 = norm_diff(XX2, ZR2, np.eye(ZR2.shape[1]))
    X = XX2 if n1 > n2 else XX1
    return [X, Rn]


def find_flex_clusters(z, r, a):
    zr1 = z.dot(r)
    k = zr1.shape[1]
    zr2, rn = flip_negative_columns(zr1)
    for st in range(0, 2):
        zr = zr1 if st == 0 else zr2
        n = zr.shape[0]
        x1 = np.zeros((n, k))
        J = np.argmax(zr.H, axis=0)
        for i in range(0, n):
            x1[i, J[0, i]] = 1

        if st == 1:
            xx1 = a*x1
        else:
            xx2 = a*x1

    n1 = norm_diff(xx1, zr1, np.eye(zr1.shape[1]))
    n2 = norm_diff(xx2, zr2, np.eye(zr2.shape[1]))
    x = xx2 if n1 > n2 else xx1
    return [x, rn]


def find_hard_clusters(z, r, a):
    zr1 = z.dot(r)
    k = zr1.shape[1]
    zr2, rn = flip_negative_columns(zr1)
    for st in range(0, 2):
        zr = zr1 if st == 0 else zr2
        n = zr.shape[0]
        x1 = np.zeros((n, k))
        J = np.argmax(zr.H, axis=0)
        for i in range(0, n):
            x1[i, J[0, i]] = 1

        x1_sum = np.sum(x1, axis=0)
        row_idx = np.argmax(x1, axis=0)
        sum_idx = np.argmax(x1_sum, axis=0)
        for j in range(0, int(k)):
            if x1_sum[j] == 0:
                x1[row_idx[sum_idx]][sum_idx] = 0
                x1[row_idx[sum_idx]][j] = 1
                x1_sum = sum(x1)
                row_idx = np.argmax(x1, axis=0)
                sum_idx = np.argmax(x1_sum, axis=0)
        if st == 0:
            xx1 = a*x1
        else:
            xx2 = a*x1

    n1 = norm_diff(xx1, zr1, np.eye(zr1.shape[1]))
    n2 = norm_diff(xx2, zr2, np.eye(zr2.shape[1]))
    x = xx2 if n1 > n2 else xx1
    return [x, rn]


def sncut(w, n_clusters=4, threshold=1e-14, max_iters=50, fast=True, method='hard', r_method=1):
    if method == 'hard':
        f = find_hard_clusters
    elif method == 'soft':
        f = find_soft_clusters
    else:
        f = find_flex_clusters

    n_vert = w.shape[0]

    # Compute degree matrix
    deg_sums = np.absolute(w).sum(axis=0)
    deg_mat = np.diagflat(deg_sums)
    deg_sums_sqrt = np.sqrt(deg_sums)
    deg_sums_sqrt[deg_sums_sqrt == 0] = 1e-14
    deg_inv = np.diagflat(np.reciprocal(deg_sums_sqrt))

    # Compute signed Laplacian
    lap = deg_mat - w

    # Compute signed normalized aplacian
    lap_sym = deg_inv.dot(lap).dot(deg_inv)

    # Initialize U to be vector of the K smallest eigenvalues of l_sym
    if fast:
        u, _, _ = randomized_svd(2 * np.eye(n_vert) - lap_sym, n_clusters)
    else:
        u, _, _ = np.linalg.svd(2 * np.eye(n_vert) - lap_sym)

    u_k = u[:, 0:n_clusters]

    # Find original relaxed solution Z1
    z1 = deg_inv.dot(u_k)

    # Normalize || Z1 ||_F = 100
    nz = np.sqrt(np.trace(z1.H.dot(z1)))
    z1 *= (100/nz)

    # Find R1 by computing (R1,Sigma,R1^T) = SVD(Z^T * Z)
    _, r1 = np.linalg.eig(z1.H.dot(z1))

    # Compute alternative Z2 = Z1*R1
    z2 = z1.dot(r1)

    # Lambda (scaling value)
    a = 100/np.sqrt(n_vert)

    # Normalize || Z1 ||_F = 100
    nz1 = np.sqrt(np.trace(z1.H.dot(z1)))
    z1 *= (100 / nz1)

    # Normalize || Z2 ||_F = 100
    nz2 = np.sqrt(np.trace(z2.H.dot(z2)))
    z2 *= (100/nz2)

    # Find best initial Rc, Xc given Z, Lambda
    X0, Z, R0 = compute_initial_x_z_r(z1, z2, n_clusters, a, f)
    R0 = find_r(X0, Z)

    # Compute initial phi(Xc, Z, Rc)
    ern = norm_diff(X0, Z, R0)
    er = ern + 1

    X_curr = X0
    R_curr = R0

    for iteration in range(max_iters):
        if ern >= er:
            break

        # Find best Xc given fixed Z, R, Lambda
        X_curr, _ = f(Z, R_curr, a)
        diff_xc = (X_curr - X0)/a
        ndxc = np.trace(diff_xc.conj().T.dot(diff_xc))/2.

        # Halt if threshold is passed
        if ndxc < threshold:
            ern = er
        else:
            # Find best R given fixed Xc, Z
            R_curr = find_r(X_curr, Z) if r_method == 1 else find_a(X_curr, Z)
            er = ern

            # Compute phi(Xc, Z, Rc) with new Rc
            ern = norm_diff(X_curr, Z, R_curr)

            # If (X0, Z, R0) < phi(X', Z, R'), then maintain X = X0, R = R0
            if er < ern:
                X_curr = X0
                R_curr = R0

            # Otherwise, we have found a better estimate
            else:
                X0 = X_curr
                R0 = R_curr

    XX = (1./a) * X_curr
    indices = np.argmax(XX.conj().T, axis=0)
    return [indices.conj().T, XX]


def print_clusters(n_clusters, indices, labels):
    for cluster in range(0, n_clusters):
        print("\n")
        print("Group %d" % cluster)
        for i in range(0, len(indices)):
            if indices[i] == cluster:
                print(labels[i])


if __name__ == "__main__":
    labels = ['Kiana', 'Trang', 'Olivia', 'Sammy', 'Philippe',
              'Ryan', 'Alex', 'Wesley', 'Stefi', 'Shane', 'Harrison',
              'Michael']
    w = np.matrix('\
            0 -0.5 0.9 0.9 0.5 0.4 0.2 0.3 0.1 0.2 0.3 0.4; \
            -0.5 0 -0.8 -0.8 -0.1 0.8 0.1 0.1 0.1 0.1 0.3 0.1; \
            0.9 -0.8 0 0.9 1.0 0.3 0.4 0.4 0.1 0.5 0.2 0.2; \
            0.9 -0.8 0.9 0 0.4 0.7 0.1 0.1 0.1 0.3 0.1 0.4; \
            0.5 -0.1 1.0 0.4 0 0.8 0.5 0.3 -0.1 0.1 0.5 0.1; \
            0.4 0.8 0.3 0.7 0.8 0 -0.2 0.1 0.1 0.1 0.4 0.3; \
            0.2 0.1 0.4 0.1 0.5 -0.2 0 0.5 -0.1 -0.1 0.4 0.1; \
            0.3 0.1 0.4 0.1 0.3 0.1 0.5 0 0.3 0.3 0.3 0.1; \
            0.1 0.1 0.1 0.1 -0.1 0.1 -0.1 0.3 0 0.2 0.1 -0.1; \
            0.2 0.1 0.5 0.3 0.1 0.1 -0.1 0.3 0.2 0 0.3 0.2; \
            0.3 0.3 0.2 0.1 0.5 0.4 0.4 0.3 0.1 0.3 0 -0.2; \
            0.4 0.1 0.2 0.4 0.1 0.3 0.1 0.1 -0.1 0.2 -0.2 0')

    k = 6
    idx, XXc_hard = sncut(w, n_clusters=k, method='hard')
    idx1, XXc_soft = sncut(w, n_clusters=k, method='soft')
    _, XXc_flex = sncut(w, n_clusters=k, method='flex')

    print_clusters(k, idx, labels)
    for cluster in range(0, k):
        print("\n")
        print("Group %d" % cluster)
        print("\t Includes:")
        for i in range(0, len(labels)):
            if idx1[i] == cluster:
                # print("\t\t" + labels[i] + " : " + str(XXc_soft[i, cluster]) + ", " + str(XXc_flex[i, cluster]))
                print("\t\t" + labels[i] + " : " + str(XXc_soft[i, cluster]))
        print("\n\t Excludes:")
        for i in range(0, len(labels)):
            if idx1[i] != cluster:
                print("\t\t" + labels[i] + " : " + str(XXc_soft[i, cluster]))
                # print("\t\t" + labels[i] + " : " + str(XXc_soft[i, cluster]) +", " + str(XXc_flex[i, cluster]))
