def project_data(X, U, K):
    """
    project_data Computes the reduced data representation when projecting only
    on to the top k eigenvectors
        project_data(X, U, K) computes the projection of
        the normalized inputs X into the reduced dimensional space spanned by
        the first K columns of U. It returns the projected examples in Z.
    """