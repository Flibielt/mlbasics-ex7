from .ex7.src import ex7, ex7_pca


def main():
    """
    Machine Learning Class - Exercise 7 | Principle Component Analysis and K-Means Clustering
    """

    # Part 1
    ex7()

    if input('Press ENTER to start the next part. (press [q] to exit here)\n') == 'q':
        print('Exit')
        exit(0)

    # Part 2
    ex7_pca()


if __name__ == '__main__':
    main()
