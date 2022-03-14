import numpy as np

actin_test_paths = "./actin/test"
tubulin_test_paths = "./tubulin/test"


def get_paths():
    paths = np.genfromtxt('./test_paths.csv', delimiter=',', dtype=None)
    scores = np.genfromtxt('./test_scores.csv', delimiter=',')
    test_paths = []
    for p in paths:
        p = str(p)
        p = p.lower().split('/')
        structure = p[-3].lower()
        name = p[-1][:-5]
        filename = "./{}/test/{}.npz".format(structure, name)
        test_paths.append(filename)
    return test_paths


def main():
    get_paths()


if __name__ == "__main__":
    main()
