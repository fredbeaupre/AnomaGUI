import numpy as np


def format_test_paths(paths):
    new_paths = []
    ratings = []
    for p in paths:
        p = p.split('/')[-3:]
        structure = 'actin' if p[0] == 'Actin' else 'tubulin'
        score_str = p[-1][:-4]
        score = float(score_str.split('-')[1])
        ratings.append(score)
        new_p = './{}/test/{}.npz'.format(structure, score_str)
        new_paths.append(new_p)
    return new_paths, ratings


def generate_classifications(paths, ratings, error_files):
    classifications = np.zeros(shape=(len(paths), 2))
    for i in range(len(paths)):
        path = paths[i]
        rat = ratings[i]
        truth = 0 if rat >= 0.70 else 1
        if path in error_files:
            classifications[i, 0] = not truth
        else:
            classifications[i, 0] = truth
        classifications[i, 1] = rat
    return classifications


def main():
    test_paths = np.genfromtxt('../test_paths.csv', dtype=str)
    test_paths, ratings = format_test_paths(test_paths)
    fn_files = np.genfromtxt('./false_negatives_Theresa.csv', dtype=str)
    fp_files = np.genfromtxt('./false_positives_Theresa.csv', dtype=str)
    error_files = np.concatenate([fn_files, fp_files])
    classifications = generate_classifications(
        test_paths, ratings, error_files)
    np.savez('Theresa', classifications=classifications)


if __name__ == "__main__":
    main()
