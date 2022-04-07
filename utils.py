import numpy as np

OPTIMAL_T = 0.213


def load_paths_and_scores():
    paths = np.genfromtxt('./test_paths.csv', dtype=str)
    scores = np.genfromtxt('./test_scores.csv')
    return paths, scores


def get_rating(path):
    p = path.split('/')[-1].split('-')[-1][:-4]
    return float(p)


def ratings_to_labels(paths):
    N = len(paths)
    labels = np.zeros((N, 3))
    for i in range(N):
        p = paths[i]
        rating = get_rating(p)
        labels[i, 2] = rating
        if rating >= 0.70:
            labels[i, 1] = 0
        else:
            labels[i, 1] = 1
        labels[i, 0] = i
    # sorting
    labels = labels[labels[:, 2].argsort()]
    return labels


def scores_to_preds(scores):
    N = len(scores)
    preds = np.zeros((N, 3))  # (id, pred, score)
    for i in range(N):
        score = scores[i]
        preds[i, 2] = score
        if score >= OPTIMAL_T:
            preds[i, 1] = 1
        else:
            preds[i, 1] = 0
        preds[i, 0] = i
    return preds
