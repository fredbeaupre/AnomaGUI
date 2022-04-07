
import numpy as np
import matplotlib.pyplot as plt

OPTIMAL_T = 0.213


def sort_classifications_by_qr(data, image_indices=None, index=1, keep_indices=False):
    # quality ratings are in index 1
    if keep_indices:
        new_data = np.zeros((data.shape[0], data.shape[1]+1))
        new_data[:, :-1] = data
        new_data[:, -1] = image_indices
        sorted_data = new_data[new_data[:, index].argsort()]
        return sorted_data
    else:
        sorted_data = data[data[:, index].argsort()]
        return sorted_data


def load_paths_and_scores():
    paths = np.genfromtxt('../test_paths.csv', dtype=str)
    scores = np.genfromtxt('../test_scores.csv')
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


def compute_regret(labels, preds):
    N = labels.shape[0]
    assert N == preds.shape[0]
    regret = np.zeros((N, ))
    for i in range(N):
        label = labels[i, 1]
        pred = preds[i, 1]
        if label != pred:
            regret[i] = 1
        else:
            regret[i] = 0
    return np.cumsum(regret)


def compute_expert_reget(labels, preds):
    N = labels.shape[0]
    assert N == preds.shape[0]
    regret = np.zeros((N, ))
    for i in range(N):
        label = labels[i, 1]
        pred = preds[i]
        if label != pred:
            regret[i] = 1
        else:
            regret[i] = 0
    return np.cumsum(regret)


def get_bin_errors(bin_labels, bin_preds):
    num_errors = 0
    for label, pred in zip(bin_labels, bin_preds):
        if label != pred:
            num_errors += 1
    return num_errors


def generate_model_curves(labels, preds):
    regret = compute_regret(labels, preds)
    expert1_regret = get_expert_regret(labels, '../Flavie/Flavie_fixed.npz')
    julia_regret = get_expert_regret(labels, '../Julia/Julia.npz')
    rankings = np.arange(0, 549, 1)
    rankings_low = np.arange(0, 228, 1)
    rankings_high = np.arange(228, 549, 1)

    bin_size = 10
    w = 1
    N = labels.shape[0]
    assert N == preds.shape[0]
    index = 0
    error_bins = np.zeros((N - bin_size,))
    for i in range(10, N):
        bin_labels = labels[i - bin_size:i]
        bin_preds = preds[i - bin_size:i]
        num_errors = get_bin_errors(bin_labels[:, 1], bin_preds[:, 1])
        error_bins[index] = num_errors
        index += 1

    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axs[0].bar(np.arange(0, 539, 1), error_bins, color='lightcoral', width=2)
    axs[0].set_ylabel('Number of errors\n per 10-image bin')
    axs[0].axvline(x=228.5, ymin=0, ymax=1, color='black',
                   ls='--', label='0.6 - 0.7 gap', lw=1)
    axs[0].set_title("Model's performance")
    axs[0].legend(loc='upper right')
    axs[1].plot(rankings, regret, color='lightcoral', lw=2, label='Model')
    axs[1].plot(rankings, expert1_regret,
                color='steelblue', lw=2, label='Expert 1')
    axs[1].plot(rankings, julia_regret,
                color='green', lw=2, label='Expert 6')
    axs[1].axvline(x=228.5, ymin=0, ymax=1, color='black',
                   ls='--', label='0.6 - 0.7 gap', lw=1)
    axs[1].legend(loc='lower left')
    axs[1].set_ylabel('Cumulative\nregret')
    axs[1].set_xlabel('Images ranked by quality rating')
    fig.savefig('./model_10folderrors_regret.pdf')


def get_expert_regret(labels, file_name):
    data = np.load(file_name)
    classifications = data['classifications']
    sorted_data = sort_classifications_by_qr(classifications)
    sorted_preds = sorted_data[:, 0]
    regret = compute_expert_reget(labels, sorted_preds)
    return regret


def main():
    paths, scores = load_paths_and_scores()
    sorted_labels = ratings_to_labels(paths)
    sorted_ids = sorted_labels[:, 0]
    sorted_ids = [int(el) for el in sorted_ids]
    preds = scores_to_preds(scores)
    sorted_preds = preds[sorted_ids]
    generate_model_curves(sorted_labels, sorted_preds)


if __name__ == "__main__":
    main()
