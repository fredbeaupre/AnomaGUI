import numpy as np
import matplotlib.pyplot as plt
from build_testpaths import get_paths
from utils import *

THERESA = './Theresa/Theresa.npz'
THERESA_2 = './THERESA/Theresa_2ndAnnotation.npz'
ANTHONY = './ANTHONY/Anthony.npz'
FLAVIE = './FLAVIE/Flavie_fixed.npz'
JM = './JM/Jean-Mich-Mush.npz'
CATHERINE = './Catherine/Catherine.npz'
JULIA = './Julia/Julia.npz'
ANDREANNE = './Andreanne/Andreanne.npz'

FILENAMES = [FLAVIE, ANTHONY, THERESA, JM, CATHERINE, JULIA, ANDREANNE]


SINGLE_EXPERT = FLAVIE


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def get_QR_labels():
    paths = np.genfromtxt('./test_paths.csv', dtype=str)
    qr_labels = []
    for p in paths:
        p = p.split('/')[-1].split('-')[1][:-4]
        rat = float(p)
        label = 0 if rat >= 0.70 else 1
        qr_labels.append(label)
    return qr_labels


def get_errors_per_image():
    truth = get_QR_labels()
    npz_files = [np.load(fname) for fname in FILENAMES]
    experts = [data['classifications'] for data in npz_files]

    num_errors = np.zeros(shape=(549, 2))
    N = len(truth)
    for i in range(N):
        true_label = truth[i]
        curr_labels = [data[i, 0] for data in experts]
        errors = curr_labels.count(not true_label)
        num_errors[i, 0] = errors
        num_errors[i, 1] = i
    return num_errors


def split_qualities(data):
    low_qual = data[data[:, 1] <= 0.60]
    high_qual = data[data[:, 1] >= 0.70]
    return low_qual, high_qual


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


def generate_ranking_curves():
    num_subplots = len(FILENAMES)
    fig, axs = plt.subplots(num_subplots, 1, figsize=(
        12, 7), sharex='col')
    for i in range(num_subplots):
        fname = FILENAMES[i]
        data = np.load(fname)
        data = data['classifications']
        sorted_data = sort_classifications_by_qr(data)
        axs[i].plot(np.arange(0, 549, 1), sorted_data[:, 0],
                    color='lightcoral', lw=1)
        axs[i].set_title('{}'.format(fname.split('/')[1]))
        axs[i].set_ylabel('Class')
    axs[-1].set_xlabel('Images ranked by ascending QR')
    plt.tight_layout()
    plt.show()
    fig.savefig('./classifications-rankings_graph.pdf')


def generate_curves():
    num_subplots = len(FILENAMES)
    rankings_low = np.arange(0, 228, 1)
    rankings_high = np.arange(228, 549, 1)
    fig, axs = plt.subplots(num_subplots, 2, figsize=(
        12, 7), sharex='col')
    for i in range(num_subplots):
        fname = FILENAMES[i]
        data = np.load(fname)
        data = data['classifications']
        low_qual, high_qual = split_qualities(data)
        sorted_low = sort_classifications_by_qr(low_qual)
        sorted_high = sort_classifications_by_qr(high_qual)
        classifications_low = sorted_low[:, 0]
        classifications_high = sorted_high[:, 0]
        print(sorted_low.shape)
        axs[i][0].plot(rankings_low, sorted_low[:, 0],
                       color='lightcoral', lw=1)
        axs[i][0].set_title('{}'.format(fname.split('/')[1]))
        axs[i][0].set_ylabel('Class')
        axs[i][1].plot(rankings_high, sorted_high[:, 0],
                       color='lightcoral', lw=1)
        axs[i][1].set_title('{}'.format(fname.split('/')[1]))
        axs[i][1].set_ylabel('Class')
    axs[-1][0].set_xlabel('Quality rating')
    axs[-1][1].set_xlabel('Quality rating')
    plt.tight_layout()
    plt.show()
    fig.savefig('./classifications_rankings_graph.pdf')
    plt.clf()

    fig, axs = plt.subplots(num_subplots, 1, figsize=(
        7, 7), sharex='col')
    for i in range(num_subplots):
        fname = FILENAMES[i]
        data = np.load(fname)
        data = data['classifications']
        low_qual, high_qual = split_qualities(data)
        sorted_low = sort_classifications_by_qr(low_qual)
        sorted_high = sort_classifications_by_qr(high_qual)
        regret_low = compute_regret(sorted_low, start=0, truth=1)
        regret_high = compute_regret(
            sorted_high, start=regret_low[-1], truth=0)
        connecting_x = [0.6, 0.7]
        connecting_y = [regret_low[-1], regret_low[-1]]
        axs[i].plot(rankings_low, regret_low, color='lightcoral', lw=1)
        axs[i].plot(rankings_high, regret_high, color='lightcoral', lw=1)
        axs[i].plot(connecting_x, connecting_y,
                    color='lightcoral', ls='--', lw=1)
        axs[i].set_title('{}'.format(fname.split('/')[1]))
        axs[i].set_ylabel('Cumulative\nregret')
        axs[i].set_ylim([-10, 230])
    axs[-1].set_xlabel('Quality rating')
    plt.tight_layout()
    plt.show()
    fig.savefig('./regret_graph.pdf')


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


def get_expert_regret(labels, file_name):
    data = np.load(file_name)
    classifications = data['classifications']
    sorted_data = sort_classifications_by_qr(classifications)
    sorted_preds = sorted_data[:, 0]
    regret = compute_expert_reget(labels, sorted_preds)
    return regret


def average_classification():
    antho_data = np.load(ANTHONY)
    quality_ratings = antho_data['classifications'][:, 1]

    julia_data = np.load(JULIA)
    julia_data = julia_data['classifications'][:, 0]
    cabou_data = np.load(CATHERINE)
    cabou_data = cabou_data['classifications'][:, 0]
    antho_data = antho_data['classifications'][:, 0]
    theresa_1_data = np.load(THERESA)
    theresa_1_data = theresa_1_data['classifications'][:, 0]
    theresa_2_data = np.load(THERESA_2)
    theresa_2_data = theresa_2_data['classifications'][:, 0]
    flavie_data = np.load(FLAVIE)
    flavie_data = flavie_data['classifications'][:, 0]
    jm_data = np.load(JM)
    jm_data = jm_data['classifications'][:, 0]
    all_classifications = np.zeros(shape=(549, 6))
    avg_classifications = np.zeros(shape=(549, 2))
    data_list = [antho_data, theresa_1_data,
                 flavie_data, jm_data, cabou_data, julia_data]
    for i in range(len(data_list)):
        all_classifications[:, i] = data_list[i]
    mean_classifications = np.mean(all_classifications, axis=1)
    avg_classifications[:, 0] = mean_classifications
    avg_classifications[:, 1] = quality_ratings
    avg_classifications = sort_classifications_by_qr(
        avg_classifications, index=-1)

    low_avgs, high_avgs = split_qualities(avg_classifications)

    fig = plt.figure(figsize=(12, 7))
    plt.plot(low_avgs[:, 1],
             low_avgs[:, 0], color='lightcoral', lw=1)
    plt.xlabel('Image quality rating')
    plt.ylabel('Average\nclassification')
    plt.show()
    fig.savefig('./low_avg_classification.pdf')

    plt.clf()
    fig = plt.figure(figsize=(12, 7))
    plt.plot(high_avgs[:, 1],
             high_avgs[:, 0], color='lightcoral', lw=1)
    plt.xlabel('Image quality rating')
    plt.ylabel('Average\nclassification')
    plt.show()
    fig.savefig('./high_avg_classification.pdf')


def get_total_errors(single_expert=SINGLE_EXPERT):
    errors = get_errors_per_image()
    data = np.load(SINGLE_EXPERT)
    classifications = data['classifications']
    classifications = sort_classifications_by_qr(
        classifications, image_indices=errors[:, 1], keep_indices=True)
    N = classifications.shape[0]
    sorted_errors = []
    img_ids_to_check = []
    for i in range(N):
        image_id = classifications[i, -1]
        row = errors[errors[:, 1] == image_id]
        err_count = row[0][0]
        img_id = row[0][1]
        if err_count >= 4:
            img_ids_to_check.append(img_id)
        sorted_errors.append(err_count)
    return sorted_errors, img_ids_to_check


def single_expert_curves(errors, single_expert=SINGLE_EXPERT):
    bin_size = 10
    w = 1
    paths, scores = load_paths_and_scores()
    sorted_raw_labels = ratings_to_labels(paths)
    sorted_ids = sorted_raw_labels[:, 0]
    sorted_labels = sorted_raw_labels[:, 1]
    sorted_ids = [int(el) for el in sorted_ids]
    data = np.load(SINGLE_EXPERT)
    preds = data['classifications']
    sorted_preds = sort_classifications_by_qr(preds)
    sorted_preds = sorted_preds[:, 0]
    N = sorted_labels.shape[0]
    assert N == sorted_preds.shape[0]
    index = 0
    error_bins = np.zeros((N - bin_size,))
    for i in range(10, N):
        bin_labels = sorted_labels[i-bin_size:i]
        bin_preds = sorted_preds[i-bin_size:i]
        num_errors = get_bin_errors(bin_labels, bin_preds)
        error_bins[index] = num_errors
        index += 1

    model_preds = scores_to_preds(scores)
    sorted_model_preds = model_preds[sorted_ids]
    model_error_bins = np.zeros((N - bin_size,))
    model_index = 0
    for i in range(10, N):
        model_bin_labels = sorted_labels[i - bin_size:i]
        model_bin_preds = sorted_model_preds[i - bin_size:i]
        num_errors = get_bin_errors(
            model_bin_labels, model_bin_preds[:, 1])
        model_error_bins[model_index] = num_errors
        model_index += 1

    fig, axs = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axs[0].bar(np.arange(0, 539, 1), model_error_bins,
               color='steelblue', width=w)
    axs[0].axvline(x=228.5, ymin=0, ymax=1, color='black',
                   ls='--', label='0.6 - 0.7 gap', lw=0.8)
    axs[0].set_title('Model')
    axs[0].set_ylabel('Number of errors\n per 10-image bin')
    axs[0].legend()

    # REGRET
    regret = compute_regret(sorted_raw_labels, sorted_model_preds)
    expert1_regret = get_expert_regret(
        sorted_raw_labels, './Flavie/Flavie_fixed.npz')
    julia_regret = get_expert_regret(sorted_raw_labels, './Julia/Julia.npz')
    rankings = np.arange(0, 549, 1)
    axs[2].plot(rankings, regret, color='steelblue', lw=2, label='Model')
    axs[2].plot(rankings, expert1_regret,
                color='lightcoral', lw=2, label='Expert 1')
    axs[2].plot(rankings, julia_regret, color='green', lw=2, label='Expert 6')
    axs[2].axvline(x=228.5, ymin=0, ymax=1, color='black',
                   ls='--', label='0.6 - 0.7 gap', lw=0.8)
    axs[2].legend()
    axs[2].set_ylabel('Cumulative\nregret')

    axs[1].bar(np.arange(0, 539, 1), error_bins, color='lightcoral', width=w)
    axs[1].set_xlabel('Images ranked by quality rating')
    axs[1].set_ylabel('Number of errors\n per 10-image bin')
    axs[1].axvline(x=228.5, ymin=0, ymax=1, color='black',
                   ls='--', label='0.6 - 0.7 gap', lw=0.8)
    axs[1].legend(loc='upper left')
    axs[1].set_title('Expert 1')
    fig.savefig('./Flavie/flavie_vs_model_and_regret.pdf')
    # axs[2].bar(rankings, errors, color='steelblue')
    # axs[2].set_xlabel('Images ranked by quality rating')
    # axs[2].set_ylabel('Expert errors\nper image')
    # axs[2].set_ylim([0, 7])
    # axs[2].set_title('All 7 experts errors')
    # fig.savefig('./Flavie/classifications_regret_all_errors.pdf')


def check_ambiguous_imgs(indices):
    paths, _ = get_paths()
    paths = np.take(paths, indices)
    paths_to_check = []
    ids_to_check = np.zeros((len(paths), ))
    forloop_idx = 0
    for p, i in zip(paths, indices):
        idx = int(i)
        img = np.load(p)
        img = img['arr_0']
        paths_to_check.append(p)
        ids_to_check[forloop_idx] = idx
        forloop_idx += 1
    paths_to_check = np.array(paths_to_check)
    print(paths_to_check.shape)
    print(paths_to_check[0])
    np.savez('./imgs_to_check',
             paths=paths_to_check, ids=ids_to_check)


def get_bin_errors(bin_labels, bin_preds):
    num_errors = 0
    for label, pred in zip(bin_labels, bin_preds):
        if label != pred:
            num_errors += 1
    return num_errors


def tenfold_errors(dest='./10fold_errors.pdf', experts=FILENAMES):
    w = 1
    bin_size = 10
    paths, scores = load_paths_and_scores()
    sorted_labels = ratings_to_labels(paths)
    sorted_ids = sorted_labels[:, 0]
    sorted_labels = sorted_labels[:, 1]
    sorted_ids = [int(el) for el in sorted_ids]
    fig, axs = plt.subplots(2, 4, figsize=(13, 7))
    col_id = 0
    row_id = 0
    for j, expert in enumerate(experts):
        data = np.load(expert)
        preds = data['classifications']
        sorted_preds = sort_classifications_by_qr(preds)
        sorted_preds = sorted_preds[:, 0]
        N = sorted_labels.shape[0]
        error_bins = np.zeros((N - bin_size,))
        assert N == sorted_preds.shape[0]
        index = 0
        for i in range(10, N):
            bin_labels = sorted_labels[i - bin_size:i]
            bin_preds = sorted_preds[i-bin_size:i]
            num_errors = get_bin_errors(bin_labels, bin_preds)
            error_bins[index] = num_errors
            index += 1
        axs[row_id][col_id].bar(np.arange(0, 539, 1),
                                error_bins, color='steelblue', width=w)
        axs[row_id][col_id].set_xlabel(
            '10-image bins\nranked by QR')
        axs[row_id][col_id].set_ylabel('Number of errors\n per bin')
        axs[row_id][col_id].set_title('Expert {}'.format(j+1))
        axs[row_id][col_id].axvline(x=228.5, ymin=0, ymax=1, color='black',
                                    ls='--', label='0.6 - 0.7 gap', lw=0.8)
        if row_id == 1:
            row_id = 0
            col_id += 1
        else:
            row_id += 1
    plt.tight_layout()
    plt.show()


def main():
    # tenfold_errors('./Anthony/10fold_errors.pdf')
    sorted_errors, img_ids_to_check = get_total_errors()
    # check_ambiguous_imgs(img_ids_to_check)
    single_expert_curves(sorted_errors)


if __name__ == "__main__":
    main()
