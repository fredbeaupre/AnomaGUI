import numpy as np
import matplotlib.pyplot as plt
from build_testpaths import get_paths

THERESA = './THERESA/Theresa.npz'
THERESA_2 = './THERESA/Theresa_2ndAnnotation.npz'
ANTHONY = './ANTHONY/Anthony.npz'
FLAVIE = './FLAVIE/Flavie_fixed.npz'
JM = './JM/Jean-Mich-Mush.npz'
CATHERINE = './Catherine/Catherine.npz'
JULIA = './Julia/Julia.npz'
ANDREANNE = 'Andreanne/Andreanne.npz'

FILENAMES = [THERESA, CATHERINE, ANTHONY, FLAVIE, JM, JULIA, ANDREANNE]

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


def compute_regret(data, start, truth):
    regret = start
    cum_regret = []
    labels = data[:, 0]
    N = labels.shape[0]
    for i in range(N):
        if labels[i] != truth:
            regret += 1
        cum_regret.append(regret)
    return cum_regret


def compute_variance(ary):
    variance = []
    N = ary.shape[0]
    for i in range(10, N):
        sub_ary = ary[i - 10: i]
        var = np.std(sub_ary)
        variance.append(var)
    return variance


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
    rankings = np.arange(0, len(errors), 1)
    rankings_low = np.arange(0, 228, 1)
    rankings_high = np.arange(228, 549, 1)
    data = np.load(SINGLE_EXPERT)
    data = data['classifications']
    sorted_data = sort_classifications_by_qr(data)
    low_qual, high_qual = split_qualities(data)
    sorted_low = sort_classifications_by_qr(low_qual)
    sorted_high = sort_classifications_by_qr(high_qual)
    regret_low = compute_regret(sorted_low, start=0, truth=1)
    regret_high = compute_regret(sorted_high, start=regret_low[-1], truth=0)
    connecting_x = [0.6, 0.7]
    connecting_y = [regret_low[-1], regret_low[-1]]

    fig, axs = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axs[0].plot(rankings, sorted_data[:, 0], color='lightcoral', lw=1)
    axs[0].set_yticks([0, 1])
    axs[0].set_yticklabels(['Normal', 'Anomaly'])
    axs[0].axvline(x=228.5, ymin=0, ymax=1, color='black',
                   ls='--', label='0.6 - 0.7 gap', lw=0.8)
    axs[0].set_title('Expert 1 classifications')
    axs[0].legend()
    axs[1].plot(rankings_low, regret_low, color='lightcoral', lw=3)
    axs[1].plot(rankings_high, regret_high, color='lightcoral', lw=3)
    # axs[1].plot(connecting_x, connecting_y, color='lightcoral', ls='--')
    axs[1].axvline(x=228.5, ymin=0, ymax=1, color='black',
                   ls='--', label='0.6 - 0.7 gap', lw=0.8)
    axs[1].legend()
    axs[1].set_ylabel('Cumulative\nregret')
    axs[1].set_title('Expert 1 regret')
    axs[2].bar(rankings, errors, color='steelblue')
    axs[2].set_xlabel('Images ranked by quality rating')
    axs[2].set_ylabel('Expert errors\nper image')
    axs[2].set_ylim([0, 7])
    axs[2].set_title('All 7 experts errors')
    fig.savefig('./Flavie/classifications_regret_all_errors.pdf')


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


def main():
    sorted_errors, img_ids_to_check = get_total_errors()
    check_ambiguous_imgs(img_ids_to_check)
    # single_expert_curves(sorted_errors)


if __name__ == "__main__":
    main()
