import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from seaborn import heatmap
import collections
plt.style.use('dark_background')
ANTHONY = './Anthony'
FLAVIE = './Flavie'
THERESA = './Theresa'
CATHERINE = './Catherine'
JULIA = './Julia'
JM = './JM'
ANDREANNE = './Andreanne'


def count_error_occurences():
    # get data
    antho_data = np.load("{}/Anthony.npz".format(ANTHONY))
    flavie_data = np.load("{}/Flavie_fixed.npz".format(FLAVIE))
    theresa_data_2nd = np.load("{}/Theresa_2ndAnnotation.npz".format(THERESA))
    theresa_fns = np.genfromtxt(
        '{}/false_negatives_Theresa.csv'.format(THERESA), dtype=str)
    theresa_fps = np.genfromtxt(
        '{}/false_positives_Theresa.csv'.format(THERESA), dtype=str)

    # get filenames of errors
    antho_fps = antho_data['fp_errors']
    antho_fns = antho_data['fn_errors']
    antho_errors = np.concatenate([antho_fps, antho_fns])
    flavie_fps = flavie_data['fp_errors']
    flavie_fns = flavie_data['fn_errors']
    theresa_fps_2nd = theresa_data_2nd['fp_errors']
    theresa_fns_2nd = theresa_data_2nd['fn_errors']

    flavie_errors = np.concatenate([flavie_fps, flavie_fns])
    theresa_errors = np.concatenate([theresa_fps, theresa_fns])
    theresa_errors_2nd = np.concatenate([theresa_fps_2nd, theresa_fns_2nd])
    # all_errors = np.concatenate([flavie_errors, antho_errors, theresa_errors])
    # this is for checking theresa against theresa
    all_theresa_errors = np.concatenate([theresa_errors, theresa_errors_2nd])
    unique, counts = np.unique(all_theresa_errors, return_counts=True)
    errors_dict = dict(zip(unique, counts))
    once = 0
    twice = 0
    thrice = 0
    index = 0
    for key, value in errors_dict.items():
        if value == 1:
            once += 1
        elif value == 2:
            twice += 1
        elif value == 3:
            thrice += 1
    print("Out of {} errors".format(all_theresa_errors.shape[0]))
    print("\tNumber of errors occuring once: {}".format(once))
    print("\tNumber of errors occuring twice: {}".format(twice))
    print("\tNumber of errors occuring thrice: {}".format(thrice))


def get_QR_labels():
    paths = np.genfromtxt('./test_paths.csv', dtype=str)
    qr_labels = []
    for p in paths:
        p = p.split('/')[-1].split('-')[1][:-4]
        rat = float(p)
        label = 0 if rat >= 0.70 else 1
        qr_labels.append(label)
    return qr_labels


def get_no_agreement_files(data1, data2):
    lower_scores = []
    upper_scores = []
    classifications1 = data1['classifications']
    classifications2 = data2['classifications']
    for i in range(classifications1.shape[0]):
        label1 = classifications1[i, 0]
        label2 = classifications2[i, 0]
        if label1 != label2:
            score = classifications1[i, 1]
            if score >= 0.70:
                upper_scores.append(score)
            else:
                lower_scores.append(score)
    upper_scores = np.array(upper_scores)
    lower_scores = np.array(lower_scores)
    return np.mean(upper_scores), np.mean(lower_scores)


def generate_conf_mat(labels1, labels2, truth, predictor):
    # Note labels1 are considered the truth
    conf_mat = confusion_matrix(y_true=labels1, y_pred=labels2)
    fig = plt.figure()
    heatmap(conf_mat, annot=True, fmt='g', cmap='Reds')
    plt.yticks([0.5, 1.5], ['Normal', 'Anomaly'])
    plt.xticks([0.5, 1.5], ['Normal', 'Anomaly'])
    plt.title('Confusion Matrix')
    plt.xlabel('{}'.format(predictor))
    plt.ylabel('{}'.format(truth))
    plt.show()
    fig.savefig('./confusion_matrices/{}-{}.png'.format(truth, predictor))


def main():

    # count_error_occurences()

    qr_labels = get_QR_labels()
    julia_data = np.load('{}/Julia.npz'.format(JULIA))
    cabou_data = np.load('{}/Catherine.npz'.format(CATHERINE))
    antho_data = np.load("{}/Anthony.npz".format(ANTHONY))
    flavie_data = np.load("{}/Flavie_fixed.npz".format(FLAVIE))
    theresa_data = np.load("{}/Theresa.npz".format(THERESA))
    andreanne_data = np.load("{}/Andreanne.npz".format(ANDREANNE))
    jm_data = np.load("{}/Jean-Mich-Mush.npz".format(JM))
    jm_labels = jm_data['classifications']
    julia_labels = julia_data['classifications']
    theresa_labels = theresa_data['classifications']
    flavie_labels = flavie_data['classifications']
    antho_labels = antho_data['classifications']
    cabou_labels = cabou_data['classifications']
    andreanne_labels = andreanne_data['classifications']
    julia_labels = julia_labels[:, 0]
    flavie_labels = flavie_labels[:, 0]
    antho_labels = antho_labels[:, 0]
    theresa_labels = theresa_labels[:, 0]
    cabou_labels = cabou_labels[:, 0]
    jm_labels = jm_labels[:, 0]
    andreanne_labels = andreanne_labels[:, 0]

    theresa_data_2nd = np.load("{}/Theresa_2ndAnnotation.npz".format(THERESA))
    HQ_mean_score, LQ_mean_score = get_no_agreement_files(
        theresa_data, theresa_data_2nd)
    theresa_labels_2nd = theresa_data_2nd['classifications']
    theresa_labels_2nd = theresa_labels_2nd[:, 0]

    # # computing confusion matrices
    # generate_conf_mat(theresa_labels, theresa_labels_2nd,
    #                   'Theresa-1', 'Theresa-2')
    # generate_conf_mat(flavie_labels, antho_labels, 'Flavie', 'Anthony')
    # generate_conf_mat(flavie_labels, theresa_labels, 'Flavie', 'Theresa')
    # generate_conf_mat(antho_labels, theresa_labels, 'Anthony', 'Theresa')
    # generate_conf_mat(theresa_labels, antho_labels, 'Theresa', 'Anthony')
    # generate_conf_mat(qr_labels, flavie_labels, 'QR', 'Flavie')
    # generate_conf_mat(qr_labels, antho_labels, 'QR', 'Anthony')
    # generate_conf_mat(qr_labels, theresa_labels, 'QR', 'Theresa')

    # Computing inter-expert agreement
    kappa_theresa_squared = cohen_kappa_score(
        theresa_labels, theresa_labels_2nd)
    kappa_flavie_antho = cohen_kappa_score(
        flavie_labels, antho_labels)
    kappa_flavie_theresa = cohen_kappa_score(flavie_labels, theresa_labels)
    kappa_theresa_antho = cohen_kappa_score(theresa_labels, antho_labels)
    kappa_catherine_flavie = cohen_kappa_score(flavie_labels, cabou_labels)
    kappa_catherine_antho = cohen_kappa_score(cabou_labels, antho_labels)
    kappa_catherine_theresa = cohen_kappa_score(cabou_labels, theresa_labels)
    kappa_julia_antho = cohen_kappa_score(julia_labels, antho_labels)
    kappa_julia_theresa = cohen_kappa_score(julia_labels, theresa_labels)
    kappa_julia_cabou = cohen_kappa_score(julia_labels, cabou_labels)
    kappa_julia_flavie = cohen_kappa_score(julia_labels, flavie_labels)
    kappa_jm_antho = cohen_kappa_score(jm_labels, antho_labels)
    kappa_jm_cabou = cohen_kappa_score(jm_labels, cabou_labels)
    kappa_jm_flavie = cohen_kappa_score(jm_labels, flavie_labels)
    kappa_jm_cabou = cohen_kappa_score(jm_labels, cabou_labels)
    kappa_jm_theresa = cohen_kappa_score(jm_labels, theresa_labels)
    kappa_jm_julia = cohen_kappa_score(jm_labels, julia_labels)
    kappa_and_antho = cohen_kappa_score(andreanne_labels, antho_labels)
    kappa_and_flavie = cohen_kappa_score(andreanne_labels, flavie_labels)
    kappa_and_cabou = cohen_kappa_score(andreanne_labels, cabou_labels)
    kappa_and_theresa = cohen_kappa_score(andreanne_labels, theresa_labels)
    kappa_and_jm = cohen_kappa_score(andreanne_labels, jm_labels)
    kappa_and_julia = cohen_kappa_score(andreanne_labels, julia_labels)
    print("Inter-expert agreement results:\n")
    print("\t Flavie-Anthony: {:.3f}".format(kappa_flavie_antho))
    print("\t Flavie-Theresa: {:.3f}".format(kappa_flavie_theresa))
    print("\t Anthony-Theresa: {:.3f}".format(kappa_theresa_antho))
    print("\t Theresa-Theresa: {:.3f}".format(kappa_theresa_squared))
    print("\t Catherine-Anthony: {:.3f}".format(kappa_catherine_antho))
    print("\t Catherine-Flavie: {:.3f}".format(kappa_catherine_flavie))
    print("\t Catherine-Theresa: {:.3f}".format(kappa_catherine_theresa))
    print("\t Julia-Anthony: {:.3f}".format(kappa_julia_antho))
    print("\t Julia-Catherine: {:.3f}".format(kappa_julia_cabou))
    print("\t Julia-Flavie: {:.3f}".format(kappa_julia_flavie))
    print("\t Julia-Theresa: {:.3f}".format(kappa_julia_theresa))

    print("\t JM-Anthony: {:.3f}".format(kappa_jm_antho))
    print("\t JM-Catherine: {:.3f}".format(kappa_jm_cabou))
    print("\t JM-Flavie: {:.3f}".format(kappa_jm_flavie))
    print("\t JM-Julia: {:.3f}".format(kappa_jm_julia))
    print("\t JM-Theresa: {:.3f}".format(kappa_jm_theresa))

    print("\t Andreanne-Anthony: {:.3f}".format(kappa_and_antho))
    print("\t Andreanne-Catherine: {:.3f}".format(kappa_and_cabou))
    print("\t Andreanne-Flavie: {:.3f}".format(kappa_and_flavie))
    print("\t Andreanne-Julia: {:.3f}".format(kappa_and_julia))
    print("\t Andreanne-Theresa: {:.3f}".format(kappa_and_theresa))
    print("\t Andreanne-JM: {:.3f}".format(kappa_and_jm))


    # count_error_occurences()
if __name__ == "__main__":
    main()
