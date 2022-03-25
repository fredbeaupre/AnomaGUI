import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from seaborn import heatmap
plt.style.use('dark_background')
ANTHONY = './Anthony'
FLAVIE = './Flavie'
THERESA = './Theresa'


def get_QR_labels():
    paths = np.genfromtxt('./test_paths.csv', dtype=str)
    qr_labels = []
    for p in paths:
        p = p.split('/')[-1].split('-')[1][:-4]
        rat = float(p)
        label = 0 if rat >= 0.70 else 1
        qr_labels.append(label)
    return qr_labels


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
    qr_labels = get_QR_labels()
    antho_data = np.load("{}/Anthony.npz".format(ANTHONY))
    flavie_data = np.load("{}/Flavie_fixed.npz".format(FLAVIE))
    theresa_data = np.load("{}/Theresa.npz".format(THERESA))
    theresa_labels = theresa_data['classifications']
    flavie_labels = flavie_data['classifications']
    antho_labels = antho_data['classifications']
    flavie_labels = flavie_labels[:, 0]
    antho_labels = antho_labels[:, 0]
    theresa_labels = theresa_labels[:, 0]

    # computing confusion matrices
    generate_conf_mat(flavie_labels, antho_labels, 'Flavie', 'Anthony')
    generate_conf_mat(flavie_labels, theresa_labels, 'Flavie', 'Theresa')
    generate_conf_mat(antho_labels, theresa_labels, 'Anthony', 'Theresa')
    generate_conf_mat(theresa_labels, antho_labels, 'Theresa', 'Anthony')
    generate_conf_mat(qr_labels, flavie_labels, 'QR', 'Flavie')
    generate_conf_mat(qr_labels, antho_labels, 'QR', 'Anthony')
    generate_conf_mat(qr_labels, theresa_labels, 'QR', 'Theresa')

    # Computing inter-expert agreement
    kappa_flavie_antho = cohen_kappa_score(
        flavie_labels, antho_labels)
    kappa_flavie_theresa = cohen_kappa_score(flavie_labels, theresa_labels)
    kappa_theresa_antho = cohen_kappa_score(theresa_labels, antho_labels)
    print("Inter-expert agreement results:\n")
    print("\t Flavie-Anthony: {:.3f}".format(kappa_flavie_antho))
    print("\t Flavie-Theresa: {:.3f}".format(kappa_flavie_theresa))
    print("\t Anthony-Theresa: {:.3f}".format(kappa_theresa_antho))


if __name__ == "__main__":
    main()
