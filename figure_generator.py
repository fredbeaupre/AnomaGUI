import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
plt.style.use('dark_background')


def generate_distributions(name):
    data = np.load('{}.npz'.format(name))
    classifications = data['classifications']
    pos_classifications = classifications[classifications[:, 0] == 1]
    neg_classifications = classifications[classifications[:, 0] == 0]
    pos_ratings = pos_classifications[:, 1]
    neg_ratings = neg_classifications[:, 1]
    pos_kde = stats.gaussian_kde(pos_ratings)
    neg_kde = stats.gaussian_kde(neg_ratings)
    x = np.linspace(0, 1.0, 1000)
    pos_dens = pos_kde.evaluate(x)
    neg_dens = neg_kde.evaluate(x)
    fig = plt.figure()
    plt.plot(x, pos_dens, color='lightblue',
             label='Positive classifications (anom)')
    plt.fill_between(x, pos_dens, facecolor='lightblue',
                     color='lightblue', alpha=0.8)
    plt.plot(x, neg_dens, color='lightcoral',
             label='Negative classifications (normal)')
    plt.fill_between(x, neg_dens, 0, facecolor='lightcoral',
                     color='lightcoral', alpha=0.8)
    plt.axvspan(0.6, 0.7, facecolor='black',
                label='Removed from train/test data', edgecolor='white', hatch='/')
    plt.xlabel('Image quality rating', fontsize=12)
    plt.ylabel('Density (# of points)', fontsize=12)
    plt.xlim([0, 1.0])
    plt.legend(fontsize=12)
    plt.show()
    fig.savefig('./{}_qualityrating_distributions.png'.format(name))


def main():
    generate_distributions('Fred')


if __name__ == "__main__":
    main()
