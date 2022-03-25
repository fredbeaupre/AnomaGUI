import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
plt.style.use('dark_background')


def fix_mistakes(name):
    data = np.load('{}.npz'.format(name))
    fp_files = data['fp_errors']
    fn_files = data['fn_errors']
    classifications = data['classifications']
    print(classifications[173, :])
    print(classifications[365, :])
    classifications[173, 0] = 1
    classifications[365, 0] = 1
    new_fn_files = []
    for f in fn_files:
        print(f)
        print(f == './actin/test/207-0.002.npz' or f ==
              './actin/test/169-0.019.npz')
        if f == './actin/test/207-0.002.npz' or f == './actin/test/169-0.019.npz':
            continue
        else:
            new_fn_files.append(f)
    np.savez('./{}'.format('Flavie_fixed.npz'),
             classifications=classifications,
             fp_errors=fp_files,
             fn_errors=new_fn_files)


def check_errors(name):
    data = np.load('{}.npz'.format(name))
    fp_files = data['fp_errors']
    fn_files = data['fn_errors']
    for i in range(len(fn_files)):
        fig = plt.figure()
        f = fn_files[i]
        img_file = np.load(f)
        quality = f.split('/')[-1].split('-')[-1][:-4]
        img = img_file['arr_0']
        plt.title('Classified as normal, Quality rating = {}'.format(quality))
        plt.xlabel('File name: {}'.format(f))
        plt.imshow(img, cmap='hot')
        plt.show()
        # fig.savefig('./Flavie/false_negatives/FN_{}.png'.format(i))
    for f in fp_files:
        img_file = np.load(f)
        quality = f.split('/')[-1].split('-')[-1][:-4]
        img = img_file['arr_0']
        plt.title('Classified as anomaly, Quality rating = {}'.format(quality))
        plt.xlabel('File name: {}'.format(f))
        plt.imshow(img, cmap='hot')
        plt.show()


def generate_distributions(name):
    data = np.load('{}.npz'.format(name))
    classifications = data['classifications']
    pos_classifications = classifications[classifications[:, 0] == 1]
    neg_classifications = classifications[classifications[:, 0] == 0]
    pos_ratings = pos_classifications[:, 1]
    neg_ratings = neg_classifications[:, 1]
    pos_kde = stats.gaussian_kde(pos_ratings)
    neg_kde = stats.gaussian_kde(neg_ratings)
    x = np.linspace(0, 1.0, 500)
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
    plt.ylim([0, 7])
    plt.legend(fontsize=12)
    plt.show()
    fig.savefig('./{}_qualityrating_distributions.png'.format(name))


def main():
    generate_distributions('./Theresa/Theresa')


if __name__ == "__main__":
    main()
