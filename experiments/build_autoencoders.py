import sys
sys.path.append('/home/riccardo/Documenti/PhD/ExplainingImageClassifiers/code/')

from experiments.exputil import get_dataset
from experiments.exputil import get_autoencoder

import warnings
warnings.filterwarnings('ignore')


def main():

    dataset = 'cifar10bw'
    ae_name = 'aae'

    epochs = 10000
    batch_size = 256
    sample_interval = 200

    path = '/Users/riccardo/Documents/PhD/ExplainImageClassifier/code/'
    # path = '/home/riccardo/Documenti/PhD/ExplainingImageClassifiers/code/'
    path_aemodels = path + 'aemodels/%s/%s/' % (dataset, ae_name)

    _, _, X_test, Y_test, use_rgb = get_dataset(dataset)
    ae = get_autoencoder(X_test, ae_name, dataset, path_aemodels)

    ae.fit(X_test, epochs=epochs, batch_size=batch_size, sample_interval=sample_interval)
    ae.save_model()
    ae.sample_images(epochs)


if __name__ == "__main__":
    main()
