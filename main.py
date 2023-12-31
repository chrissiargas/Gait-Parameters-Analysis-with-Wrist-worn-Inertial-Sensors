from experiments import Tan, Romi
import warnings
import os
import time
import tensorflow as tf

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

implementations = ['Tan', 'Romi']


def main():
    experiment = 'Tan'
    archive = os.path.join("archive", experiment)

    if experiment == 'Tan':
        Tan(archive)

    elif experiment == 'Romi':
        Romi(archive)


if __name__ == '__main__':
    main()
