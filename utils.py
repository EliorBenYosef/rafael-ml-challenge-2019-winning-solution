import os

import numpy as np
import matplotlib.pyplot as plt
import pickle

import tensorflow as tf
from tensorflow.python.client import device_lib

import tensorflow.python.keras.backend as keras_tensorflow_backend
from tensorflow.python.keras.backend import set_session as keras_set_session
import tensorflow.python.keras.optimizers as optimizers


class Plotter:
    # colors = ['r--', 'g--', 'b--', 'c--', 'm--', 'y--', 'k--', 'w--']

    # colors = ['#FF0000', '#fa3c3c', '#E53729',
    #           '#f08228', '#FB9946', '#FF7F00',
    #           '#e6af2d',
    #           '#e6dc32', '#FFFF00',
    #           '#a0e632', '#00FF00',  '#00dc00',
    #           '#17A858', '#00d28c',
    #           '#00c8c8', '#0DB0DD',  '#00a0ff', '#1e3cff', '#0000FF',
    #           '#6e00dc', '#8B00FF',  '#4B0082', '#a000c8', '#662371',
    #           '#f00082']

    # colors = ['#FF0000', '#E53729',
    #           '#f08228', '#FF7F00',
    #           '#e6af2d',
    #           '#e6dc32', '#FFFF00',
    #           '#a0e632', '#00dc00',
    #           '#17A858', '#00d28c',
    #           '#00c8c8', '#1e3cff',
    #           '#6e00dc', '#a000c8',
    #           '#f00082']

    colors = ['#FF0000',  # '#E53729',
              # '#f08228',
              '#FF7F00',
              # '#e6af2d',
              # '#e6dc32',
              '#FFFF00',
              # '#a0e632',
              '#00dc00',
              '#17A858',  # '#00d28c',
              # '#00c8c8',
              '#1e3cff',
              '#6e00dc',  # '#a000c8',
              '#f00082']

    @staticmethod
    def get_running_avg(scores, window):
        episodes = len(scores)

        if episodes >= window + 50:
            x = [i + 1 for i in range(window - 1, episodes)]

            running_avg = np.empty(episodes - window + 1)
            for t in range(window - 1, episodes):
                running_avg[t - window + 1] = np.mean(scores[(t - window + 1):(t + 1)])

        else:
            x = [i + 1 for i in range(episodes)]

            running_avg = np.empty(episodes)
            for t in range(episodes):
                running_avg[t] = np.mean(scores[max(0, t - window):(t + 1)])

        return x, running_avg

    @staticmethod
    def plot_running_average(env_name, method_name, scores, window=100, show=False, file_name=None, directory='',
                             ylabel=None):
        plt.close()
        plt.title(
            env_name + ' - ' + method_name + (' - Score' if window == 0 else ' - Running Score Avg. (%d)' % window))
        plt.ylabel(ylabel if ylabel else 'Score')
        plt.xlabel('Episode')
        plt.plot(*Plotter.get_running_avg(scores, window))
        if file_name:
            plt.savefig(directory + file_name + '.png')
        if show:
            plt.show()
        plt.close()


class DeviceGetUtils:

    @staticmethod
    def tf_get_local_devices(GPUs_only=False):
        local_devices = device_lib.list_local_devices()  # local_device_protos
        if GPUs_only:
            return [dev.name for dev in local_devices if 'GPU' in dev.device_type]
        else:
            return [dev.name for dev in local_devices]

    @staticmethod
    def keras_get_available_GPUs():  # To Check if keras(>=2.1.1) is using GPU:
        return keras_tensorflow_backend._get_available_gpus()


class DeviceSetUtils:

    @staticmethod
    def set_device(devices_dict=None):  # {type: bus_id}
        if devices_dict is not None:
            designated_GPUs_bus_id_str = ''
            for device_type, device_bus_id in devices_dict.items():
                if len(designated_GPUs_bus_id_str) > 0:
                    designated_GPUs_bus_id_str += ','
                designated_GPUs_bus_id_str += str(device_bus_id)

            DeviceSetUtils.tf_set_device(designated_GPUs_bus_id_str)
            DeviceSetUtils.keras_set_session_according_to_device(devices_dict)

    @staticmethod
    def tf_set_device(designated_GPUs_bus_id_str):  # can be singular: '0', or multiple: '0,1'
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # set GPUs (CUDA devices) IDs' order by pci bus IDs (so it's consistent with nvidia-smi's output).
        os.environ['CUDA_VISIBLE_DEVICES'] = designated_GPUs_bus_id_str  # specify which GPU ID(s) to be used.
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    @staticmethod
    def tf_get_session_according_to_device(devices_dict):
        if devices_dict is not None:
            gpu_options = tf.GPUOptions(allow_growth=True)  # starts with allocating an approximated amount of GPU memory, and expands if necessary
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)  # set the fraction of GPU memory to be allocated
            config = tf.ConfigProto(device_count=devices_dict, gpu_options=gpu_options, log_device_placement=False)  # log device placement tells which device is used.
            # config.gpu_options.allow_growth = True
            # config.gpu_options.per_process_gpu_memory_fraction = 0.5
            sess = tf.Session(config=config)
        else:
            sess = tf.Session()
        return sess

    @staticmethod
    def keras_set_session_according_to_device(devices_dict):
        keras_set_session(DeviceSetUtils.tf_get_session_according_to_device(devices_dict))


class Optimizers:

    OPTIMIZER_Adam = 0
    OPTIMIZER_RMSprop = 1
    OPTIMIZER_Adadelta = 2
    OPTIMIZER_Adagrad = 3
    OPTIMIZER_SGD = 4

    @staticmethod
    def tf_get_optimizer(optimizer_type, lr, momentum=None):  # momentum=0.9
        if optimizer_type == Optimizers.OPTIMIZER_SGD:
            if momentum is None:
                return tf.train.GradientDescentOptimizer(lr)
            else:
                return tf.train.MomentumOptimizer(lr, momentum)
        elif optimizer_type == Optimizers.OPTIMIZER_Adagrad:
            return tf.train.AdagradOptimizer(lr)
        elif optimizer_type == Optimizers.OPTIMIZER_Adadelta:
            return tf.train.AdadeltaOptimizer(lr)
        elif optimizer_type == Optimizers.OPTIMIZER_RMSprop:
            if momentum is None:
                return tf.train.RMSPropOptimizer(lr)
            else:
                return tf.train.RMSPropOptimizer(lr, decay=0.99, momentum=momentum, epsilon=1e-6)
        else:  # optimizer_type == Optimizers.OPTIMIZER_Adam
            return tf.train.AdamOptimizer(lr)

    @staticmethod
    def keras_get_optimizer(optimizer_type, lr, momentum=0., rho=None, epsilon=None, decay=0., beta_1=0.9, beta_2=0.999):
        if optimizer_type == Optimizers.OPTIMIZER_SGD:
            return optimizers.SGD(lr, momentum, decay)  # momentum=0.9
        elif optimizer_type == Optimizers.OPTIMIZER_Adagrad:
            return optimizers.Adagrad(lr, epsilon, decay)
        elif optimizer_type == Optimizers.OPTIMIZER_Adadelta:
            return optimizers.Adadelta(lr, rho if rho is not None else 0.95, epsilon, decay)
        elif optimizer_type == Optimizers.OPTIMIZER_RMSprop:
            return optimizers.RMSprop(lr, rho if rho is not None else 0.9, epsilon, decay)  # momentum= ?
            # return optimizers.RMSprop(lr, rho=0.99, epsilon=0.1)
            # return optimizers.RMSprop(lr, epsilon=1e-6, decay=0.99)
        else:  # optimizer_type == Optimizers.OPTIMIZER_Adam
            return optimizers.Adam(lr, beta_1, beta_2, epsilon, decay)


class SaverLoader:

    @staticmethod
    def pickle_load(file_name, directory=''):
        with open(directory + file_name + '.pkl', 'rb') as file:  # .pickle  # rb = read binary
            var = pickle.load(file)  # var == [X_train, y_train]
        return var

    @staticmethod
    def pickle_save(var, file_name, directory=''):
        with open(directory + file_name + '.pkl', 'wb') as file:  # .pickle  # wb = write binary
            pickle.dump(var, file)  # var == [X_train, y_train]


class General:

    @staticmethod
    def make_sure_dir_exists(dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
