import keras.models as models
import keras.layers as layers

import utils


fc_layers_dims_fire_ang_assessment = [512, 512, 512]
fc_layers_dims_collision_assessment = [512, 512, 512]
fc_layers_dims_collision_time_steps_assessment = [512, 512, 512]
fc_layers_dims_interception_assessment = [512, 512, 512]
fc_layers_dims_interception_time_steps_assessment = [512, 512, 512]
fc_layers_dims_pre_fire_interception_assessment = [512, 512, 512]
fc_layers_dims_pre_fire_interception_time_steps_assessment = [512, 512, 512]


class NetworksBuilder:

    @staticmethod
    def build_fire_ang_assessment_network(optimizer_type=None, ALPHA=None):
        """
        input - 1 rocket data (x,y,dx,dy,dist,ang)
        :return:
        """
        rocket = layers.Input(shape=(6,), dtype='float32', name='rocket')

        x = layers.Dense(fc_layers_dims_fire_ang_assessment[0], activation='relu')(rocket)
        x = layers.Dense(fc_layers_dims_fire_ang_assessment[1], activation='relu')(x)
        x = layers.Dense(fc_layers_dims_fire_ang_assessment[2], activation='relu')(x)
        x = layers.Dense(1, activation='tanh', name='fire_ang_assessment_scaled')(x)
        x = layers.Lambda(lambda v: v * 84, dtype='float32', name='fire_ang_assessment')(x)

        fire_ang_assessment = models.Model(inputs=rocket, outputs=x)
        if optimizer_type is not None and ALPHA is not None:
            optimizer = utils.Optimizers.keras_get_optimizer(optimizer_type, ALPHA)
            fire_ang_assessment.compile(optimizer, loss='mean_squared_error', metrics=['accuracy'])

        return fire_ang_assessment

    @staticmethod
    def build_collision_assessment_networks(optimizer_type=None, ALPHA=None):
        """
        input - 1 rocket data (x,y,dx,dy), 2 X city data (x_left,x_right)
        :return:
        1) collision_assessment - a single rocket (non-interceptor) collision assessment network
                output - collision classification - binary classification: ground (0), city (1)
        2) collision_time_steps_assessment - a single rocket time-steps to collision assessment network
                output - time-steps to collision
        """
        rocket_cities = layers.Input(shape=(8,), dtype='float32', name='rocket_cities')

        x = layers.Dense(fc_layers_dims_collision_assessment[0], activation='relu')(rocket_cities)
        x = layers.Dense(fc_layers_dims_collision_assessment[1], activation='relu')(x)
        x = layers.Dense(fc_layers_dims_collision_assessment[2], activation='relu')(x)
        x = layers.Dense(2, activation='sigmoid', name='collision_assessment')(x)
        collision_assessment = models.Model(inputs=rocket_cities, outputs=x)
        if optimizer_type is not None and ALPHA is not None:
            optimizer = utils.Optimizers.keras_get_optimizer(optimizer_type, ALPHA)
            collision_assessment.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        x = layers.Dense(fc_layers_dims_collision_time_steps_assessment[0], activation='relu')(rocket_cities)
        x = layers.Dense(fc_layers_dims_collision_time_steps_assessment[1], activation='relu')(x)
        x = layers.Dense(fc_layers_dims_collision_time_steps_assessment[2], activation='relu')(x)
        x = layers.Dense(1, activation='relu', name='collision_time_steps_assessment')(x)
        collision_time_steps_assessment = models.Model(inputs=rocket_cities, outputs=x)
        if optimizer_type is not None and ALPHA is not None:
            optimizer = utils.Optimizers.keras_get_optimizer(optimizer_type, ALPHA)
            collision_time_steps_assessment.compile(optimizer, loss='mean_squared_error', metrics=['accuracy'])

        return collision_assessment, collision_time_steps_assessment

    @staticmethod
    def build_interception_assessment_networks(optimizer_type=None, ALPHA=None):
        """
        input - 1 rocket data (x,y,dx,dy), 1 interceptor data (x,y,dx,dy)
        :return:
        1) interception_assessment - a single rocket interception (rocket-interceptor collision) assessment network
                output - collision classification - binary classification: miss (0), hit (1)
        2) interception_time_steps_assessment - a single rocket time-steps to interception (rocket-interceptor collision) assessment network
                only in case of a successful interception!
                output - time-steps to interception
        """
        rocket_interceptor = layers.Input(shape=(8,), dtype='float32', name='rocket_interceptor')

        x = layers.Dense(fc_layers_dims_interception_assessment[0], activation='relu')(rocket_interceptor)
        x = layers.Dense(fc_layers_dims_interception_assessment[1], activation='relu')(x)
        x = layers.Dense(fc_layers_dims_interception_assessment[2], activation='relu')(x)
        x = layers.Dense(2, activation='sigmoid', name='interception_assessment')(x)
        interception_assessment = models.Model(inputs=rocket_interceptor, outputs=x)
        if optimizer_type is not None and ALPHA is not None:
            optimizer = utils.Optimizers.keras_get_optimizer(optimizer_type, ALPHA)
            interception_assessment.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        x = layers.Dense(fc_layers_dims_interception_time_steps_assessment[0], activation='relu')(rocket_interceptor)
        x = layers.Dense(fc_layers_dims_interception_time_steps_assessment[1], activation='relu')(x)
        x = layers.Dense(fc_layers_dims_interception_time_steps_assessment[2], activation='relu')(x)
        x = layers.Dense(1, activation='relu', name='interception_time_steps_assessment')(x)
        interception_time_steps_assessment = models.Model(inputs=rocket_interceptor, outputs=x)
        if optimizer_type is not None and ALPHA is not None:
            optimizer = utils.Optimizers.keras_get_optimizer(optimizer_type, ALPHA)
            interception_time_steps_assessment.compile(optimizer, loss='mean_squared_error', metrics=['accuracy'])

        return interception_assessment, interception_time_steps_assessment

    @staticmethod
    def build_pre_fire_interception_assessment_networks(optimizer_type=None, ALPHA=None):
        """
        input - 1 rocket data (x,y,dx,dy,dist,ang), fire_ang
        :return:
        1) pre_fire_interception_assessment - a single rocket interception (rocket-interceptor collision) assessment network
                output - collision classification - binary classification: miss (0), hit (1)
        2) pre_fire_interception_time_steps_assessment - a single rocket time-steps to interception (rocket-interceptor collision) assessment network
                only in case of a successful interception!
                output - time-steps to interception
        """
        rocket_pre_fire_interceptor = layers.Input(shape=(7,), dtype='float32', name='rocket_pre_fire_interceptor')

        x = layers.Dense(fc_layers_dims_pre_fire_interception_assessment[0], activation='relu')(rocket_pre_fire_interceptor)
        x = layers.Dense(fc_layers_dims_pre_fire_interception_assessment[1], activation='relu')(x)
        x = layers.Dense(fc_layers_dims_pre_fire_interception_assessment[2], activation='relu')(x)
        x = layers.Dense(2, activation='sigmoid', name='pre_fire_interception_assessment')(x)
        pre_fire_interception_assessment = models.Model(inputs=rocket_pre_fire_interceptor, outputs=x)
        if optimizer_type is not None and ALPHA is not None:
            optimizer = utils.Optimizers.keras_get_optimizer(optimizer_type, ALPHA)
            pre_fire_interception_assessment.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        x = layers.Dense(fc_layers_dims_pre_fire_interception_time_steps_assessment[0], activation='relu')(rocket_pre_fire_interceptor)
        x = layers.Dense(fc_layers_dims_pre_fire_interception_time_steps_assessment[1], activation='relu')(x)
        x = layers.Dense(fc_layers_dims_pre_fire_interception_time_steps_assessment[2], activation='relu')(x)
        x = layers.Dense(1, activation='relu', name='pre_fire_interception_time_steps_assessment')(x)
        pre_fire_interception_time_steps_assessment = models.Model(inputs=rocket_pre_fire_interceptor, outputs=x)
        if optimizer_type is not None and ALPHA is not None:
            optimizer = utils.Optimizers.keras_get_optimizer(optimizer_type, ALPHA)
            pre_fire_interception_time_steps_assessment.compile(optimizer, loss='mean_squared_error', metrics=['accuracy'])

        return pre_fire_interception_assessment, pre_fire_interception_time_steps_assessment
