from numpy.random import seed
seed(1)
import pickle
import datetime

from phase_2_solution_algorithm.solution import Solution
from Interceptor_V2 import Init, Draw, Game_step
from utils import Plotter


def run(config):

    scores_history = []

    solution = Solution(config)
    for i in range(1000):
        ep_start_time = datetime.datetime.now()

        solution.reset()

        Init()
        observation = Game_step(2)
        # Draw()
        for stp in range(1000):
            # print(str(stp))
            action_button = solution.get_action(observation)
            observation = Game_step(action_button)
            # Draw()

        scores_history.append(observation[4])
        with open('config_0' + str(config) + '.pkl', 'wb') as file:  # .pickle  # wb = write binary
            pickle.dump(scores_history, file)

        print('Episode ' + str(i + 1) + ' - Runtime: %s ' % (str(datetime.datetime.now() - ep_start_time).split('.')[0]))  # score: %.2f

    Plotter.plot_running_average(
        'Interceptor', 'config 0' + str(config), scores_history, window=100, file_name='config_0' + str(config)
    )


if __name__ == '__main__':
    run(config=0)
