import argparse
from .training_op import main_cloud


def parse_args():
    parser = argparse.ArgumentParser(description='Command-line Utility for training RL models')
    parser.add_argument('train_type', type=int)
    parser.add_argument('n_episodes', type=int)
    parser.add_argument('ep_batch_num', type=int)
    parser.add_argument('new_save_folder_mark', type=int)
    parser.add_argument('--load', action='store_true')  # load_checkpoint = False
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    main_cloud(args.train_type, args.n_episodes, args.ep_batch_num, args.new_save_folder_mark, args.load)
