from .usage_example import run
import argparse


parser = argparse.ArgumentParser(description='Command-line Utility for training RL models')
parser.add_argument('config', type=int)
args = parser.parse_args()
config = args.config

run(config)
