import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument('--config_file', '-c', type=str, default='v1', help='config file name')
    parser.add_argument('--run_mode', '-r', type=str, default='train', help='run mode: train or test')

    args = parser.parse_args()

    return args