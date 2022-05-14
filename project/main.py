from argparse import Namespace

from train import prepare_trainer
from utils.parse import parse_arguments


args: Namespace = parse_arguments()


def main():
    if args.run_mode == 'train':
        trainer, learner, train_loader, val_loader = prepare_trainer(args.config_file)
        trainer.fit(learner, train_loader, val_loader)

    elif args.run_mode == 'test':
        # test function

    else:
        raise ValueError('run mode must be train or test')


if __name__ == '__main__':
    main()
