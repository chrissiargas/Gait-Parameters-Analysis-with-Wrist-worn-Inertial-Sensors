import argparse
import yaml


class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="pre-processing and training parameters"
        )

    def __call__(self, *args, **kwargs):
        self.parser.add_argument(
            '--config',
            default='./config.yaml',
            help='config file location'
        )

        self.parser.add_argument(
            '--data_args',
            default=dict(),
            type=dict,
            help='pre-processing arguments'
        )

        self.parser.add_argument(
            '--train_args',
            default=dict(),
            type=dict,
            help='training arguments'
        )

    def get_args(self):
        self.__call__()
        args = self.parser.parse_args(args=[])
        configFile = args.config

        assert configFile is not None

        with open(configFile, 'r') as cf:
            defaultArgs = yaml.load(cf, Loader=yaml.FullLoader)

        keys = vars(args).keys()

        for defaultKey in defaultArgs.keys():
            if defaultKey not in keys:
                print('WRONG ARG: {}'.format(defaultKey))
                assert (defaultKey in keys)

        self.parser.set_defaults(**defaultArgs)
        args = self.parser.parse_args(args=[])

        self.dataset = args.data_args['dataset']
        self.path = args.data_args['path']
        self.length = args.train_args['length']
        self.stride = args.train_args['stride']
        self.sampling_rate = args.train_args['sampling_rate']
        self.signals = args.train_args['signals']
        self.positions = args.train_args['positions']
        self.augmentations = args.train_args['augmentations']
        self.class_mode = args.train_args['class_mode']
        self.events = args.train_args['events']
        self.subjects = args.train_args['subjects']
        self.activities = args.train_args['activities']
        self.split_walk_run = args.train_args['split_walk_run']
        self.oversampling = args.train_args['oversampling']
        self.filter = args.train_args['filter']
        self.filter_window = args.train_args['filter_window']
        self.scaler = args.train_args['scaler']
        self.train_test_split = args.train_args['train_test_split']
        self.train_val_split = args.train_args['train_val_split']
        self.test_size = args.train_args['test_size']
        self.val_size = args.train_args['val_size']
        self.batch_size = args.train_args['batch_size']
        self.target = args.train_args['target']
        self.epochs = args.train_args['epochs']
        self.learning_rate = args.train_args['learning_rate']
        self.model = args.train_args['model']
        self.val_scores = args.train_args['val_scores']
        self.val_tables = args.train_args['val_tables']
        self.test_scores = args.train_args['test_scores']
        self.test_tables = args.train_args['test_tables']
        self.load = args.train_args['load']
        self.bf_preprocessing = args.train_args['before_preprocessing']

        return
