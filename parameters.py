TAN_params = {'oversampling': 'simple',
              'sampling_window': 3,
              'target': 'future',
              'offset': 1,
              'class_mode': 'multi_label',
              'scaler': 'MinMax',
              'filter': 'median',
              'filter_window': 3,
              'train_test_split': 'start',
              'test_size': 30,
              'train_val_split': 'random',
              'val_size': 33,
              'length': 3,
              'stride': 1,
              'sampling_rate': 128,
              'acc_resampler': 'interp_linear',
              'event_resampler': 'preserve',
              'signals': ['Acc_x', 'Acc_y', 'Acc_z', 'Acc', 'Acc_xy', 'Acc_yz', 'Acc_xz'],
              'before_preprocessing': True,
              'positions': ['LF', 'RF'],
              'augmentations': [],
              'batch_size': 20,
              'learning_rate': 0.001,
              'epochs': 50,
              'model': 'Tan',
              'val_scores': ['f1', 'precision', 'recall', 'accuracy'],
              'test_scores': ['f1', 'precision', 'recall', 'accuracy'],
              'val_tables': ['confusion'],
              'test_tables': ['confusion'],
              'average': 'binary'}

ROMI_params = {'oversampling': 'gaussian',
               'sampling_window': 8,
               'subjects': 'all',
               'activities': 'all',
               'target': 'full',
               'offset': 0,
               'class_mode': 'multi_label',
               'scaler': 'Standard',
               'filter': None,
               'filter_window': 1,
               'train_test_split': 'loso',
               'test_size': 33,
               'train_val_split': 'loso',
               'val_size': 33,
               'length': 400,
               'stride': 200,
               'sampling_rate': 200,
               'acc_resampler': 'interp_linear',
               'event_resampler': 'preserve',
               'signals': ['Acc_x', 'Acc_y', 'Acc_z'],
               'before_preprocessing': False,
               'augmentations': [],
               'batch_size': 16,
               'learning_rate': 0.0003,
               'epochs': 10,
               'model': 'Romi',
               'val_scores': ['f1', 'precision', 'recall', 'accuracy'],
               'test_scores': ['f1', 'precision', 'recall', 'accuracy'],
               'val_tables': ['confusion'],
               'test_tables': ['confusion'],
               'average': 'macro'}