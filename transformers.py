from configParser import Parser
from operator import itemgetter
from augment import *


class binary:
    def __init__(self):
        self.args = Parser()
        self.args.get_args()
        self.n_events = len(self.args.events)

        if self.args.dataset == 'MAREA':

            self.event_dict = {
                'LF_HS': 0,
                'RF_HS': 1,
                'LF_TO': 2,
                'RF_TO': 3
            }

        self.event_indices = itemgetter(*self.args.events)(self.event_dict)
        if isinstance(self.event_indices, int):
            self.event_indices = [self.event_indices]

        else:
            self.event_indices = list(self.event_indices)

    def __call__(self, events, time_info=False):

        output = events[self.event_indices]

        if self.args.class_mode == 'multi_class':
            no_event = 0 if np.sum(output) == 0 else 1
            output = np.hstack((no_event, output))

        if time_info:
            return output, events[-3:]
        else:
            return output, None

    def get_shape(self):
        if self.args.class_mode == 'multi_class':
            return self.n_events + 1
        elif self.args.class_mode == 'multi_label':
            return self.n_events


class temporal:

    def __init__(self):
        self.args = Parser()
        self.args.get_args()

        if self.args.bf_preprocessing:
            self.initial_signals = {
                'Acc_x': 0,
                'Acc_y': 1,
                'Acc_z': 2,
                'Acc': 3,
                'Acc_xy': 4,
                'Acc_yz': 5,
                'Acc_xz': 6,
            }

            self.initials = 7

        else:
            self.initial_signals = {
                'Acc_x': 0,
                'Acc_y': 1,
                'Acc_z': 2
            }

            self.initials = 3

            self.virtual_signals = [
                'Acc',
                'Acc_xy',
                'Acc_yz',
                'Acc_xz'
            ]

        if self.args.dataset == 'MAREA':

            self.pos_dict = {
                'LF': 0,
                'RF': 1,
                'Waist': 2,
                'Wrist': 3
            }

        self.available_augs = ['Jittering', 'TimeWarp', 'Rotation']

        assert all(signal in self.initial_signals
                   or signal in self.virtual_signals
                   for signal in self.args.signals), "un-available signal"

        assert all(pos in self.pos_dict for pos in self.args.positions), "unavailable position"

        if self.args.augmentations is not None:
            assert all(aug in self.available_augs for aug in self.args.augmentations), "unavailable augmentation"

        self.pos_indices = itemgetter(*self.args.positions)(self.pos_dict)
        self.n_positions = len(self.args.positions)
        self.n_signals = len(self.args.signals)
        self.channels = self.n_signals * self.n_positions

    def get_shape(self):
        return self.args.length, self.channels

    def get_time_shape(self):
        return self.args.length, 3

    def __call__(self, window, training=True, time_info=False):
        signal = None
        output = None
        time = None

        raws = np.array([window[:, self.initials * pos_i: self.initials * pos_i + self.initials] for pos_i in self.pos_indices]).astype('float')

        if time_info:
            time = window[:, -3:]

        if training and self.args.augmentations:
            for aug in self.args.augmentations:
                if aug == 'Jittering':
                    noise = np.random.normal(0., 1., size=raws.shape[1:])
                    raws = np.array([raw + noise for raw in raws])

                elif aug == 'TimeWarp':
                    tt_new, x_range = DA_TimeWarp(self.args.length, 1.)
                    raws = np.array([np.array(
                        [np.interp(x_range, tt_new, raw[:, orientation]) for orientation in
                         range(3)]).transpose() for raw in raws])

                elif aug == 'Rotation':
                    raws = np.array([DA_Rotation(raw) for raw in raws])

        for thisSignal in self.args.signals:
            if thisSignal in self.initial_signals:
                s_i = self.initial_signals[thisSignal]
                signal = raws[:, :, s_i]

            else:
                if thisSignal == 'Acc':
                    signal = np.sqrt(np.sum(raws ** 2, axis=2))

                elif thisSignal == 'Acc_xy':
                    signal = np.sqrt(np.sum(raws[:, :, :2] ** 2, axis=2))

                elif thisSignal == 'Acc_xz':
                    signal = np.sqrt(np.sum(raws[:, :, ::2] ** 2, axis=2))

                elif thisSignal == 'Acc_yz':
                    signal = np.sqrt(np.sum(raws[:, :, 1:] ** 2, axis=2))

                elif thisSignal == 'Jerk':
                    J = np.array([np.array([(raw[1:, orientation] - raw[:-1, orientation]) * self.args.sampling_rate
                                  for orientation in range(3)]).transpose() for raw in raws])
                    signal = np.sqrt(np.sum(J ** 2, axis=2))
                    signal = np.concatenate((signal, np.zeros((signal.shape[0], 1))), axis=1)

            if output is None:
                output = signal[:, :, np.newaxis]

            else:
                output = np.concatenate(
                    (output, signal[:, :, np.newaxis]),
                    axis=2
                )

        transposed = np.transpose(output, (1, 0, 2))
        output = np.reshape(transposed, (self.args.length, self.channels))

        return output, time


