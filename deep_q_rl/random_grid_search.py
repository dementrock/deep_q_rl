import os
os.environ['CGT_COMPAT_MODE'] = 'theano'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['THEANO_FLAGS'] = 'device=cpu'

import random
import os
import time
from multiprocessing import Pool, cpu_count

class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 250000
    EPOCHS = 200
    STEPS_PER_TEST = 125000

    # ----------------------
    # ALE Parameters
    # ----------------------
    BASE_ROM_PATH = "../roms/"
    ROM = 'breakout.bin'
    FRAME_SKIP = 4
    REPEAT_ACTION_PROBABILITY = 0

    # ----------------------
    # Agent/Network parameters:
    # ----------------------
    UPDATE_RULE = 'deepmind_rmsprop'
    BATCH_ACCUMULATOR = 'sum'
    LEARNING_RATE = .00025
    DISCOUNT = .99
    RMS_DECAY = .95 # (Rho)
    RMS_EPSILON = .01
    MOMENTUM = 0 # Note that the "momentum" value mentioned in the Nature
                 # paper is not used in the same way as a traditional momentum
                 # term.  It is used to track gradient for the purpose of
                 # estimating the standard deviation. This package uses
                 # rho/RMS_DECAY to track both the history of the gradient
                 # and the squared gradient.
    CLIP_DELTA = 1.0
    EPSILON_START = 1.0
    EPSILON_MIN = .1
    EPSILON_DECAY = 1000000
    PHI_LENGTH = 4
    UPDATE_FREQUENCY = 4
    REPLAY_MEMORY_SIZE = 1000000
    BATCH_SIZE = 32
    NETWORK_TYPE = "nature_dnn"
    FREEZE_INTERVAL = 10000
    REPLAY_START_SIZE = 50000
    RESIZE_METHOD = 'scale'
    RESIZED_WIDTH = 84
    RESIZED_HEIGHT = 84
    DEATH_ENDS_EPISODE = 'true'
    MAX_START_NULLOPS = 30
    DETERMINISTIC = True
    CUDNN_DETERMINISTIC = False
    HIDDEN_SIZES = [64]


def search(i):

    batch_sizes = [32]
    learning_rates = [0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.000075, 0.00005, 0.000025, 0.00001]
    hidden_sizes = [[64], [128], [256], [64, 64], [128, 64], [128, 128], [64, 64, 64], [128, 128, 64], [128, 128, 128], [256, 128, 64], [256, 128, 128]]
    rms_decays = [0.9, 0.95, 0.97, 0.99, 0.999, 0.9999, 0.99999]
    rms_epsilons = [0.01, 0.001, 0.1, 0.0001, 0.00001]

    game = "pong"

    time.sleep(i*10)

    while True:
        batch_size = random.choice(batch_sizes)
        learning_rate = random.choice(learning_rates)
        hidden_size = random.choice(hidden_sizes)
        rms_decay = random.choice(rms_decays)
        rms_epsilon = random.choice(rms_epsilons)
        experiment_prefix = "%s_%s_%s" % (game, "_".join(map(str, hidden_size)), "batch%d" % batch_size)
        import launcher
        import sys

        command = """-r %(game)s -e 10 --batch-accumulator mean --hidden-sizes %(hidden_sizes)s --rms-decay %(rms_decay)s --rms-epsilon %(rms_epsilon)s --learning-rate %(learning_rate)s --batch-size %(batch_size)s --experiment-prefix %(prefix)s""" % dict(
            game=game,
            hidden_sizes=" ".join(map(str, hidden_size)),
            rms_decay=str(rms_decay),
            rms_epsilon=str(rms_epsilon),
            learning_rate="%f" % learning_rate,
            batch_size=str(batch_size),
            prefix="%s_hidden_%s_rmsdecay_%s_rmsepsilon_%s_lr_%s_batchsize_%s" % (game, "_".join(map(str, hidden_size)), str(rms_decay), str(rms_epsilon), "%f" % learning_rate, str(batch_size)))
        args = command.split(" ")
        print command
        launcher.launch(args, Defaults, __doc__)
        time.sleep(1)

#search(0)
#search()
p = Pool(cpu_count())
p.map(search, range(cpu_count()))
