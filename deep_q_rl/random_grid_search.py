import random
import os
batch_sizes = [32, 64, 128, 256]
learning_rates = [0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.000075, 0.00005, 0.000025, 0.00001]
hidden_sizes = [[64], [128], [256], [64, 64], [128, 64], [128, 128], [64, 64, 64], [128, 128, 64], [128, 128, 128], [256, 128, 64], [256, 128, 128]]
rms_decays = [0.9, 0.95, 0.97, 0.99, 0.999, 0.9999, 0.99999]
rms_epsilons = [0.01, 0.001, 0.1, 0.0001, 0.00001]

while True:
    batch_size = random.choice(batch_sizes)
    learning_rate = random.choice(learning_rates)
    hidden_size = random.choice(hidden_sizes)
    rms_decay = random.choice(rms_decays)
    rms_epsilon = random.choice(rms_epsilons)
    experiment_prefix = "pong_%s_%s" % ("_".join(map(str, hidden_size)), "batch%d" % batch_size)
    command = """THEANO_FLAGS="device=cpu" python run_nature.py -r pong -e 5 --batch-accumulator mean --hidden-sizes %(hidden_sizes)s --rms-decay %(rms_decay)s --rms-epsilon %(rms_epsilon)s --learning-rate %(learning_rate)s --batch-size %(batch_size)s --experiment-prefix %(prefix)s""" % dict(
        hidden_sizes=" ".join(map(str, hidden_size)),
        rms_decay=str(rms_decay),
        rms_epsilon=str(rms_epsilon),
        learning_rate="%f" % learning_rate,
        batch_size=str(batch_size),
        prefix="pong_hidden_%s_rmsdecay_%s_rmsepsilon_%s_lr_%s_batchsize_%s" % ("_".join(map(str, hidden_size)), str(rms_decay), str(rms_epsilon), "%f" % learning_rate, str(batch_size)))
    print command
    os.system(command)
