import numpy as np
import train
import utils
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/small', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def search():
    args = parser.parse_args()
    args.model_dir = '.'
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    params.num_epochs = 100
    params.train_file = "train_1000.pkl"
    params.val_file = "val_500.pkl"

    learning_rates = generate_values(10, -6, -2)
    margins = generate_values(5, -2, 0.5)

    best_val_loss = 100000
    best_val_frac_hard = -1
    opt_rate = 0
    opt_margin = 0
    for rate in learning_rates:
        for margin in margins:
            params.margin = margin
            params.learning_rate = learning_rate

            (val_loss, val_frac_hard) = train.main(params)
            if val_loss < best_val_oss:
                opt_rate = rate
                opt_margin = margin
                best_val_loss = loss
                best_val_frac_hard = val_frac_hard

    print('Optimal Learning Rate: %f,\nOptimal Margin: %f' % (opt_rate, opt_margin))

def generate_values(num, low, high):
    log_vals = np.random.rand(num)
    log_vals = low + (log_vals * (high - low))
    return np.power(10, log_vals)