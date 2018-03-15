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

    learning_rates = generate_values(10, -5, 0.5)
    margins = generate_values(6, -4, 0)
    hidden_dims = [32, 64, 128, 256, 512]
    reg_strengths = generate_values(9, -6, -2)

    best_val_prctile = float("-inf")
    best_val_dist_diff = -1
    opt_rate = 0
    opt_margin = 0
    opt_hidden_dim = 0
    opt_reg_strength = 0
    for rate in learning_rates:
        for margin in margins:
            for hidden_dim in hidden_dims:
                for reg_strength in reg_strengths:
                    params.margin = float(margin)
                    params.learning_rate = float(rate)
                    params.hidden_dim = hidden_dim
                    params.reg_strength = float(reg_strength)

                    (avg_prctile, avg_dist_diff) = train.main(params, args)
                    if avg_prctile > best_val_prctile:
                        opt_rate = rate
                        opt_margin = margin
                        opt_hidden_dim = hidden_dim
                        opt_reg_strength = reg_strength
                        best_val_prctile = avg_prctile
                        best_val_dist_diff = avg_dist_diff

    print('Optimal Learning Rate: %f\nOptimal Margin: %f\nOptimal hidden_dim: %d\nOptimal Regularization: %f' % (opt_rate, opt_margin, opt_hidden_dim, opt_reg_strength))

def generate_values(num, low, high):
    log_vals = np.random.rand(num)
    log_vals = low + (log_vals * (high - low))
    return np.power(10, log_vals)

search()