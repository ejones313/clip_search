import numpy as np

def search():
    learning_rates = generate_values(10, -6, -2)
    margins = generate_values(5, -2, 0.5)

    best_loss = 100000
    opt_rate = 0
    opt_margin = 0
    for rate in learning_rates:
        for margin in margins:
            loss = run_model(rate, margin)
            if loss < best_loss:
                opt_rate = 0
                opt_margin = 0
    print('Optimal Learning Rate: %f,\nOptimal Margin: %f' % (opt_rate, opt_margin))

def generate_values(num, low, high):
    log_vals = np.random.rand(num)
    log_vals = low + (log_vals * (high - low))
    return np.power(10, log_vals)