import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 1500


def smooth_result(result, factor=10):
    result = np.array(result)
    size = int(result.shape[0] / factor)
    result = np.convolve(result, np.ones(size) / size, mode='valid')
    return result


def create_run_name(alg, env, num_layers, hidden_dim, eps_start,
                    eps_end, decay, gamma, batch_size, lr, num_ep, num_step, updt_freq=None,
                    sw_freq=None):

    name = str(alg) + '_' + str(env) + '_' + str(num_layers) + '_x_' + str(hidden_dim)
    name += '_eps-' + str(round(eps_start, 1)) + '_' + str(round(eps_end, 4)) + '_' + str(round(decay, 3))
    name += '_gamma-' + str(round(gamma, 2)) + '_lr-' + str(round(lr, 4)) + '_batch-' + str(batch_size)
    name += '_ep-' + str(num_ep) + '_x_' + str(num_step)

    if updt_freq is not None:
        name += ('_updt_f-' + str(updt_freq))
    if sw_freq is not None:
        name += ('_sw_f-' + str(sw_freq))

    return name


def visualize_result(returns, td_errors, policy_errors=None):

    if policy_errors is None:
        fig, (ax1, ax2) = plt.subplots(2, figsize=[6.4, 6], gridspec_kw={'height_ratios': [2, 1]})
        ax3 = None
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=[6.4, 8], gridspec_kw={'height_ratios': [2, 1, 1]})

    ax1.plot(returns, label='returns', color='#3366ff')
    ax1.plot(smooth_result(returns), label='smoothed', color='#000066')
    ax1.set_title('Returns')
    ax1.set_xlabel('episodes')
    ax1.legend()

    ax2.plot(td_errors, color='#ff3300')
    ax2.set_xlabel('update steps')
    ax2.set_title('TD Errors')

    if policy_errors is not None:
        ax3.plot(policy_errors, color='#ff0000')
        ax3.set_xlabel('update steps')
        ax3.set_title('Policy Loss')

    fig.tight_layout()
    return fig
