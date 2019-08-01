import os
import datetime
import matplotlib.pyplot as plt
import io


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    return buf.getvalue()


def get_output_dir(output_root='.'):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    return os.path.join(output_root, 'output', timestamp)


def ensure_parent_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='#'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s %s%%' % (prefix, bar, suffix, percent), end='\r')
    # print new line on complete
    if iteration == total:
        print()
