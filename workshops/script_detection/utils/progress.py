import time
import logging


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
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s %s%%' % (prefix, bar, suffix, percent), end='\r')
    # print new line on complete
    if iteration == total:
        print()


class ProgressBar(object):
    def __init__(self, total, name='', format='%H:%M:%S'):
        self._total = total
        self._name = name
        self._format = format
        self._start = time.time()

    def show(self, iteration, suffix=None):
        now = time.time()
        avg = (now - self._start) / (iteration + 1)
        eta = max(0, self._start + avg * self._total - now)
        eta_str = "{0}".format(self._format_timestamp(eta))
        iteration_str = "{0}K".format(iteration // 1000)
        if suffix:
            suffix = "{0} {1} ({2})".format(suffix, eta_str, iteration_str)
        else:
            suffix = "{0} ({1})".format(eta_str, iteration_str)
        progress_bar(iteration + 1, self._total, prefix=self._name, suffix=suffix)

    def total(self, suffix=None):
        now = time.time()
        elapsed = self._format_timestamp(now - self._start)
        if suffix:
            logging.info("{0} lasted {1} {2}".format(self._name, elapsed, suffix))
        else:
            logging.info("{0} lasted {1}".format(self._name, elapsed))

    def _format_timestamp(self, timestamp):
        return time.strftime(self._format, time.gmtime(timestamp))
