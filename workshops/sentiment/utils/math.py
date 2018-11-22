def running_sum(xs):
    xs_sum = 0
    running = []
    for x in xs:
        xs_sum += x
        running.append(xs_sum)
    return running
