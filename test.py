from statsd import StatsClient

from statsd import StatsClient

statsd = StatsClient()

with statsd.timer('foo'):
    # This block will be timed.
    for i in range(0, 100000):
        i ** 2
