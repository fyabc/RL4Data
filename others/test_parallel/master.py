#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function

import os
import psutil
import subprocess
import time


def main():
    process_size = 4

    for episode in range(5):
        ret_values = [None for _ in range(process_size)]
        results = [None for _ in range(process_size)]

        envs = [os.environ.copy() for _ in range(process_size)]

        for i, env in enumerate(envs):
            env[str('THEANO_FLAGS')] = str('device=cpu,floatX=float{}'.format(32 if i % 2 == 0 else 16))

        time_before = time.time()
        pool = [
            psutil.Popen(
                ['python', 'slave.py'],
                stdout=subprocess.PIPE,
                env=envs[i],
            )
            for i in range(process_size)
        ]

        # time_after = time.time()

        # Roll polling
        while any(e is None for e in ret_values):
            for i, process in enumerate(pool):
                ret_values[i] = process.poll()
            time.sleep(1.0)

        time_after = time.time()

        for i, process in enumerate(pool):
            if ret_values[i] == 0:
                results[i], _ = process.communicate()

        print('Time: {:.6}s'.format(time_after - time_before))
        print(*results, sep='')


if __name__ == '__main__':
    main()
