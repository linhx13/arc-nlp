# -*- coding: utf-8 -*-

import logging
from multiprocessing import cpu_count, Process, Queue

logger = logging.getLogger(__name__)


TYPE_NORMAL = "OK"
TYPE_DONE = "DONE"


class Task(object):
    @classmethod
    def create_normal(cls, payload):
        return cls(TYPE_NORMAL, payload)

    @classmethod
    def create_done(cls):
        return cls(TYPE_DONE, None)

    def __init__(self, type, payload):
        self.type = type
        self.payload = payload


def input_func(iterable, input_queue, num_workers):
    for data in iterable:
        input_queue.put(Task.create_normal(data))
    for _ in range(num_workers):
        input_queue.put(Task.create_done())


def worker_func(func, input_queue, output_queue):
    while True:
        task = input_queue.get()
        if task.type == TYPE_DONE:
            break
        try:
            result = func(task.payload)
            if result is None:
                continue
        except Exception as ex:
            logger.warn("work_func process func error: %s" % ex)
            continue
        output_queue.put(Task.create_normal(result))
    output_queue.put(Task.create_done())


def output_func(output_queue, num_workers):
    n_done = 0
    while True:
        task = output_queue.get()
        if task.type == TYPE_DONE:
            n_done += 1
            if n_done == num_workers:
                break
        else:
            yield task.payload


class SimplePool(object):
    def __init__(self, processes=None):
        self.processes = processes if processes > 0 else cpu_count()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def imap_unordered(self, func, iterable, **kwargs):
        input_queue = Queue()
        output_queue = Queue()

        workers = []
        for _ in range(self.processes):
            worker = Process(target=worker_func, args=(func, input_queue, output_queue))
            worker.start()
            workers.append(worker)
        inputter = Process(target=input_func, args=(iterable, input_queue, self.processes))
        inputter.start()
        for x in output_func(output_queue, self.processes):
            yield x
        for worker in workers:
            worker.join()
        inputter.join()

    def imap(self, func, iterable, **kwargs):
        for x in self.imap_unordered(func, iterable, **kwargs):
            yield x

    def close(self):
        pass

    def join(self):
        pass