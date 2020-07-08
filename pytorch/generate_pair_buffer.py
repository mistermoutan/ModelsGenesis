import multiprocessing
import threading

from image_transformations import *


class GeneratePairBuffer:
    def __init__(self, dataset, config):

        from collections import deque

        self.dataset = dataset
        self.config = config
        self.tr, self.val, self.ts = deque(), deque(), deque()
        self._start_processes()

    def get_train(self):

        return_tuple = self.tr.popleft()
        return return_tuple

    def get_val(self):

        return_tuple = self.val.popleft()
        return return_tuple

    def get_ts(self):

        return_tuple = self.ts.popleft()
        return return_tuple

    def _start_processes(self):

        self.p_tr = threading.Thread(target=self._tr_process)  # multiprocessing.Process(target=self._tr_process)
        self.p_val = threading.Thread(target=self._val_process)  # multiprocessing.Process(target=self._val_process)
        # self.p_ts = threading.Thread(target=self._ts_process)#multiprocessing.Process(target=self._ts_process)
        # processes = [self.p_tr, self.p_val]
        self.p_tr.start()
        # self.p_val.start()

    def _tr_process(self):

        while True:
            x, _ = self.dataset.get_train(batch_size=self.config.batch_size_ss, return_tensor=False)
            if x is None:
                self.tr.append((x, _))  # so None gets passed to Trainer
                while self.dataset.reseted is False:
                    time.sleep(0.2)
                continue  # dataset has been reset lets resume generation
            x_transform, y = generate_pair(x, self.config.batch_size_ss, self.config, make_tensors=True)
            print("GOT HERE")
            self.tr.append((x_transform, y))

    def _val_process(self):

        while True:
            x, _ = self.dataset.get_val(batch_size=self.config.batch_size_ss, return_tensor=False)
            if x is None:
                self.val.append((x, _))  # so None gets passed to Trainer
                while self.dataset.reseted is False:
                    time.sleep(0.2)
                continue  # dataset has been reset lets resume generation
            x_transform, y = generate_pair(x, self.config.batch_size_ss, self.config, make_tensors=True)
            self.val.append((x_transform, y))

    def _ts_process(self):

        while True:
            x, _ = self.dataset.get_test(batch_size=self.config.batch_size_ss)
            if x is None:  # epoch done
                self.ts.append((x, _))  # so None gets passed to Trainer
                while self.dataset.reseted is False:
                    time.sleep(0.2)
                continue  # dataset has been reset lets resume generation
            x_transform, y = generate_pair(x, self.config.batch_size_ss, self.config, make_tensors=True)
            self.ts.append((x_transform, y))
