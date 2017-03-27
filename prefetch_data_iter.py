# -*- coding: utf-8 -*-
import mxnet
import mxnet as mx
import cPickle
import numpy
import random
from random import shuffle
import threading
import cv2
import time
import logging
import os
from mxnet import ndarray as nd

class PrefetchDataIter(mx.io.DataIter):
    """Base class for prefetching iterators. Takes one or more DataIters (
    or any class with "reset" and "next" methods) and combine them with
    prefetching. For example:

    Parameters
    ----------
    iters : DataIter or list of DataIter
        one or more DataIters (or any class with "reset" and "next" methods)
    rename_data : None or list of dict
        i-th element is a renaming map for i-th iter, in the form of
        {'original_name' : 'new_name'}. Should have one entry for each entry
        in iter[i].provide_data
    rename_label : None or list of dict
        Similar to rename_data
    """
    
    def __init__(self, image_list, batch_size, is_color, is_mirror=True, root_dir='', nThread = 3):
        super(PrefetchDataIter, self).__init__()
        self.image_list = [line.strip() for line in open(image_list).readlines()]
        self.is_mirror = is_mirror
        self.is_color = is_color
        self.mean = 127.5
        self.scale = 0.0078125
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.image_count = 0

        self.reset()
        infer_shape_imgname = os.path.join(root_dir, self.image_list[0].split()[0])
        height, width, channel = cv2.imread(infer_shape_imgname).shape
        if not is_color:
            channel = 1
        self.data_shape = (batch_size, channel, height, width)
        self.label_shape = (batch_size,)
        print self.data_shape
        print self.label_shape

        self.nThread = nThread
        self.data_ready = [threading.Event() for i in range(self.nThread)]
        self.data_taken = [threading.Event() for i in range(self.nThread)]
        self.cur_ok = threading.Event()
        self.cur_ok.set()
        for e in self.data_taken:
            e.set()
        for e in self.data_ready:
            e.clear()
        self.started = True
        self.current_batch = None
        self.data_queue = [None for i in range(self.nThread)]
        def prefetch_func(self, i):
            """Thread entry"""
            while True:
                self.data_taken[i].wait()
                if not self.started:
                    break
                try:
                    self.cur_ok.wait()
                    self.cur_ok.clear()
                    self.data_queue[i] = self._prepare_batch(i)
                    self.cur_ok.set()
                    time.sleep(0.001)
                except StopIteration:
                    self.next_batch[i] = None
                self.data_taken[i].clear()
                self.data_ready[i].set()

        self.prefetch_threads = [threading.Thread(target=prefetch_func, 
                                                  args=[self, i]) for i in range(self.nThread)]

        for thread in self.prefetch_threads:
            thread.setDaemon(True)
            thread.start()
    
    def __del__(self):
        self.started = False
        for e in self.data_taken:
            e.set()
        for thread in self.prefetch_threads:
            thread.join()

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [('data', self.data_shape)]


    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        return [('softmax_label' , self.label_shape), ('center_label', self.label_shape)]

    def reset(self):
        self.cur_ = 0
        shuffle(self.image_list)
    
    def _prepare_batch(self, i):
        # print "thread  %d" %i
        data = nd.zeros(self.data_shape)
        label = nd.zeros(self.label_shape)
        for num_i in range(self.batch_size):
            if self.cur_ == len(self.image_list):
                self.reset()
                # raise StopIteration
            # print self.cur_
            img_name = os.path.join(self.root_dir, self.image_list[self.cur_].split()[0])
            label_i = int (self.image_list[self.cur_].split()[1])
            if self.is_color:
                im = cv2.imread(img_name)
                im = numpy.transpose(im, axes = (2,0,1))
            else:
                im = cv2.imread(img_name, 0)
                im = numpy.reshape(im, (1,)+ im.shape)
            im = (im - 127.5)*self.scale
            if self.is_mirror and random.random() > 0.5:
                im = im[:, ::-1, ]
            data[num_i][:]  = im
            # print num_i, label_i
            label[num_i] = label_i
            self.cur_ += 1 
        return mx.io.DataBatch([data], [label, label])

    def iter_next(self):
        if self.image_count > len(self.image_list):
            self.image_count = 0
            raise StopIteration

        # keep looping until getting the databatch
        while True:
            for i, dataBatch in enumerate(self.data_queue):
                if not self.started:
                    quit()
                if dataBatch == None:
                    continue
                self.data_ready[i].wait()
                self.current_batch = dataBatch
                self.data_queue[i] = None
                self.data_ready[i].clear()
                self.data_taken[i].set()
                self.image_count += self.batch_size
                return True
            time.sleep(0.005)# key part!!!!!!!!!!!!

    def next(self):
        if self.iter_next():
            return self.current_batch
        else:
            raise StopIteration

    def getdata(self):
        return self.current_batch.data

    def getlabel(self):
        return self.current_batch.label

    def getindex(self):
        return self.current_batch.index

    def getpad(self):
        return self.current_batch.pad


def testPrefetchingFaceIter():
    logging.getLogger().setLevel(logging.DEBUG)
    myIter = PrefetchDataIter(image_list = './data/test_correct.txt', batch_size = 256,\
                                     root_dir='./data/',\
                                     is_color = False, nThread = 2)
    numFetch = 100
    start = time.time()
    for loop in range(numFetch):
        batch=myIter.next()
        # pdb.set_trace()
        # print loop, batch.data[0].shape, batch.label[0].shape


    print 'total fetching time: %f s, average %f s per iter' % ((time.time()-start), (time.time()-start)/numFetch)
if __name__=='__main__':
    testPrefetchingFaceIter()


