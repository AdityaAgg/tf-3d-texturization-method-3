import numpy as np
import util
import config

class Dataset(object):

    def __init__(self):
        self.index_in_epoch = 0
        self.load_data()
        self.random_shuffle()



    def load_data(self):
        self.styles = np.load('data/preprocessed/colored.npz')['a']/255.0
        self.geometry = np.load('data/preprocessed/voxels.npz')['a']

        print np.amax(self.styles)
        self.pictures = np.load('data/preprocessed/image.npz')['a']
        print np.amax(self.pictures)
        print self.geometry.shape
        print self.pictures.shape
        print self.styles.shape
        self.num_examples = self.styles.shape[0]


        self.train_max = int(0.8*self.num_examples)

        #test set
        self.test_geometry = self.geometry[self.train_max:self.num_examples]
        self.test_styles = self.styles[self.train_max:self.num_examples]
        self.test_pictures = self.pictures[self.train_max:self.num_examples]

        #train set
        self.geometry = self.geometry[0:self.train_max]
        self.styles = self.styles[0:self.train_max]
        self.pictures = self.pictures[0:self.train_max]

    #for train
    def random_shuffle(self):
        indices = np.random.shuffle(np.arange(self.train_max))
        self.styles = self.styles[indices][0]
        self.geometry = self.geometry[indices][0]
        self.pictures = self.pictures[indices][0]


    def next_batch(self, batch_size):
        print "batch size ", batch_size
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.train_max:
            self.random_shuffle()
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.train_max
        end = self.index_in_epoch
        return self.read_data(start, end)


    def read_data(self, start, end):
        print start
        print end
        styles = self.styles[start:end]
        print self.styles.shape
        geometry = np.expand_dims(self.geometry[start:end], -1)
        print self.pictures.shape
        pictures = self.pictures[start:end, :, :, 0:3]
        print pictures.shape
        return styles, pictures, geometry




    #for test

    def get_random_sample(self):
        random_range =  self.train_max
        random_index = np.random.randint(random_range)
        return np.expand_dims(self.styles[random_index], 0), np.expand_dims(self.pictures[random_index, :, :, 0:3],0), np.expand_dims(np.expand_dims(self.geometry[random_index], -1),0)




