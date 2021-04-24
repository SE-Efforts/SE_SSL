from __future__ import print_function, division
import os
import torch
import numpy as np
from os.path import join
from utils import ConfigBase, Logger
import pdb


class Config(ConfigBase):
    def __init__(self):
        super(Config, self).__init__()
        self.dataset_config = "====== Simulated dataset ======"
        self.sample_num = 1000
        self.sample_dim= 20
        # self.intrinsic_dim = 3
        self.intrinsic_dim = 5
        # fake intrinsic dim, try totally randomly one


class Dataset(object):
    def __init__(self, config):
        sample_num = config.sample_num
        sample_dim = config.sample_dim
        intrinsic_dim = config.intrinsic_dim
        print(">> preparing dataset")
        print("  generating intrinsic samples")
        self.samples = self.generate_intrinsic_samples(sample_num, intrinsic_dim)
        # print("  shifting means")
        # self.shift_mean(np.random.randn(intrinsic_dim))
        # #######################
        # print("  scaling variances")
        # self.scale_variance(np.random.randn(intrinsic_dim))
        # print("  transforming into high dimension space")
        # self.transform_to_high_dim(sample_dim)
        # ##########################
        print(">> finish creating dataset")
        
    def generate_intrinsic_samples(self, sample_num, intrinsic_dim):
        # simplest iid standard gaussian distribution
        samples = np.random.randn(sample_num, intrinsic_dim)
        return samples

    def shift_mean(self, shifts):
        """ shifts is a vector with dim equals to the sample dim
                This func can be used to testify the effect of shift the mean of samples
        """
        assert self.samples.shape[1] == shifts.shape[0], \
                "vector shifts has shape {} doesn't match sample dimension {}".format(
                    shifts.shape[0], self.samples.shape[1]
                )
        self.samples = self.samples + shifts

    def scale_variance(self, scales):
        """ scales is a vector with dim equals to the sample dim
                This func can be used to testify the effect of change the variance of samples
        """
        assert self.samples.shape[1] == scales.shape[0], \
                "vector scales has shape {} doesn't match sample dimension {}".format(
                    scales.shape[0], self.samples.shape[1]
                )
        means = self.samples.mean(axis=0)
        self.shift_mean(-means)
        self.samples = self.samples * scales
        self.shift_mean(means)

    def transform_to_high_dim(self, output_sample_dim):
        input_dim = self.samples.shape[1]
        # column-stochastic matrix
        mat = np.random.randn(input_dim, output_sample_dim).clip(min=0)
        mat = mat / mat.sum(axis=1)[:, None]
        # perform transformation
        self.samples = np.matmul(self.samples, mat)
        
    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, index):
        return self.samples[index]


def test(config):
    dataset = Dataset(config)
    print(">> Pass ")
    

if __name__ == '__main__':
    from utils import Logger
    config = Config()
    config.parse_args()  # revise configurations from cmd line
    # # Results will be printed on screen and log file
    # sys.stdout = Logger('{0}/{1}'.format(config.log_dir, config.log_filename))
    config.print_args()
    test(config)
