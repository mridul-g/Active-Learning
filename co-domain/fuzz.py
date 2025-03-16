import sys
import copy
import random
import numpy as np
import time
from tqdm import tqdm
import itertools
import gc
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from torchvision.utils import save_image

import coverage
import utility
# from style_operator import Stylized
# import image_transforms

from PIL import Image


class Parameters(object):
    def __init__(self, base_args):
        self.model = base_args.model
        self.dataset = base_args.dataset
        self.criterion = base_args.criterion
        self.only_last = base_args.only_last
        self.max_testsuite_size = base_args.max_testsuite_size
        self.seed_id = base_args.seed_id

        self.use_sc = self.criterion in ['LSC', 'DSC', 'MDSC']
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 2

        self.batch_size = 128
        self.mutate_batch_size = 1

        self.image_size = constants.data_info[self.dataset]['img_size']
        self.nc = constants.data_info[self.dataset]['n_img_channels']
        self.n_test_class = constants.data_info[self.dataset]['n_test_class']
        self.input_shape = (1, self.image_size, self.image_size, self.nc)
        self.num_per_class = 1000 // self.n_test_class

        if self.criterion != 'random':
            self.criterion_hp = constants.hyper_map[self.criterion]
        
        # self.input_scale = 255
        self.noise_data = False
        self.K = 64
        self.batch1 = 64
        self.batch2 = 16

        self.alpha = 0.2  # default 0.02
        self.beta = 0.4  # default 0.2
        self.TRY_NUM = 50
        self.save_every = 100
        self.output_dir = base_args.output_dir

        """
        translation = list(itertools.product([getattr(image_transforms, "image_translation")],
                                             [(-5, -5), (-5, 0), (0, -5), (0, 0), (5, 0), (0, 5), (5, 5)]))
        scale = list(itertools.product([getattr(image_transforms, "image_scale")], list(np.arange(0.8, 1, 0.05))))
        # shear = list(itertools.product([getattr(image_transforms, "image_shear")], list(range(-3, 3))))
        rotation = list(itertools.product([getattr(image_transforms, "image_rotation")], list(range(-30, 30))))

        contrast = list(itertools.product(
            [getattr(image_transforms, "image_contrast")], [0.8 + 0.2 * k for k in range(7)]))
        brightness = list(itertools.product(
            [getattr(image_transforms, "image_brightness")], [10 + 10 * k for k in range(7)]))
        blur = list(itertools.product([getattr(image_transforms, "image_blur")], [k + 1 for k in range(10)]))

        self.stylized = Stylized(self.image_size)

        self.G = translation + scale + rotation  # + shear
        self.P = contrast + brightness + blur
        self.S = []  # list(itertools.product([self.stylized.transform], [0.4, 0.6, 0.8]))
        """
        self.save_batch = False


class INFO(object):
    def __init__(self):
        self.dict = {}
        self.transformlist = []

    def __getitem__(self, i):
        _i = str(i)
        if _i in self.dict:
            return self.dict[_i]
        else:
            I0, state = i, 0
            return I0, state

    def __setitem__(self, i, s):
        _i = str(i)
        self.dict[_i] = s
        return self.dict[_i]


class Fuzzer:
    def __init__(self, params, criterion):
        self.params = params
        self.epoch = 0
        self.time_slot = 60 * 10
        self.time_idx = 0
        self.info = INFO()
        self.hyper_params = {
            'alpha': 0.2,  # [0, 1], default 0.02, 0.1 # number of pix
            'beta': 0.4,  # [0, 1], default 0.2, 0.5 # max abs pix
            'TRY_NUM': 50,
            'p_min': 0.01,
            'gamma': 5,
            'K': 64
        }
        self.logger = utility.Logger(params, self)
        self.criterion = criterion
        if criterion != 'random':
            self.initial_coverage = copy.deepcopy(criterion.current)
        else:
            self.initial_coverage = 0
        self.delta_time = 0
        self.coverage_time = 0
        self.delta_batch = 0
        self.num_ae = 0

        self.transform_list = constants.dataset_transforms[self.params.dataset]
        self.n_transforms = len(self.transform_list["pixel"]) + len(self.transform_list["affine"])

        self.testsuite = []

    def exit(self):
        self.print_info()
        if self.criterion!='random':
            self.criterion.save(self.params.coverage_dir + 'coverage.pt')
        # self.logger.save()
        self.logger.exit()

        self.testsuite.append((self.coverage_time, self.delta_time))
        with open(f'{self.params.coverage_dir}testsuite.pkl', "wb") as f:
            pickle.dump(self.testsuite, f)

    def can_terminate(self):
        condition = sum([
            self.epoch > 10000,
            self.delta_time > 60 * 60 * 6,
            len(self.testsuite) > self.params.max_testsuite_size,
        ])
        return condition > 0

    def print_info(self):
        self.logger.update(self)

    def is_adversarial(self, image, label, k=1):
        """
        image will be given as input to the model
        """
        with torch.no_grad():
            scores = self.criterion.model(image)
            _, ind = scores.topk(k, dim=1, largest=True, sorted=True)

            correct = ind.eq(label.view(-1, 1).expand_as(ind))
            wrong = ~correct
            index = (wrong == True).nonzero(as_tuple=True)[0]
            wrong_total = wrong.view(-1).float().sum()

            return wrong_total, index

    """
    def to_batch(self, data_list):
        batch_list = []
        batch = []
        for i, data in enumerate(data_list):
            if i and i % self.params.mutate_batch_size == 0:
                batch_list.append(np.stack(batch, 0))
                batch = []
            batch.append(data)
        if len(batch):
            batch_list.append(np.stack(batch, 0))
        return batch_list

    def image_to_input(self, image):
        scaled_image = image / self.params.input_scale
        tensor_image = torch.from_numpy(scaled_image).transpose(1, 3)
        normalized_image = utility.image_normalize(tensor_image, self.params.dataset)
        return normalized_image

    def Preprocess(self, image_list, label_list):
        randomize_idx = np.arange(len(image_list))
        np.random.shuffle(randomize_idx)
        image_list = [image_list[idx] * self.params.input_scale for idx in randomize_idx]
        label_list = [label_list[idx] for idx in randomize_idx]

        # len(Bs) is number of batches, len(Bs[0]) is number of images in one mutation batch i.e. 1
        Bs = self.to_batch(image_list)
        Bs_label = self.to_batch(label_list)

        return list(np.zeros(len(Bs))), Bs, Bs_label

    def Sample(self, B):
        c = np.random.choice(len(B), size=self.params.mutate_batch_size, replace=False)
        return B[c]

    def BatchPrioritize(self, T, B_id):
        B_c, Bs, Bs_label = T
        B_c[B_id] += 1

    def randomPick(self, A):
        c = np.random.randint(0, len(A))
        return A[c]
    """

    def run(self, I_input, L_input):
        # F = np.array([]).reshape(0, *(self.params.input_shape[1:])).astype('float32')   # F.shape: (0, 32, 32, 3)
        # T = self.Preprocess(I_input, L_input)   # images are now multiplied by 255 without removing the normalization

        # del I_input
        # del L_input
        # gc.collect()
        """
        B is a list containing an image as a numpy array of shape (32, 32, 3)
        B_label is a list containing a scalar int as a label
        B_id is the index of the selected seed
        """
        # T = (list(np.zeros(len(I_input))), I_input, L_input)
        selection_priority = (list(np.zeros(len(I_input))))

        self.epoch = 0
        start_time = time.time()
        selected_seed_id = self.SelectNext(selection_priority)
        while not self.can_terminate():
            if self.epoch % 500 == 0:
                self.print_info()
            # S = self.Sample(B)
            # S = B
            # S_label = B_label

            seed_img = I_input[selected_seed_id]
            label = L_input[selected_seed_id]
            Ps = self.PowerSchedule([seed_img], self.hyper_params['K'])
            img_gen = False

            """
            print(Ps(0))
            exit()
            B_new = np.array([]).reshape(0, *(self.params.input_shape[1:])).astype('float32')
            B_old = np.array([]).reshape(0, *(self.params.input_shape[1:])).astype('float32')
            B_label_new = []
            for s_i in range(len(S)):
                I = S[s_i]
                L = S_label[s_i]
            """

            # number of times a seed is mutated; calculated using power schedule, why? terminated if coverage increases, why?
            # why the validity of new image is not checked?
            time_per_seed = 0
            for i in range(1, Ps(0) + 1):
                I_new = self.Mutate(seed_img)
                if self.isFailedTest(I_new):
                    F += np.concatenate((F, [I_new.cpu().numpy()]))
                elif self.isChanged(seed_img, I_new):
                    if seed_set.normalise is not None:
                        I_new_norm = seed_set.normalise(I_new)
                    else:
                        I_new_norm = I_new

                    I_new_norm = I_new_norm.to(self.params.device).unsqueeze(0)
                    label = label.to(self.params.device).unsqueeze(0)

                    # print(I_new_norm.shape)
                    if self.criterion != 'random':
                        cov_start_time = time.time()
                        if self.params.use_sc or self.params.criterion=='CDC_v2':
                            cov_dict = self.criterion.calculate(I_new_norm, label)
                        else:
                            cov_dict = self.criterion.calculate(I_new_norm)
                        gain = self.criterion.gain(cov_dict)

                        if self.CoverageGain(gain):
                            self.criterion.update(cov_dict, gain)
                            img_gen = True
                            # B_new = np.concatenate((B_new, [I_new]))
                            # B_old = np.concatenate((B_old, [I]))
                            # B_label_new += [L]
                        cov_end_time = time.time()
                        time_per_seed += (cov_end_time - cov_start_time)
                    else:
                        img_gen = True
                    # exit()
                    if img_gen == True:
                        break

            if img_gen:
                I_input.append(I_new)
                L_input.append(label.cpu().squeeze())
                selection_priority.append(selection_priority[selected_seed_id] * 0)
                selection_priority[selected_seed_id] += 1
                self.delta_batch += 1

                # if self.criterion != 'random':
                #     num_wrong, _ = self.is_adversarial(I_new_norm, label)
                #     if num_wrong > 0:
                #         self.num_ae += num_wrong

                #     if self.epoch % self.params.save_every == 0:
                #         self.saveImage(I_new,
                #                     self.params.image_dir + ('%03d_new.jpg' % self.epoch))
                #         self.saveImage(seed_img,
                #                     self.params.image_dir + ('%03d_old.jpg' % self.epoch))
                #         if num_wrong > 0:
                #             print('Saving AE images...')
                #             save_image(I_new, self.params.image_dir +
                #                     ('%03d_ae.jpg' % self.epoch))
                        
                self.testsuite.append((I_new, label))

            gc.collect()

            selected_seed_id = self.SelectNext(selection_priority)
            self.epoch += 1     # number of seeds selected before exiting
            self.coverage_time += time_per_seed
            self.delta_time = time.time() - start_time

    def calc_priority(self, B_ci):
        if B_ci < (1 - self.hyper_params['p_min']) * self.hyper_params['gamma']:
            return 1 - B_ci / self.hyper_params['gamma']
        else:
            return self.hyper_params['p_min']

    def SelectNext(self, B_c):
        B_p = [self.calc_priority(B_c[i]) for i in range(len(B_c))]
        c = np.random.choice(len(B_c), p=B_p / np.sum(B_p))
        return c

    def PowerSchedule(self, S, K):
        potentials = []
        for i in range(len(S)):
            I = S[i]
            I0, state = self.info[I]
            # print(I0.shape)
            p = self.hyper_params['beta'] * 255 * torch.sum(I > 0) - torch.sum(torch.abs(I - I0))
            potentials.append(p)
        potentials = np.array(potentials) / np.sum(potentials)  # potentials = [1.]

        def Ps(I_id):
            p = potentials[I_id]
            return int(np.ceil(p * K))
        return Ps

    def isFailedTest(self, I_new):
        return False

    def isChanged(self, I, I_new):
        return torch.any(I != I_new)

    def CoverageGain(self, gain):
        if gain is not None:
            if isinstance(gain, tuple):
                return gain[0] > 0
            else:
                return gain > 0
        else:
            return False

    def Mutate(self, I: torch.tensor) -> torch.tensor:
        """
        Mutate image by applying transformations. 
        affine_trans is initially False and is set to True when an affine transformation is applied. 
        However, if the affine transformed image is not valid then, affine_trans is set to False again.
        """
        I0, state = self.info[I]
        assert (I0.ndim == 3) and (I.ndim == 3), "image size is incorrect"

        affine_trans = False
        for i in range(1, self.hyper_params['TRY_NUM']):
            if state == 0:
                tidx = np.random.choice(self.n_transforms, size=1, replace=False)[0]

                # if affine transformation is selected
                if tidx >= len(self.transform_list["pixel"]):
                    affine_trans = True
                    tidx = tidx - len(self.transform_list["pixel"])
                    I_mutated = self.transform_list["affine"][tidx](I)
                    I0_G = self.transform_list["affine"][tidx](I0)
                else:
                    I_mutated = self.transform_list["pixel"][tidx](I)
            else:
                tidx = np.random.choice(len(self.transform_list["pixel"]), size=1, replace=False)[0]
                I_mutated = self.transform_list["pixel"][tidx](I)

            I_mutated = torch.clamp(I_mutated, min=0, max=1)

            if self.f(I0, I_mutated):
                if affine_trans:
                    state = 1
                    self.info[I_mutated] = (I0_G, state)
                else:
                    self.info[I_mutated] = (I0, state)
                return I_mutated

            # if affine_trans:
            #     state = 1
            #     self.info[I_mutated] = (I0_G, state)
            #     return I_mutated
            # elif self.f(I0, I_mutated):
            #     self.info[I_mutated] = (I0, state)
            #     return I_mutated
        return I

    def saveImage(self, image, path):
        if image is not None:
            print('Saving mutated images in %s...' % path)
            save_image(image, path)

    def f(self, I, I_new):
        l0_dist = torch.sum((I - I_new) != 0)
        linf_dist = torch.max(torch.abs(I - I_new))

        if (l0_dist < self.hyper_params['alpha'] * torch.sum(I > 0)):
            return linf_dist <= 255
        else:
            return linf_dist <= self.hyper_params['beta'] * 255


if __name__ == '__main__':
    import os
    import sys
    import argparse
    import torchvision
    import gc

    import utility
    import models
    from models.mnist_models import LeNet
    import tool
    import coverage
    import constants
    # import data_loader
    from datasets import Data
    from imagenet_data import ImageNet_Data

    import signal

    def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        try:
            if engine is not None:
                engine.print_info()
                # if engine.logger is not None:
                #     engine.logger.save()
                #     engine.logger.exit()
                # if engine.criterion is not None:
                #     engine.criterion.save(args.coverage_dir + 'coverage_int.pth')
                engine.exit()
        except:
            pass
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        choices=['MNIST', 'FashionMNIST', 'SVHN', 'CIFAR10', 'CIFAR100', 'ImageNet'])
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['lenet5', 'LeNet', 'resnet18', 'resnet34', 'resnet50', 'vgg16', 'mobilenet_v2'])
    parser.add_argument('--criterion', type=str, default='NLC',
                        choices=['NLC', 'NC', 'KMNC', 'SNAC', 'NBC', 'TKNC', 'TKNP', 'CC',
                                 'LSC', 'DSC', 'MDSC', 'CDC', 'random', 'CDC_v2'])
    parser.add_argument('--seed_id', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./temp')
    parser.add_argument('--only_last', action='store_true')

    parser.add_argument('--max_testsuite_size', type=int, default=10000)

    parser.add_argument('--n_bins', type=int, default=100)
    parser.add_argument('--max_per_bin', type=int, default=10)

    # parser.add_argument('--hyper', type=str, default=None)
    base_args = parser.parse_args()

    if 'CDC' in base_args.criterion:
        constants.hyper_map[base_args.criterion]['n_class'] = constants.data_info[base_args.dataset]['n_class']
        constants.hyper_map[base_args.criterion]['n_bins'] = base_args.n_bins
        constants.hyper_map[base_args.criterion]['max_per_bin'] = base_args.max_per_bin

    args = Parameters(base_args)

    args.exp_name = ('%s-%s-%s' % (args.dataset, args.model, args.criterion))
    # args.exp_name = ('%s-%s-random' % (args.dataset, args.model))
    if args.criterion in ["NLC", "CC"] and args.only_last:
        args.exp_name = args.exp_name + "-only-last-layer"
    elif 'CDC' in args.criterion:
        args.exp_name = f"{args.exp_name}_{constants.hyper_map[args.criterion]['n_bins']}_{constants.hyper_map[args.criterion]['max_per_bin']}"
    elif args.criterion in ['LSC', 'DSC']:
        args.exp_name = f"{args.exp_name}_{constants.hyper_map[args.criterion]['ub']}_{constants.hyper_map[args.criterion]['n_bins']}"
    elif args.criterion != 'random':
        args.exp_name = f"{args.exp_name}_{constants.hyper_map[args.criterion]}"
    args.exp_name = f"{args.exp_name}_fixed_{args.max_testsuite_size}"

    args.image_dir = f'{args.output_dir}/{args.dataset}/{args.exp_name}/image/'
    args.coverage_dir = f'{args.output_dir}/{args.dataset}/{args.exp_name}/coverage/'
    args.log_dir = f'{args.output_dir}/{args.dataset}/{args.exp_name}/log/'

    print(args.__dict__)

    utility.make_path(args.image_dir)
    utility.make_path(args.coverage_dir)
    utility.make_path(args.log_dir)

    """
    python fuzz.py --dataset CIFAR10 --model resnet50 --criterion NC
    
    print(args.__dict__)

    {'model': 'resnet50', 
    'dataset': 'CIFAR10', 
    'criterion': 'NC', 
    'use_sc': False, 
    'device': device(type='cuda', index=0), 
    'num_workers': 4, 
    'batch_size': 50, 
    'mutate_batch_size': 1, 
    'nc': 3,
    'image_size': 32,
    'input_shape': (1, 32, 32, 3), 
    'num_class': 10, 
    'num_per_class': 100, 
    'input_scale': 255, 
    'noise_data': False, 
    'K': 64, 
    'batch1': 64, 
    'batch2': 16, 
    'alpha': 0.2, 
    'beta': 0.5, 
    'TRY_NUM': 50, 
    'save_every': 100, 
    'output_dir': './Fuzzer_output/', 
    'G': [(<function image_translation at 0x7ff6727cfa60>, (-5, -5)), (<function image_translation at 0x7ff6727cfa60>, (-5, 0)), (<function image_translation at 0x7ff6727cfa60>, (0, -5)), (<function image_translation at 0x7ff6727cfa60>, (0, 0)), (<function image_translation at 0x7ff6727cfa60>, (5, 0)), (<function image_translation at 0x7ff6727cfa60>, (0, 5)), (<function image_translation at 0x7ff6727cfa60>, (5, 5)), (<function image_scale at 0x7ff6727cfca0>, 0.8), (<function image_scale at 0x7ff6727cfca0>, 0.8500000000000001), (<function image_scale at 0x7ff6727cfca0>, 0.9000000000000001), (<function image_scale at 0x7ff6727cfca0>, 0.9500000000000002), (<function image_rotation at 0x7ff666f20d30>, -30), (<function image_rotation at 0x7ff666f20d30>, -29), (<function image_rotation at 0x7ff666f20d30>, -28), (<function image_rotation at 0x7ff666f20d30>, -27), (<function image_rotation at 0x7ff666f20d30>, -26), (<function image_rotation at 0x7ff666f20d30>, -25), (<function image_rotation at 0x7ff666f20d30>, -24), (<function image_rotation at 0x7ff666f20d30>, -23), (<function image_rotation at 0x7ff666f20d30>, -22), (<function image_rotation at 0x7ff666f20d30>, -21), (<function image_rotation at 0x7ff666f20d30>, -20), (<function image_rotation at 0x7ff666f20d30>, -19), (<function image_rotation at 0x7ff666f20d30>, -18), (<function image_rotation at 0x7ff666f20d30>, -17), (<function image_rotation at 0x7ff666f20d30>, -16), (<function image_rotation at 0x7ff666f20d30>, -15), (<function image_rotation at 0x7ff666f20d30>, -14), (<function image_rotation at 0x7ff666f20d30>, -13), (<function image_rotation at 0x7ff666f20d30>, -12), (<function image_rotation at 0x7ff666f20d30>, -11), (<function image_rotation at 0x7ff666f20d30>, -10), (<function image_rotation at 0x7ff666f20d30>, -9), (<function image_rotation at 0x7ff666f20d30>, -8), (<function image_rotation at 0x7ff666f20d30>, -7), (<function image_rotation at 0x7ff666f20d30>, -6), (<function image_rotation at 0x7ff666f20d30>, -5), (<function image_rotation at 0x7ff666f20d30>, -4), (<function image_rotation at 0x7ff666f20d30>, -3), (<function image_rotation at 0x7ff666f20d30>, -2), (<function image_rotation at 0x7ff666f20d30>, -1), (<function image_rotation at 0x7ff666f20d30>, 0), (<function image_rotation at 0x7ff666f20d30>, 1), (<function image_rotation at 0x7ff666f20d30>, 2), (<function image_rotation at 0x7ff666f20d30>, 3), (<function image_rotation at 0x7ff666f20d30>, 4), (<function image_rotation at 0x7ff666f20d30>, 5), (<function image_rotation at 0x7ff666f20d30>, 6), (<function image_rotation at 0x7ff666f20d30>, 7), (<function image_rotation at 0x7ff666f20d30>, 8), (<function image_rotation at 0x7ff666f20d30>, 9), (<function image_rotation at 0x7ff666f20d30>, 10), (<function image_rotation at 0x7ff666f20d30>, 11), (<function image_rotation at 0x7ff666f20d30>, 12), (<function image_rotation at 0x7ff666f20d30>, 13), (<function image_rotation at 0x7ff666f20d30>, 14), (<function image_rotation at 0x7ff666f20d30>, 15), (<function image_rotation at 0x7ff666f20d30>, 16), (<function image_rotation at 0x7ff666f20d30>, 17), (<function image_rotation at 0x7ff666f20d30>, 18), (<function image_rotation at 0x7ff666f20d30>, 19), (<function image_rotation at 0x7ff666f20d30>, 20), (<function image_rotation at 0x7ff666f20d30>, 21), (<function image_rotation at 0x7ff666f20d30>, 22), (<function image_rotation at 0x7ff666f20d30>, 23), (<function image_rotation at 0x7ff666f20d30>, 24), (<function image_rotation at 0x7ff666f20d30>, 25), (<function image_rotation at 0x7ff666f20d30>, 26), (<function image_rotation at 0x7ff666f20d30>, 27), (<function image_rotation at 0x7ff666f20d30>, 28), (<function image_rotation at 0x7ff666f20d30>, 29)], 
    'P': [(<function image_contrast at 0x7ff666f20dc0>, 0.8), (<function image_contrast at 0x7ff666f20dc0>, 1.0), (<function image_contrast at 0x7ff666f20dc0>, 1.2000000000000002), (<function image_contrast at 0x7ff666f20dc0>, 1.4000000000000001), (<function image_contrast at 0x7ff666f20dc0>, 1.6), (<function image_contrast at 0x7ff666f20dc0>, 1.8), (<function image_contrast at 0x7ff666f20dc0>, 2.0), (<function image_brightness at 0x7ff666f20e50>, 10), (<function image_brightness at 0x7ff666f20e50>, 20), (<function image_brightness at 0x7ff666f20e50>, 30), (<function image_brightness at 0x7ff666f20e50>, 40), (<function image_brightness at 0x7ff666f20e50>, 50), (<function image_brightness at 0x7ff666f20e50>, 60), (<function image_brightness at 0x7ff666f20e50>, 70), (<function image_blur at 0x7ff666f20ee0>, 1), (<function image_blur at 0x7ff666f20ee0>, 2), (<function image_blur at 0x7ff666f20ee0>, 3), (<function image_blur at 0x7ff666f20ee0>, 4), (<function image_blur at 0x7ff666f20ee0>, 5), (<function image_blur at 0x7ff666f20ee0>, 6), (<function image_blur at 0x7ff666f20ee0>, 7), (<function image_blur at 0x7ff666f20ee0>, 8), (<function image_blur at 0x7ff666f20ee0>, 9), (<function image_blur at 0x7ff666f20ee0>, 10)], 
    'S': [], 
    'save_batch': False, 
    'exp_name': 'CIFAR10-resnet50-NC', 
    'image_dir': './Fuzzer_output/CIFAR10-resnet50-NC/image/', 
    'coverage_dir': './Fuzzer_output/CIFAR10-resnet50-NC/coverage/', 
    'log_dir': './Fuzzer_output/CIFAR10-resnet50-NC/log/'}
    """
    TOTAL_CLASS_NUM = constants.data_info[args.dataset]['n_class']
    if args.dataset == 'ImageNet':
        model = torchvision.models.__dict__[args.model](pretrained=True)
        path = None
        # path = os.path.join(constants.PRETRAINED_MODELS, ('%s/%s.pth' % (args.dataset, args.model)))
        assert args.image_size == 128
        assert args.n_test_class <= 1000
    elif 'CIFAR' in args.dataset:
        model = getattr(models, args.model)(pretrained=False, num_classes=TOTAL_CLASS_NUM)
        if 'resnet' in args.model:
            model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = torch.nn.Identity()
        path = os.path.join(constants.PRETRAINED_MODELS, ('%s/%s.pth' % (args.dataset, args.model)))
        assert args.image_size == 32
        assert args.n_test_class <= 10 if args.dataset == 'CIFAR10' else args.n_test_class <= 100
    elif args.dataset == 'MNIST':
        model = getattr(models, args.model)(pretrained=False)
        path = os.path.join(constants.PRETRAINED_MODELS, ('%s/%s.pt' % (args.dataset, args.model)))
        assert args.image_size == 28
        assert args.n_test_class <= 10
        assert args.nc == 1, "number of channels incorrect"
    elif args.dataset == 'SVHN':
        # model = torchvision.models.vgg16_bn(pretrained=True)
        # model.classifier[6] = torch.nn.Linear(4096, 10, True)
        path = os.path.join(constants.PRETRAINED_MODELS, ('%s/%s.pth' % (args.dataset, args.model)))
        assert args.image_size == 32
        assert args.n_test_class <= 10
        assert args.nc == 3, "number of channels incorrect"
    elif args.dataset == 'FashionMNIST':
        model = LeNet()
        path = os.path.join(constants.PRETRAINED_MODELS, ('%s/%s.pth' % (args.dataset, args.model)))
        assert args.image_size == 28
        assert args.n_test_class <= 10
        assert args.nc == 1, "number of channels incorrect"

    # test model with pretrained weights
    # model.load_state_dict(torch.load("NeuraL-Coverage/pretrained_models/ImageNet/mobilenet_v2_pytorch_weights.pth"))
    if path is not None:
        if "CIFAR" in args.dataset and "resnet" in args.model:
            sd = torch.load(path)
            model.load_state_dict(sd["net"])
        elif args.dataset == 'SVHN':
            model = torch.load(path)
        elif args.dataset == 'FashionMNIST':
            model = torch.load(path)
        else:
            model.load_state_dict(torch.load(path))
    model.to(args.device)
    model.eval()

    if args.dataset != 'ImageNet':
        print("Evaluating model's performance on test dataset")
        print("----------------------------------------------")
        # if args.dataset == 'ImageNet':
        #     dataloader = torch.utils.data.DataLoader(
        #         ImageNet_Data(image_dir=constants.IMAGENET_JPEG_DIR,
        #                       label2index_file=constants.IMAGENET_LABEL_TO_INDEX,
        #                       split='val',
        #                       download=False,
        #                       normalise_imgs=True,
        #                       image_size=128),
        #         batch_size=512,
        #         shuffle=False, drop_last=False)
        # else:
        testdata = Data(dataset_name=args.dataset,
                    root="/data/cse87/aish/vision_datasets",
                    train=False,
                    download=True,
                    normalise_imgs=True)
        dataloader = torch.utils.data.DataLoader(
            testdata,
            batch_size=512,
            shuffle=False, drop_last=False)

        with torch.no_grad():
            correct = 0

            for idx, (img, label) in enumerate(dataloader):
                img = img.to(args.device)
                label = label.to(args.device)

                output = model(img)
                pred_label = torch.max(output, dim=1)[1]

                assert (pred_label.shape == label.shape), print("Shape mismatch.")
                # assert (torch.max(pred_label) <= args.n_class-1 and torch.min(pred_label)>=0), print("pred_label should be class ids")

                correct += torch.sum(pred_label == label).item()

                if idx == 0:
                    pred_vs_true = (pred_label == label)
                    all_pred_label = pred_label
                else:
                    pred_vs_true = torch.cat((pred_vs_true, (pred_label == label)), dim=0)
                    all_pred_label = torch.cat((all_pred_label, pred_label), dim=0)
        print("Model performance on test dataset: {}/{} = {}\n".format(correct,
                                                                       len(dataloader.dataset), 100.0 * correct / len(dataloader.dataset)))
        pred_vs_true = pred_vs_true.cpu().numpy()
        np.save(file=f"pred_vs_true_{args.dataset}_{args.model}.npy", arr=pred_vs_true)
    else:
        pred_vs_true = np.load("pred_vs_true_imagenet_val_mobilenet_v2.npy")
        print("Model performance on test dataset: {}/{} = {}\n".format(np.sum(pred_vs_true),
                                                                       pred_vs_true.shape[0], 100.0 * np.sum(pred_vs_true) / pred_vs_true.shape[0]))

    print("Creating a seed set and evaluating model on it")
    print("----------------------------------------------")

    if args.dataset == 'ImageNet':
        seed_set = ImageNet_Data(
            image_dir=constants.IMAGENET_JPEG_DIR,
            label2index_file=constants.IMAGENET_LABEL_TO_INDEX,
            split='val',
            download=False,
            pred_vs_true=pred_vs_true,
            num_class=args.n_test_class,
            samples_per_class=args.num_per_class,
            normalise_imgs=True,
            seed_save_dir='seed_set_directory',
            seed_id=args.seed_id)
    else:
        seed_set = Data(
            dataset_name=args.dataset,
            root="/data/cse87/aish/vision_datasets",
            train=False,
            download=True,
            samples_per_class=args.num_per_class,
            seed_save_dir='seed_set_directory',
            pred_vs_true=pred_vs_true,
            normalise_imgs=True,
            seed_id=args.seed_id
        )
    dataloader = torch.utils.data.DataLoader(seed_set, batch_size=512, shuffle=False, drop_last=False)

    with torch.no_grad():
        correct = 0

        for idx, (img, label) in enumerate(dataloader):
            img = img.to(args.device)
            label = label.to(args.device)

            output = model(img)
            pred_label = torch.max(output, dim=1)[1]

            assert (pred_label.shape == label.shape), print("Shape mismatch.")
            correct += torch.sum(pred_label == label).item()

            if idx == 0:
                pred_vs_true1 = (pred_label == label)
                all_pred_label = pred_label
            else:
                pred_vs_true1 = torch.cat((pred_vs_true1, (pred_label == label)), dim=0)
                all_pred_label = torch.cat((all_pred_label, pred_label), dim=0)
        pred_vs_true1 = pred_vs_true1.cpu().numpy()
    print("Model's performance on the seed set: {}/{} = {}\n".format(correct,
          len(dataloader.dataset), 100.0 * correct / len(dataloader.dataset)))
    
    """
    img_transform = torchvision.transforms.Compose([
                        torchvision.transforms.Resize(128),
                        torchvision.transforms.CenterCrop(128),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
    val_data   = torchvision.datasets.ImageNet(root="/data/cse87/aish/vision_datasets/ImageNet", split = 'val', transform=img_transform)
    print(val_data.__dict__["wnid_to_classes"])
    print(val_data.__dict__.keys())
    sys.exit()

    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=256, shuffle=False)

    print(len(val_data), len(val_loader))
    
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_img, batch_label in val_loader:
            output = model(batch_img.cuda())
            label = torch.argmax(output, dim=1)
            assert (label.shape==batch_label.shape), "shape mismatch"
            correct += torch.sum(label==batch_label.cuda()).item()
            total += batch_label.shape[0]
            print(f"correct: {correct} \t total: {total} \t accuracy: {100.0*correct/total}")
        print(f"correct: {correct} \t total: {total} \t accuracy: {correct/total}")
    sys.exit()
    """

    print("Name of the model's layers and their output shape")
    print("-------------------------------------------------")
    input_size = (1, args.nc, args.image_size, args.image_size)
    random_data = torch.randn(input_size).to(args.device)
    layer_size_dict = tool.get_layer_output_sizes(model, random_data)
    print(layer_size_dict)

    """
    python fuzz.py --dataset CIFAR10 --model resnet50 --criterion NC

    print(layer_size_dict)
    
    {'Conv2d-1': [64, 32, 32], 'Conv2d-2': [64, 16, 16], 'Conv2d-3': [64, 16, 16], 'Conv2d-4': [256, 16, 16], 
    'Conv2d-5': [256, 16, 16], 'Conv2d-6': [64, 16, 16], 'Conv2d-7': [64, 16, 16], 'Conv2d-8': [256, 16, 16], 
    'Conv2d-9': [64, 16, 16], 'Conv2d-10': [64, 16, 16], 'Conv2d-11': [256, 16, 16], 'Conv2d-12': [128, 16, 16], 
    'Conv2d-13': [128, 8, 8], 'Conv2d-14': [512, 8, 8], 'Conv2d-15': [512, 8, 8], 'Conv2d-16': [128, 8, 8], 
    'Conv2d-17': [128, 8, 8], 'Conv2d-18': [512, 8, 8], 'Conv2d-19': [128, 8, 8], 'Conv2d-20': [128, 8, 8], 
    'Conv2d-21': [512, 8, 8], 'Conv2d-22': [128, 8, 8], 'Conv2d-23': [128, 8, 8], 'Conv2d-24': [512, 8, 8], 
    'Conv2d-25': [256, 8, 8], 'Conv2d-26': [256, 4, 4], 'Conv2d-27': [1024, 4, 4], 'Conv2d-28': [1024, 4, 4], 
    'Conv2d-29': [256, 4, 4], 'Conv2d-30': [256, 4, 4], 'Conv2d-31': [1024, 4, 4], 'Conv2d-32': [256, 4, 4], 
    'Conv2d-33': [256, 4, 4], 'Conv2d-34': [1024, 4, 4], 'Conv2d-35': [256, 4, 4], 'Conv2d-36': [256, 4, 4], 
    'Conv2d-37': [1024, 4, 4], 'Conv2d-38': [256, 4, 4], 'Conv2d-39': [256, 4, 4], 'Conv2d-40': [1024, 4, 4], 
    'Conv2d-41': [256, 4, 4], 'Conv2d-42': [256, 4, 4], 'Conv2d-43': [1024, 4, 4], 'Conv2d-44': [512, 4, 4], 
    'Conv2d-45': [512, 2, 2], 'Conv2d-46': [2048, 2, 2], 'Conv2d-47': [2048, 2, 2], 'Conv2d-48': [512, 2, 2], 
    'Conv2d-49': [512, 2, 2], 'Conv2d-50': [2048, 2, 2], 'Conv2d-51': [512, 2, 2], 'Conv2d-52': [512, 2, 2], 
    'Conv2d-53': [2048, 2, 2], 
    'Linear-1': [10]}

    

    # using my datasets.py file instead of data_loader.py
    # --------------------------------------------------
    # if args.dataset == 'CIFAR10':
    #     data_set = data_loader.CIFAR10FuzzDataset(args, split='test')
    # elif args.dataset == 'ImageNet':
    #     data_set = data_loader.ImageNetFuzzDataset(args, split='val')

    # TOTAL_CLASS_NUM, train_loader, test_loader, seed_loader = data_loader.get_loader(args)

    # image_list, label_list = data_set.build()


    image_list: list of tensors. Each image is read as an RGB image, transformed using torchvision.transforms and stored as a tensor. Images are not normalized.
    label_list: list of tensors.
    image_list[0].shape = torch.Size([3, 32, 32]) 
    label_list[0].shape = torch.Size([])

    # saving seed set by loading and saving images to have consistency with saving generated images.
    # args.seed_dir = args.output_dir + args.exp_name + '/seed_set/'
    # utility.make_path(args.seed_dir)


    # using my datasets.py file instead of data_loader.py
    # --------------------------------------------------
    # to_image = transforms.ToPILImage()
    # for img_id, img_tensor in enumerate(image_list):
    #     img = to_image(img_tensor)
    #     img.save(args.seed_dir+f"image_{img_id}.png")

    # image_numpy_list = data_set.to_numpy(image_list)
    # label_numpy_list = data_set.to_numpy(label_list, False)

    image_numpy_list: list of numpy arrays. image tensors are converted to numpy and transposed.
    label_numpy_list: list of numpy scalars.
    image_numpy_list[0].shape = (32, 32, 3)
    """
    # recreating seed_set as normalise_imgs = False because the images have to be mutated
    print("\nCreating same seed set again with normalise_imgs=False")
    print("----------------------------------------------")
    if args.dataset == 'ImageNet':
        seed_set = ImageNet_Data(
            image_dir=constants.IMAGENET_JPEG_DIR,
            label2index_file=constants.IMAGENET_LABEL_TO_INDEX,
            split='val',
            download=False,
            pred_vs_true=pred_vs_true,
            num_class=args.n_test_class,
            samples_per_class=args.num_per_class,
            normalise_imgs=False,
            seed_save_dir='seed_set_directory',
            seed_id=args.seed_id
        )
    else:
        seed_set = Data(
            dataset_name=args.dataset,
            root="/data/cse87/aish/vision_datasets",
            train=False,
            download=False,
            seed_save_dir='seed_set_directory',
            normalise_imgs=False,
            seed_id=args.seed_id
        )
    image_list, label_list = seed_set.build_and_shuffle()

    # print(seed_set.selected_idx[:20])
    print(len(image_list), len(label_list), type(image_list[0]), type(label_list[0]), label_list[0])

    # image_numpy_list = seed_set.to_numpy(image_list)
    # label_numpy_list = label_list

    # del image_list
    # del label_list
    gc.collect()

    if args.criterion != 'random':
        if args.use_sc:
            criterion = getattr(coverage, args.criterion)(model, layer_size_dict,
                                                        hyper=constants.hyper_map[args.criterion], min_var=1e-5, num_class=TOTAL_CLASS_NUM)
        else:
            criterion = getattr(coverage, args.criterion)(model, layer_size_dict, hyper=constants.hyper_map[args.criterion], only_last=args.only_last)

        print("\nCreating a subset of train dataset for building and computing the initial coverage")
        print("----------------------------------------------")
        if args.dataset == 'ImageNet':
            build_set = ImageNet_Data(
                image_dir=constants.IMAGENET_JPEG_DIR,
                label2index_file=constants.IMAGENET_LABEL_TO_INDEX,
                split='val',
                download=False,
                pred_vs_true=None,
                num_class=args.n_test_class,
                samples_per_class=args.num_per_class,
                normalise_imgs=True)
        else:
            build_set = Data(dataset_name=args.dataset,
                            root="/data/cse87/aish/vision_datasets",
                            train=True,
                            download=True,
                            samples_per_class=args.num_per_class,
                            normalise_imgs=True
                            )

        train_loader = torch.utils.data.DataLoader(build_set, batch_size=512, shuffle=False, drop_last=False)
        criterion.build(train_loader)
        start_time = time.time()
        criterion.assess(train_loader)
        end_time = time.time()
        print("Coverage on training data: ", criterion.current, "\n")
        print("Time taken: ", end_time - start_time, "\n")

        # with open(f'coverage_training_time_{args.dataset}.txt', 'a') as f:
        #     f.write(
        #         f'${args.criterion}$ & ${constants.hyper_map[args.criterion]}$ &  ${criterion.current}$ & ${end_time-start_time}$')
        #     f.write("\n")

        # access_loader = torch.utils.data.DataLoader(
        #     Data(dataset_name=args.dataset,
        #          root="/data/cse87/aish/vision_datasets",
        #          train=False,
        #          download=True,
        #          samples_per_class=1000,
        #          normalise_imgs=True
        #          ),
        #     batch_size=512, shuffle=False, drop_last=False)
        # criterion.assess(access_loader)
        # print("Coverage on training data: ", criterion.current, "\n")

        '''
        # coverage cannot be initialised from the training data
        # if args.criterion not in ['LSC', 'DSC', 'MDSC']:

        For LSC/DSC/MDSC/CC/TKNP, initialization with training data is too slow (sometimes may
        exceed the memory limit). You can skip this step to speed up the experiment, which
        will not affect the conclusion because we only compare the relative order of coverage
        values, rather than the exact numbers.
        '''
        # exit()

        initial_coverage = copy.deepcopy(criterion.current)
        print('Initial Coverage: %f' % initial_coverage, "\n")

        if args.criterion in ['LSC', 'DSC']:
            print(criterion.train_upper_bound)
    else:
        criterion = 'random'

    engine = Fuzzer(args, criterion)
    engine.run(image_list, label_list)
    engine.exit()
