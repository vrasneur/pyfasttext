#!/usr/bin/env python3

from collections import OrderedDict, namedtuple

import numpy as np
import bayesopt

from sklearn.metrics import f1_score
from pyfasttext import FastText

class FastTextBayesOpt(object):
    TRAIN_PATH = '/path/to/dataset.train'
    VALID_PATH = '/path/to/dataset.valid'
    TEST_PATH = '/path/to/dataset.test'
    VECTORS_PATH = '/path/to/unsupmodel.vec'
    MODEL_PATH = '/path/to/supmodel'
    
    def __init__(self, noisy=False, metrics_fun=None, verbose=True):
        self.noisy = noisy
        self.metrics_fun = self.metrics_fun if metrics_fun is None else metrics_fun
        self.verbose = verbose
        self.y_true = self.generate_y_true()
        self.bounds = self.generate_bounds()
        self.hyperparams = self.generate_bayesopt_hyperparams()

    # use a weighted F1 score because dataset may be imbalanced
    def metrics_fun(self, y_pred):
        return f1_score(self.y_true, y_pred, average='weighted')

    def generate_bayesopt_hyperparams(self):
        params = {'l_type': 'L_MCMC',
                  # TODO: better estimates
                  'noise': 1e-10 if not self.noisy else 0.01,
                  'verbose_level': 4}
        params['noise'] = 
        return params
        
    @classmethod
    def generate_y_true(cls):
        model = FastText()
        return model.extract_classes(cls.VALID_PATH)
        
    @staticmethod
    def generate_bounds():
        Bound = namedtuple('Bound', ['low', 'high', 'fun'])
        bounds = OrderedDict()
        bounds['epoch'] = Bound(1, 50, int)
        bounds['lr'] = Bound(0.025, 1.5, float)
        bounds['bucket'] = Bound(200000, 16000000, int)
        bounds['neg'] = Bound(50, 5000, int)
        bounds['minn'] = Bound(0, 10, int)
        bounds['maxn'] = Bound(0, 10, int)
        bounds['wordNgrams'] = Bound(0, 10, int)
        bounds['pretrainedVectors'] = Bound(0.0, 1.0, lambda x: x > 0.5)
        return bounds
      
    @property
    def lower_bound(self):
        return np.array([bound.low for bound in self.bounds.values()], dtype=float)

    @property
    def upper_bound(self):
        return np.array([bound.high for bound in self.bounds.values()], dtype=float)
   
    @property
    def dim(self):
        return len(self.bounds)
    
    def generate_fastText_hyperparams(self, params):
        kwargs = OrderedDict()
        for (key, bound), param in zip(self.bounds.items(), params):
            if key == 'pretrainedVectors':
                if bound.fun(param):
                    kwargs[key] = self.VECTORS_PATH
            else:
                kwargs[key] = bound.fun(param)

        if not self.noisy:
            kwargs['thread'] = 1
                
        return kwargs

    def train(self, params, test_path):
        self.message('parameters:', params)
        try:
            kwargs = self.generate_fastText_hyperparams(params)
            self.message('fastText parameters:')
            for key, val in kwargs.items():
                self.message(' ', key, ':', val)

            model = FastText()
            model.supervised(input=self.TRAIN_PATH, output=self.MODEL_PATH, **kwargs)
            y_pred = [item[0] for item in model.predict_file(test_path)]
            acc = self.metrics_fun(y_pred)
            self.message('metrics:', acc)
            return (1.0 - acc)
        except Exception as exc:
            self.message('failed to train!')
            self.message(exc)
            return 1.0

    def train_valid(self, params):
        return self.train(params, self.VALID_PATH)
        
    def train_test(self, params):
        return self.train(params, self.TEST_PATH)
        
    def optimize(self):
        return bayesopt.optimize(self.train_valid, self.dim, self.lower_bound, self.upper_bound, self.hyperparams)

    def message(self, msg, *args, **kwargs):
        if self.verbose:
            print('[*] ', msg, *args, **kwargs)

def main():
    fopt = FastTextBayesOpt(noisy=True)
    res = fopt.optimize()
    fopt.train_test(res)
    print('results:', res)

if __name__ == '__main__':
    main()
