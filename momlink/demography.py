import numpy as np
import csv

class Demography(object):
    def __init__(self, fn=None, popsize=None):
        if fn is None and popsize is None:
            raise InvalidDemography('Please specify demography file or fixed population size.')
        elif popsize is not None and fn is not None:
            raise InvalidDemography('Please specify either a demography file or a fixed population size but not both.')
        elif popsize is None:
            self.interval_type = []
            self.event_times = [] 
            self.init_pop_sizes = []
            self.exp_rate = []
            self.parse_dem(fn)
            self.n0 = self.init_pop_sizes[0]
            self.last_time = self.event_times[-1]
            self.seg_num = len(self.interval_type)
        elif fn is None:
            self.interval_type = ['setSize']
            self.event_times = [0] 
            self.init_pop_sizes = [popsize]
            self.n0 = self.init_pop_sizes[0]
            self.last_time = self.event_times[-1]
            self.seg_num = len(self.interval_type)

    def parse_dem(self, fn):
        # Input order: interval type, start time, popsize/scale/rate
        with open(fn) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            
            for i, line in enumerate(csv_reader):

                # Set interval type
                self.interval_type.append(line[0].strip())
                time = float(line[1].strip())
                # Make sure initial interval specifies the initial population with a 'setSize' statment
                
                if i == 0 and self.interval_type[0] != 'setSize':
                    raise IndentationError('First line of demography must specify the initial population size using a "setSize" epoch type.')
                elif i > 1 and time < self.event_times[-1]:
                    raise InvalidDemography('Event times must be in increasing order.')

                # Set event time
                try: 
                    self.event_times.append(time)
                except ValueError:
                    raise InvalidDemography('Event times must be numeric.')
                
                # Set population size
                if self.interval_type[-1] == 'setSize':
                    try:
                        self.init_pop_sizes.append(int(line[2].strip()))
                        prev_size = self.init_pop_sizes[-1]
                        self.exp_rate.append(0.)
                    except ValueError:
                        raise InvalidDemography('Population sizes must be specified in integers.')
                elif self.interval_type[-1] == 'reSize':
                    try:
                        frac = float(line[2].strip())
                        self.exp_rate.append(0.)
                    except ValueError:
                        raise InvalidDemography('reSize parameters must be numeric.')

                    if frac > 1 or frac < 0:
                        raise InvalidDemography('reSize parameters must be between 0.0 and 1.0.')
                    prev_size = prev_size * frac
                    self.init_pop_sizes.append(prev_size)
                elif self.interval_type[-1] == 'expGrow':
                    try:
                        frac = float(line[2].strip())
                        self.init_pop_sizes.append(prev_size)
                        self.exp_rate.append(frac)
                    except ValueError:
                        raise InvalidDemography('Growth rate must be numeric.')
                    int_len = self.event_times[-1] - self.event_times[-2]
                    prev_size = prev_size * np.exp(frac * int_len)
                else:
                    raise InvalidDemography('Must specify "setSize", "reSize", or "expGrow" for interval type.')
                
    def initialize(self):
        self.time = 0
        self.current_seg = self.interval_type[0]
        self.current_seg_num = 0
        self.current_pop_size = self.init_pop_sizes[0]

    def get_popsize_at(self, gen):
        epoch = np.searchsorted(self.event_times, gen, side='right')-1
        current_seg = self.interval_type[epoch]
        if current_seg in ['setSize', 'reSize']:
            out = self.init_pop_sizes[epoch]
        elif current_seg == 'expGrow':
            out = self.init_pop_sizes[epoch]*np.exp(self.exp_rate[epoch]*(gen - self.event_times[epoch]))
        return out


class InvalidDemography(Exception):
    """Raised when invalid demography is given."""
    pass