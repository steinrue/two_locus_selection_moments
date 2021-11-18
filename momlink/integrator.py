import numpy as np
import pdb
from scipy import optimize

class Integrator(object):
    def __init__(self, func, clip=True, renorm=True):
        self.func = func
        self.clip = clip
        self.renorm = renorm

    def solve_ivp_rk45(self, y0, T, tol=10**-11, initial_step_size=.2, t0=0, zero_thresh=0, min_step_size=10**-6, time_points=None, num_points=1):
        yn = np.array(y0)
        t = t0
        h_last = initial_step_size
        eps = tol
        i = 0
        y_traj = []
        t_traj = []
        last_save = 0
        if not time_points:
            if num_points == 1:
                time_points = [T]
            else:
                time_points = list(np.linspace(t0, T, num_points))
        next_save = 0
        for time in time_points: 
            if time==t0:
                t_traj.append(t)
                y_traj.append(yn)
            while t < time:
                h = max(h_last, min_step_size) 
                h = min(h, time-t)
                if self.clip:
                    yn[yn < zero_thresh] = 0.
                    yn[yn > 1-zero_thresh] = 1.
                if self.renorm:
                    yn /= np.sum(yn)
                [ynp1_1, ynp1_2] = self.rkstep(h, t, yn)
                R = np.linalg.norm(ynp1_1-ynp1_2)
                if R > 0:
                    dlt = .84*(eps/R)**(1/4)
                else:
                    dlt = 2
                if R <= eps or h <= min_step_size:
                    h_last = h
                    t = t + h
                    yn = ynp1_1
                    if t >= time:
                        t_traj.append(t)
                        y_traj.append(yn)                   
                    i = i + 1
                    print('Step: ' + str(i) + ' at t = ' + str(t) + ' of ' + str(T) + '                                ', end='\r')
                if h > min_step_size:
                    h_last = dlt*h_last
        return np.array(t_traj), np.array(y_traj)

    def rkstep(self, h, t, yn):
        k1 = h*self.func(t, yn)
        k2 = h*self.func(t+h/4, yn+k1/4)
        k3 = h*self.func(t+3*h/8, yn+3*k1/32+9*k2/32)
        k4 = h*self.func(t+12*h/13, yn+1932*k1/2197-7200*k2/2197+7296*k3/2197)
        k5 = h*self.func(t+h, yn+439*k1/216-8*k2+3680*k3/513-845*k4/4104)
        k6 = h*self.func(t+h/2, yn-8*k1/27+2*k2-3544*k3/2565+1859*k4/4104-11*k5/40)

        step1 = 25*k1/216+1408*k3/2565+2197*k4/4104-k5/5
        step2 = 16*k1/135+6656*k3/12825+28561*k4/56430-9*k5/50+2*k6/55
        if self.clip:
            step1[np.logical_and(yn<=0, step1 < 0)] = 0
            step2[np.logical_and(yn<=0, step2 < 0)] = 0
            step1[np.logical_and(yn>=1, step1 > 0)] = 0
            step2[np.logical_and(yn>=1, step2 > 0)] = 0
        ynp1_1 = yn + step1
        ynp1_2 = yn + step2
        return ynp1_1, ynp1_2