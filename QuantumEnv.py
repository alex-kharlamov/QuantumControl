from qutip import *
import numpy as np
import matplotlib.pyplot as pp
import time
import scipy.fftpack as fft
import numpy.linalg
import scipy.sparse as sparse
import scipy.sparse.linalg
from math import *
from copy import copy

import gym
from gym import error, spaces, utils
from gym.utils import seeding

class qubitParams:
    C = None
    Ic = None
    def __init__(self):
        self.Cq = 1.01e-13
        self.Ic = 30e-9
        self.Cx = 1e-16
        self.Cg = 1e-14

class QuantumEnv(gym.Env):
    def __init__(self):
        
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        
        
        self.e = 1
        self.hbar = 1
        self.hplank = self.hbar * 2 * pi
        self.Fi0 = 2*pi*self.hbar/(2*self.e)

        self.nmax = 5


        self.qp = qubitParams()
        self.Cx = self.qp.Cx
        self.Cg = self.qp.Cg
        self.Cq = self.qp.Cq
        self.Zr = 50
        self.extFlux = -.58
        self.extVoltage = 0.0
        self.omega = 6

        self.Q = 1e3
        self.T1 = 1e9

        self.Cr = 1/(self.omega*1e9*self.Zr)
        (ev, phis, evec) = self.__qubitSpectrum(self.qp, 100, self.extFlux, self.extVoltage)
        self.epsilon  = ev[1] - ev[0]
        #print('Epsilon = %e, detuning = %.2f' % (epsilon, (epsilon - omega) / g))
        self.g = self.Cg/(2*sqrt(self.Cq*self.Cr))*sqrt(self.omega*self.epsilon)

        self.Aplus = tensor(create(self.nmax),qeye(2))
        self.Aminus = tensor(destroy(self.nmax), qeye(2))
        self.eField = self.Aminus + self.Aplus

        self.SigmaX = tensor(qeye(self.nmax), sigmax())
        self.SigmaZ = tensor(qeye(self.nmax), sigmaz())
        self.SigmaP = tensor(qeye(self.nmax), sigmap())
        self.SigmaN = tensor(qeye(self.nmax), sigmam())
        self.SigmaPopulation = tensor(qeye(self.nmax), (sigmaz() + qeye(2)) / 2)
        
        self.H = self.hplank * self.omega  * (self.Aplus*self.Aminus) + 0.5*self.hplank*self.epsilon*self.SigmaZ + self.hplank*self.g*self.eField*self.SigmaX



        # default frequency

        self.wDrive = self.epsilon
        self.times = np.linspace(0, 100, 8000) # control period (8000 is 8ns)

        # initial state

        qubitStartState = fock(2, 1)
        qubitStartState = qubitStartState.unit()
        qubitStartState = qubitStartState*qubitStartState.dag()
        resonatorStartState = fock(self.nmax, 0)
        resonatorStartState = resonatorStartState*resonatorStartState.dag()

        # target state

        qubitTgtState = fock(2, 0)
        qubitTgtState = qubitTgtState.unit()
        qubitTgtState = qubitTgtState*qubitTgtState.dag()
        
        self.qubitStartState = copy(qubitStartState)
        self.qubitTgtState = copy(qubitTgtState)
        self.resonatorStartState = copy(resonatorStartState)
        
        self.state = copy(self.qubitStartState)
        self.time_stamp = 0
        self.max_stamp = 10
        self.action_steps = []
        
        
        self.seed()
        self.reset()
        
    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        self.time_stamp += 1
        if self.time_stamp >= self.max_stamp:
            episode_over = True
        else:
            episode_over = False
            
        self.action_steps.append(action)
        reward, new_state = self.__qubit_eval(np.array(self.action_steps))
        self.state = new_state
        state_data = self.__state_data()
        return state_data, reward, episode_over, None
    
    def reset(self):
        self.state = copy(self.qubitStartState)
        self.time_stamp = 0
        self.action_steps = []
        return self.__state_data()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def render(self, mode='human', close=False):
        self.state
    
    def __state_data(self):
        tmp = self.state.data.toarray()
        real = tmp.real.flatten()
        img = tmp.imag.flatten()
        return np.hstack((real, img))
    
    def __qubitSpectrum(self, qubitParams, N, extFlux, extVoltage):

        step = 2*pi/N


        h = 6.626e-34;
        hbar = h/2/pi
        e = 1.6e-19;
        Fi0 = h/2/e;

        #L = qubitParam.L/cos(extFlux*pi/Fi0);
        C = qubitParams.Cq + qubitParams.Cg + qubitParams.Cx;

        #Ic =  qubitParams.Ic*abs(cos(pi*extFlux/Fi0))
        Ic =  qubitParams.Ic * abs(cos(pi*extFlux))

        Ej = (2 * Ic * Fi0 / 2) / (h*1e9) # GHz
        Ec = ((1 * e)**2 / (2 * C)) / (h*1e9) # GHz
        ng = self.extVoltage * qubitParams.Cx / (2 * e);
        #print('Ec = %e, Ej = %e, Ej/Ec = %f, E01 = %e,' % (Ec, Ej, Ej/Ec, sqrt(8*Ec*Ej) - Ec))
        #print(ng)



        phi = np.linspace(-pi, pi, N+1);
        phi = phi[0 : -1];

        def alpha(phi):
            return -4*Ec
        def beta(phi):
            return 0
        def gamma(phi):
           return -cos(phi)*Ej

        diagCentr = np.zeros([N], dtype='complex')
        diagUp = np.zeros([N], dtype='complex')
        diagDown = np.zeros([N], dtype='complex')

        for i in range(N):
            diagCentr[i] = gamma(phi[i])  - 2*alpha(phi[i])/(step*step)
            diagUp[i] = alpha(phi[i])/(step*step) + beta(phi[i])/2/step
            diagDown[i] = alpha(phi[i])/(step*step) - beta(phi[i])/2/step

        phasefactor = np.exp(1j*ng*pi)


        sm = sparse.diags([[np.conj(phasefactor)*diagUp[-1]], diagDown[1:], diagCentr, diagUp[0: -1], [phasefactor*diagDown[1]]], [-N + 1, -1, 0, 1, N -1])
        sm = sm.toarray();
        (ev, evec) = np.linalg.eigh(sm)
        #sparse.linalg.eigs(sm, 3, which='SM')

        return ev, phi, evec

    def _dummyExternalDrive(self, t, start, target):
        return 1e-5*sin(2*pi*t*self.wDrive)


    def _ramseyExternalDrive(self, t, start, target):
        tint = t
        ramseyDelay = 4
        pi4Time = 3
        if tint < pi4Time:
            return 0.1*sin(2*pi*t*self.wDrive)
        tint = tint - pi4Time

        if tint < ramseyDelay:
            return 0.0
        tint = tint - ramseyDelay

        if tint < pi4Time:
            return 0.1*sin(2*pi*t*self.wDrive)
        tint = tint - pi4Time
        return 0


    def __qubit_eval(self, externalDriveLevels):
        Ht = [self.e* self.Cx *(self.Cq + self.Cg) / (self.Cq * self.Cg * 4.125e-6) * self.SigmaX, externalDriveLevels]

        gamma = self.omega*0.69/self.Q
        temp = 0.0 # Temperature, [GHz]

        cOps = [self.Aminus*sqrt(gamma*(1+temp/self.omega)), 
                self.Aplus*sqrt(gamma*temp/self.omega),
                self.SigmaP*sqrt(temp/self.epsilon/self.T1),
                self.SigmaN*sqrt((1+temp/self.epsilon)/self.T1)]
        if self.time_stamp > 2:
            options = Options(store_states=True, rhs_reuse=True)
        else:
            options = Options(store_states=True)
            
        result = mesolve([self.H, Ht], tensor(self.resonatorStartState, self.qubitStartState), 
            self.times[:self.time_stamp], cOps, 
            [self.eField, self.Aplus*self.Aminus, self.SigmaZ, self.SigmaX, self.SigmaPopulation],
            options=options)
        # test

        end_dm = result.states[-1]
        end_dm = end_dm.ptrace(1)
        loss = fidelity(end_dm, self.qubitTgtState)
        return loss, end_dm