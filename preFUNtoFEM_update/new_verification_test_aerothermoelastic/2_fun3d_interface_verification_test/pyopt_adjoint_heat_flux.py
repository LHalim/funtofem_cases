#!/usr/bin/env python

from __future__ import print_function

from pyfuntofem.model  import *
from pyfuntofem.driver import *
from pyfuntofem.fun3d_interface_verification_test import *

from tacs_model import wedgeTACS
from mpi4py import MPI
import os


class wedge_adjoint(object):
    """
    -------------------------------------------------------------------------------
    TOGW minimization
    -------------------------------------------------------------------------------
    """

    def __init__(self):
        print('start')

        # cruise conditions
        self.v_inf = 171.5                          # freestream velocity [m/s]
        self.rho = 0.01841                          # freestream density [kg/m^3]
        self.cruise_q = 12092.5527126               # dynamic pressure [N/m^2]
        self.grav = 9.81                            # gravity acc. [m/s^2]
        self.thermal_scale = 0.5 * self.rho * (self.v_inf)**3

        # Set up the communicators
        n_tacs_procs = 1

        comm = MPI.COMM_WORLD
        self.comm = comm
        print('set comm')

        world_rank = comm.Get_rank()
        if world_rank < n_tacs_procs:
            color = 55
            key = world_rank
        else:
            color = MPI.UNDEFINED
            key = world_rank
        self.tacs_comm = comm.Split(color,key)
        print('comm misc')

        # Set up the FUNtoFEM model for the TOGW problem
        self._build_model()
        print('built model')
        self.ndv = len(self.model.get_variables())
        print("ndvs: ",self.ndv)

        # instantiate TACS on the master
        solvers = {}
        solvers['flow'] = Fun3dInterface(self.comm,self.model,flow_dt=1.0)#flow_dt=0.1
        solvers['structural'] = wedgeTACS(self.comm,self.tacs_comm,self.model,n_tacs_procs)

        # L&D transfer options
        transfer_options = {'analysis_type': 'aerothermoelastic','scheme': 'meld', 'thermal_scheme': 'meld'}

        # instantiate the driver
        self.driver = FUNtoFEMnlbgs(solvers,self.comm,self.tacs_comm,0,self.comm,0,transfer_options,model=self.model)

        # Set up some variables and constants related to the problem
        self.cruise_lift   = None
        self.cruise_drag   = None
        self.num_con = 1
        self.mass = None

        self.var_scale        = np.ones(self.ndv,dtype=TransferScheme.dtype)
        self.struct_tacs = solvers['structural'].assembler

    def _build_model(self):

        thickness = 0.015

        # Build the model
        model = FUNtoFEMmodel('wedge')
        plate = Body('plate',analysis_type='aerothermoelastic',group=0,boundary=1)
        plate.add_variable('structural',Variable('thickness',value=thickness,lower = 0.01, upper = 0.1))
        model.add_body(plate)

        steady = Scenario('steady',group=0,steps=100)
        steady.set_variable('aerodynamic',name='AOA',value=0.0,lower=-15.0,upper=15.0)
        temp = Function('temperature',analysis_type='structural') 
        steady.add_function(temp)

        lift = Function('cl',analysis_type='aerodynamic')
        steady.add_function(lift)

        drag = Function('cd',analysis_type='aerodynamic')
        steady.add_function(drag)

        model.add_scenario(steady)

        self.model = model

    def verification_test(self):

        steady = self.model.scenarios[0]
        bodies = self.model.bodies

        fail = self.driver.solve_forward()
        print('FINISHED FORWARD')
        fail = self.driver.solve_adjoint()
        print('FINISHED ADJOINT')
        #fail = self.driver.solve_forward()
        #print('FINISHED FORWARD')

        self.driver.solvers['flow'].adjoint_test(steady, bodies, epsilon=1e-6) 
        print('FINISHED ADJOINT TEST')

###############################################################################
dp = wedge_adjoint()
print('Created Object')

print('VERIFICATION TEST')
dp.verification_test()

print('FINISHED')
