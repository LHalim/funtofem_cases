#!/usr/bin/env python
"""
This file is part of the package FUNtoFEM for coupled aeroelastic simulation
and design optimization.

Copyright (C) 2015 Georgia Tech Research Corporation.
Additional copyright (C) 2015 Kevin Jacobson, Jan Kiviaho and Graeme Kennedy.
All rights reserved.

FUNtoFEM is licensed under the Apache License, Version 2.0 (the "License");
you may not use this software except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import print_function

from pyfuntofem.model  import *
from pyfuntofem.driver import *
from pyfuntofem.fun3d_interface import *
#from pyfuntofem.massoud_body import *

from tacs_model import wedgeTACS
#from pyOpt import Optimization,SLSQP
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
        self.v_inf = 171.5                  # freestream velocity [m/s]
        self.rho = 0.01841                  # freestream density [kg/m^3]
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
        self.driver = FUNtoFEMnlbgs_aerothermoelastic(solvers,self.comm,self.tacs_comm,0,self.comm,0,transfer_options,model=self.model)

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
        plate = Body('plate',group=0,boundary=1)
        plate.add_variable('structural',Variable('thickness',value=thickness,lower = 0.01, upper = 0.1))
        model.add_body(plate)

        steady = Scenario('steady',group=0,steps=100)
#        steady.add_variable('structural',Variable('thickness',value=thickness,lower = 0.01, upper = 0.1))
        steady.set_variable('aerodynamic',name='AOA',value=0.0,lower=-15.0,upper=15.0)
        #steady.set_variable('aerodynamic',name='thermal scale',value=self.thermal_scale, lower=0.0,upper=1e12)
        temp = Function('temperature',analysis_type='structural') #temperature
        steady.add_function(temp)

        lift = Function('cl',analysis_type='aerodynamic')
        steady.add_function(lift)

        drag = Function('cd',analysis_type='aerodynamic')
        steady.add_function(drag)

        model.add_scenario(steady)

        self.model = model

    def eval_objcon(self,x):
        #x_vec = self.struct_tacs.createVec()
        #x_arr = x_vec.getArray()
        #x_arr[:] = x#*self.var_scale
        #x[15] += 1e-20j
        fail = 0
        var = x*self.var_scale

        self.model.set_variables(var)

        variables = self.model.get_variables()

        #shape_dv = []
        #for var in variables:
        #    if 'shape' in var.name:
        #        shape_dv.append(var.value)

        #thickness_dv = []
        #for var in variables:
        #    if 'thickness' in var.name:
        #        thickness_dv.append(var.value)



        ########################### Simulations ################################
        # Simulate the maneuver condition
        fail = self.driver.solve_forward()#steps = 1)
        if fail == 1:
            print("simulation failed")
            return 0.0, 0.0, fail

        functions = self.model.get_functions()

        #self.cruise_drag = functions[1].value# * 2.0 * self.cruise_q
        #self.cruise_lift = functions[2].value# * 2.0 * self.cruise_q
        #self.maneuver_ks   = functions[3].value
        #self.maneuver_lift = functions[4].value * 2.0 * self.maneuver_q

        ########################## Objective evaluation ########################

        # Get the wing weight from TACS
        temp = functions[0].value
        #self.w_wing = mass * self.grav

        # total weight
        #W2 = 2.0 * self.w_wing + self.w_fixed + self.w_reservefuel + self.w_secondary
        #self.W2 = W2

        # form the final objective (TOGW) [W2 + fuel]
        #TOGW = W2 * np.exp( self.range * self.tsfc / self.v_inf *
        #                   (self.cruise_drag/self.cruise_lift)   )
        #obj = self.c1 * TOGW
        obj = temp
#        print('Forward Temp: ',temp)
#        print('Forward Drag: ',self.cruise_drag)

        #print('obj: ',obj)
        ####################### Constraints evaluations ########################
        con = np.zeros(self.num_con,dtype=TransferScheme.dtype)

        # lift trim constraints (num=2)
        #cruise_weight = 0.5 * (TOGW + W2)
        #con[0] = self.c3 * (cruise_weight - self.cruise_lift)

        #maneuver_weight = 0.5 * (TOGW + W2)
        #con[1] = self.c3 * (2.5 * maneuver_weight - self.maneuver_lift)


        # fixed area (num=1)
        #area = self.eval_area(shape_dv)
        #con[2] = self.c4 * ( area - self.area0 )

        # KS failure constraint (num=1)
        #con[3] = self.c2 * (self.safety_factor * self.maneuver_ks - 1.0 )

        # thickness smooth variation (num = 2*(187-3) )
        #con[4:] = self.c5 * self.eval_smoothness(thickness_dv)
        print('variables:')
        for i in range(self.ndv):
            print(variables[i].name, variables[i].value)

        return obj, con, fail

    def eval_objcon_grad(self,x):#,x, obj, con):

        var = x*self.var_scale
        self.model.set_variables(var)


        variables = self.model.get_variables()

        shape_dv = []
        for var in variables:
            if 'shape' in var.name:
                shape_dv.append(var.value)

        thickness_dv = []
        for var in variables:
            if 'thickness' in var.name:
                thickness_dv.append(var.value)

        fail = self.driver.solve_adjoint()
        grads = self.model.get_function_gradients()
        funcs = self.model.get_functions()

        if self.comm.Get_rank()==0:
            for i, func in enumerate(funcs):
                print("Func ", func.name, " ", funcs[i].value)
                for j, var in enumerate(variables):
                    print("%s %s %s %s %s %.30E" % ("Grad ", func.name, "Var: ", var.name, " ", grads[i][j]))
#                    print("Grad ", func.name, "Var: ", var.name, " ", grads[i][j])

        cruise_lift_grad = np.array(grads[1][:])# * 2.0 * self.cruise_q
        cruise_drag_grad = np.array(grads[2][:])# * 2.0 * self.cruise_q
        temp_grad = np.array(grads[0][:])

        #maneuver_ks_grad = np.array(grads[3][:])

        #maneuver_lift_grad = np.array(grads[4][:]) * 2.0 * self.maneuver_q



        ########################## Objective Gradient ##########################
        g = np.zeros((3,self.ndv),dtype=TransferScheme.dtype)
        g[0,:] = temp_grad
        g[1,:] = cruise_lift_grad
        g[2,:] = cruise_drag_grad
        #dW2dx = 2.0 * mass_grad * self.grav

        # TOGW gradient
        #TOGW_grad = np.zeros(self.ndv,dtype=TransferScheme.dtype)

        #expon = np.exp(self.range * self.tsfc / self.v_inf *
        #                    self.cruise_drag / self.cruise_lift )

        #TOGW_grad += dW2dx * expon

        #TOGW_grad += ( self.W2 * expon * self.range * self.tsfc / self.v_inf
        #             * cruise_drag_grad / self.cruise_lift                 )

        #TOGW_grad += ( self.W2 * expon * self.range * self.tsfc /self.v_inf
        #             * self.cruise_drag * -cruise_lift_grad
        #             /(self.cruise_lift**2.0)                             )

        #g[0,:]  = self.c1 * TOGW_grad
        #g[0,:] *= self.var_scale

        ########################## Constraint Gradients ########################
        A = np.zeros((self.num_con,self.ndv),dtype=TransferScheme.dtype)

        # lift trim constraint gradients
        #cruise_weight_grad = 0.5 * (TOGW_grad + dW2dx)
        #A[0,:] = self.c3 * (cruise_weight_grad - cruise_lift_grad  )
        #A[0,:] *= self.var_scale

        #maneuver_weight_grad = 0.5 * (TOGW_grad + dW2dx)
        #A[1,:] = self.c3 * (2.5 * maneuver_weight_grad - maneuver_lift_grad  )
        #A[1,:] *= self.var_scale

        # fixed area gradient
        #A[2,:]  = self.c4 * self.area_grad(shape_dv)
        #A[2,:] *= self.var_scale

        # KS failure constraint gradient
        #A[3,:] = self.c2 * self.safety_factor *  maneuver_ks_grad
        #A[3,:] *= self.var_scale

        # smoothness gradients
        #A[4:,:]  = self.c5 * self.eval_smoothness_grad(thickness_dv)
        #A[4:,:] *= self.var_scale
        print('variables:')
        for i in range(self.ndv):
            print(variables[i].name, variables[i].value)
        return g, A, fail


################################################################################
x = np.array([0.0,0.015])# AOA and Thickness, first Aero then struct
#x = np.array([0.015])
dp = wedge_adjoint()
print('created object')
#dp.eval_forward()
#for i in range(1):
obj, con, fail = dp.eval_objcon(x)
print ('objective = ',obj)
g, A, fail = dp.eval_objcon_grad(x)
print('grad = ',g)
print('FINISHED')
