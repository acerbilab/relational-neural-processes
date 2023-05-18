
from sympy import *
from spatialpy import *

import matplotlib.pylab as plt
import numpy as np
import spatialpy
import pandas as pd

#definition of the simulator

def RD_Simulator(k1_ini=10,k2_ini=100,r1_ini=0.003,r2_ini=0.03,r3_ini=2,alpha12_ini=0,alpha21_ini=0,diff2_ini=0.001,diff3_ini=0.01,d3_ini=0,d1_ini=0.5):
    # First call the gillespy2.Model initializer.
    model = spatialpy.Model(name='Spatial Birth-Death')
    
    # Define Domain Type IDs as constants of the Model
    model.HABITAT = "Habitat"

    # Define domain points and attributes of a regional space for simulation.
    domain = spatialpy.Domain.create_2D_domain(
        xlim=(0, 1), ylim=(0, 1), numx=10, numy=10, type_id=model.HABITAT, fixed=True
    )
    model.add_domain(domain)

    # Define variables for the biochemical species 
    N1 = spatialpy.Species(name='N1', diffusion_coefficient=0)
    N2 = spatialpy.Species(name='N2', diffusion_coefficient="D2")
    L = spatialpy.Species(name='L', diffusion_coefficient="D3")
    model.add_species([N1,N2,L])

    ou1=np.random.choice(range(10))/10
    ou2=np.random.choice(range(10))/10

    # Scatter the initial condition for Rabbits randomly over all types.
    init_N1_pop = spatialpy.UniformInitialCondition(species='N1', count=10)
    init_N2_pop = spatialpy.PlaceInitialCondition('N2', 10,[ou1,ou2,0])
    init_L_pop = spatialpy.ScatterInitialCondition(species='L', count=0)
    model.add_initial_condition([init_N1_pop,init_N2_pop,init_L_pop])

    rand=np.random.uniform(0.9,1.1,5)

    # Define parameters for the rates of creation and destruction.
    k1 = spatialpy.Parameter(name='K1', expression=k1_ini)
    k2 = spatialpy.Parameter(name='K2', expression=k2_ini)
    r1 = spatialpy.Parameter(name='r1', expression=r1_ini*rand[0])
    r2 = spatialpy.Parameter(name='r2', expression=r2_ini*rand[1])
    r3 = spatialpy.Parameter(name='r3', expression=r3_ini*rand[2])
    alpha12 = spatialpy.Parameter(name='alpha12', expression=alpha12_ini)
    alpha21 = spatialpy.Parameter(name='alpha21', expression=alpha21_ini)
    diff2 = spatialpy.Parameter(name='D2', expression=diff2_ini*rand[4])
    diff3 = spatialpy.Parameter(name='D3', expression=diff3_ini*rand[3])
    d3= spatialpy.Parameter(name='d3', expression=d3_ini)
    d1= spatialpy.Parameter(name='d1', expression=d1_ini)
    death1 = spatialpy.Parameter(name='death1', expression=r1_ini/k1_ini)
    death2 = spatialpy.Parameter(name='death2', expression=r2_ini/k2_ini)
    model.add_parameter([k1,k2,r1,r2,r3,alpha12,alpha21,diff2,diff3,d3,d1,death1,death2])

    # Define reactions channels which cause the system to change over time.
    # The list of reactants and products for a Reaction object are each a
    # Python dictionary in which the dictionary keys are Species objects
    # and the values are stoichiometries of the species in the reaction.
    birth_N1 = spatialpy.Reaction(name='birth_N1', reactants={}, products={"N1":1}, rate="r1")
    death_N1 = spatialpy.Reaction(name='death_N1', reactants={"N1":1}, products={}, rate="death1")
    birth_N2 = spatialpy.Reaction(name='birth_N2', reactants={}, products={"N2":1}, rate="r2")
    #change = spatialpy.Reaction(name='death_N1', reactants={"N1":1}, products={"N2":1}, propensity_function="alpha12 * r1 * N1 / k1")
    death_N1_2 = spatialpy.Reaction(name='death_N1_2', reactants={"N1":1,"L":1}, products={"L":1}, rate="d1")
    death_N2 = spatialpy.Reaction(name='death_N2', reactants={"N2":1}, products={}, rate="death2")
    birth_L = spatialpy.Reaction(name='birth_L', reactants={"N2":1}, products={"N2":1, "L":1}, propensity_function="r3*N2")
    death_L = spatialpy.Reaction(name='death_L', reactants={"L":1}, products={}, rate="d3")
    model.add_reaction([birth_N1, death_N1,birth_N2, death_N2,birth_L, death_L,death_N1_2])
    #model.add_reaction([birth_N1,birth_N2,birth_L, death_L])

    # Set the timespan of the simulation.
    tspan = spatialpy.TimeSpan.linspace(t=5, num_points=6)
    model.timespan(tspan)
    return model


#generates the data
all_train=[]
all_test=[]
for i in range(4000):
    model=RD_Simulator()
    results=model.run()
    all_train.append([results.get_species("N1"),results.get_species("N2"),results.get_species("L")])

for i in range(1000):
    model=RD_Simulator()
    results=model.run()
    all_test.append([results.get_species("N1"),results.get_species("N2"),results.get_species("L")])
