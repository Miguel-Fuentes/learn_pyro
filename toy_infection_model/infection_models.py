import torch
import pyro

def num_infected_fixed_param(params):
    '''
    This defines a distribution over the following factors:
        Quarentine protocol -> Bernoulli var of Whether the infected person follows quarentine protocol
        Contacted people -> Poisson distribution (depends on protocol)
        Infection rate -> Beta distribution (depends on protocol)    
        Num_infected -> Poisson (based on contacted people and infection rate)
    The bistribution is defined by the following fixed paramaters:
        distancing -> The probability that an individual follows social distancing [0,1]
        normal_contact -> The exptected amound of people someone will contact following normal behavior [0, inf]
        distance_contact -> The exptected amound of people someone will contact following social distancing [0, inf]
        normal_ir_conc -> This will pull the infection rate from a beta distribution with concentrations (conc1, conc0) [0, inf]
        distance_ir_conc -> This will pull the infection rate from a beta distribution with concentrations (conc1, conc0) [0, inf]
    '''
    protocol = pyro.sample('protocol', pyro.distributions.Bernoulli(params['distancing']))
    protocol = 'normal_behavior' if protocol.item() == 1.0 else 'social_distancing'
    
    rate_contacted_people = {
        'normal_behavior': params['normal_contact'],
        'social_distancing': params['distance_contact']}[protocol]
    contacted_people = pyro.sample('contacted_people', pyro.distributions.Poisson(rate_contacted_people))
    
    #alpha, beta pairs for beta distribution
    infection_rate_params = {
        'social_distancing': params['normal_ir_conc'],
        'normal_behavior': params['distance_ir_conc']
    }[protocol]
    infection_rate = pyro.sample('infection_rate', pyro.distributions.Beta(*infection_rate_params))

    rate_num_infected = contacted_people * infection_rate
    num_infected = pyro.sample('num_infected', pyro.distributions.Poisson(rate_num_infected))
    return protocol, contacted_people.item(), infection_rate.item(), num_infected.item()