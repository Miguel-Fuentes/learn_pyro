import torch
from torch.distributions import constraints

import pyro

def sample_transmission():
    '''
    This defines a distribution over the following factors:
        Quarentine protocol -> Bernoulli var of Whether the infected person follows quarentine protocol
        Contacted people -> Poisson distribution (depends on protocol)
        Infection rate -> Beta distribution (depends on protocol)    
        Num_infected -> Poisson (based on contacted people and infection rate)
    '''
    # Determine whether sampled person will follow social distancing
    distancing = torch.tensor(0.3)
    protocol = pyro.sample('protocol', pyro.distributions.Bernoulli(distancing))
    protocol = 'normal_behavior' if protocol.item() == 1.0 else 'social_distancing'
    
    # Determine how many people they are expected to run into over the course of their transmittable period
    normal_contact = torch.tensor(100.0)
    distance_contact = torch.tensor(30.0)
    rate_contacted_people = {
        'normal_behavior': normal_contact,
        'social_distancing': distance_contact
    }[protocol]
    contacted_people = pyro.sample('contacted_people', pyro.distributions.Poisson(rate_contacted_people))

    # concentrations for the 2 beta distributions
    normal_conc0 = torch.tensor(24.0)
    normal_conc1 = torch.tensor(6.0)
    distance_conc0 = torch.tensor(27.0)
    distance_conc1 = torch.tensor(3.0)
    
    # determine the infection rate of the carrier
    infection_rate_params = {
        'normal_behavior': (normal_conc1, normal_conc0),
        'social_distancing': (distance_conc1, distance_conc0)
    }[protocol]
    infection_rate = pyro.sample('infection_rate', pyro.distributions.Beta(*infection_rate_params))

    # determine how many people they will infect over the course of the carrier's transmission period
    rate_num_infected = contacted_people * infection_rate
    num_infected = pyro.sample('num_infected', pyro.distributions.Poisson(rate_num_infected))
    
    # return the sample
    return protocol, contacted_people, infection_rate, num_infected

