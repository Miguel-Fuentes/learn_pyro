import torch
import pyro

def num_infected():
    '''
    Here I want to model the number of people that will be infected by someone who has coronavirus.
    I'll have these factors:
        Quarentine protocol -> Bernoulli var of Whether the infected person follows quarentine protocol
        Contacted people -> Poisson distribution (depends on protocol)
        Infection rate -> Beta distribution (depends on protocol)    
    And the Variable of interest
        Num_infected -> Poisson (based on contacted people and infection rate)
    '''
    protocol = pyro.sample('protocol', pyro.distributions.Bernoulli(0.3))
    protocol = 'normal_behavior' if protocol.item() == 1.0 else 'social_distancing'
    
    rate_contacted_people = {'normal_behavior': 100.0, 'social_distancing': 30.0}[protocol]
    contacted_people = pyro.sample('contacted_people', pyro.distributions.Poisson(rate_contacted_people))
    
    #alpha, beta pairs for beta distribution
    infection_rate_params = {
        'social_distancing': (3, 27),
        'normal_behavior': (6, 24)
    }[protocol]
    infection_rate = pyro.sample('infection_rate', pyro.distributions.Beta(*infection_rate_params))

    rate_num_infected = contacted_people * infection_rate
    num_infected = pyro.sample('num_infected', pyro.distributions.Poisson(rate_num_infected))
    return protocol, contacted_people.item(), infection_rate.item(), num_infected.item()