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


def observe_transmission(data):
    '''
    This model observes defines a prior distribution over the amount of people an infected individual will transmit to,
    then it observes data of actual transmissions. This uses all continuous distributions and latent variables to make VI easier.
    It is assumed that the data will be a list of tuples, where the first value states which protocol the person follows and
    the second value is the number of people the infected individual transmitted to.
    The latent variables for this model are the following:
        cotacted_people_i -> How many people individual i contacted
        infection_rate_i -> The infection rate of individual i
    The observations thie model makes are the following:
        num_infected_i -> how many people individual i infected
    '''
    
    normal_contact_loc = torch.tensor(50.0)
    normal_contact_scale = torch.tensor(10.0)
    normal_conc0 = torch.tensor(10.0)
    normal_conc1 = torch.tensor(10.0)

    distance_contact_loc = torch.tensor(25.0)
    distance_contact_scale = torch.tensor(5.0)
    distance_conc0 = torch.tensor(10.0)
    distance_conc1 = torch.tensor(10.0)

    num_infected_scale = pyro.param('num_infected_scale', torch.tensor(5.0), constraint=constraints.positive)
    
    for i, datum in enumerate(data):
        protocol, num_infected = datum
        if protocol == 'normal_behavior':
            contacted_people = pyro.sample(f'contacted_people_{i}', pyro.distributions.Normal(normal_contact_loc, normal_contact_scale))
            infection_rate = pyro.sample(f'infection_rate_{i}', pyro.distributions.Beta(normal_conc1, normal_conc0))
        elif protocol == 'social_distancing':
            contacted_people = pyro.sample(f'contacted_people_{i}', pyro.distributions.Normal(distance_contact_loc, distance_contact_scale))
            infection_rate = pyro.sample(f'infection_rate_{i}', pyro.distributions.Beta(distance_conc1, distance_conc0))
        else:
            raise ValueError(f'Data point {i} has protocol value {protocol}, expects values \'social_distancing\' or \'normal_behavior\'')
        num_infected_loc = contacted_people * infection_rate
        num_infected = pyro.sample(f'num_infected_{i}', pyro.distributions.Normal(num_infected_loc, num_infected_scale), obs=num_infected)

