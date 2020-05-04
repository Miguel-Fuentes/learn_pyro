# Toy Infection Model
Miguel Fuentes
Created: 5/1/2020
Last Updated: 5/4/2020

## Purpose
The point of this project is just for me learn the probabilistic programming language Pyro and as much of the underlying theory as possible. I know nothing about epidemiology so the point isn't to create an accurate model for the trnasmission of the coronavirus. As I started copying over the examples from the Pyto ducumentation and running the code I realized that I will learn better if I try to apply all of the lessons to my own proble. I thought about what problem to model for a while and I eventually decided to model different aspects of coronavirus transmission because since I am currently in lockdown because of COVID-19 it has been on my mind

## Overview
I plan to keep adding different models and functionality as I learn more about pyro and what kinds of things I can do with it. I will also try to keep an up to date overview of what I have done so far here.

### Sample Transmission Model
I made a simple model for transmission. The model tries to answer this question: if someone gets an infectious disease, how many people will they go on to infect. To do this the model defines some latent variables: protocol (social distancing or normal behavior), contacted people, and infection rate. Contacted people and infection rate depend on the protocol, then the variable of interest is num infected, this depends on the contacted people and the infection rate.  
I sample this model in the notebook model_sampling and plot some of the distributions that come out.  
In the notebook inter_prior, I condition this model on the protocol being normal behavior then try to use VI to learn the prior distribution of the model with a guide function of the same form. For a long time this wasn't working correctly and I was confused but I just understood my mistake and added a note to the notebook explaining my mistake in the conclusion.

### Observe Transmission
Here, I tried to make an infection model which will make observations of data. The model takes data points which specifiy the protocol the onfected individual is following an the number of people they infect (not realistic to have this data I know but this isn't meant to be a serious model anyway so I'll start here and maybe get more realistic later). I try to do this by defining the latent variables contacted people and infection rate then using them to define a distribution over number of infected people and then try to maximize the probability of seeing the data based on this distribution. The form is similar to the sample transmission model but the latent variables are now all to make inference easier. In the notebook infer_mle_params I try to run inference over this but I run into an error because my loss and params end up going to nan very quickle. I am going to try to look as some more examples to understand SVI better then come back and try to fix this later because as of writing this I have no idea how to fix it. 