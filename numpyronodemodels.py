import jax
from jax import random
import jax.numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import trace, seed, condition,  block
from numpyro.infer import MCMC, NUTS, Predictive
import funsor
import numpy as onp
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
import inspect
from contextlib import ExitStack
import arviz as az
from arviz.utils import Numba
Numba.disable_numba()
Numba.numba_flag

"""
beta verision
Still lacking:

testing
compile speed up strategies
documentation in code

"""

class NodeModel():

    def __init__(self, model_args={}):
        self.model_args = model_args
        self.nodes = {}

    def __getitem__(self,name):
        return self.nodes[name]

    def add_node(self,name,prior_distribution=None,node_function=None,arg_priors={},plate=None):
        
        self.nodes[name] = {
            'prior_distribution':prior_distribution,
            'node_function':node_function,
            'plate':plate,
            'arg_priors':arg_priors}
        
        if arg_priors != {}:
            for key in arg_priors:
                self.nodes[key+" ("+name+")"] = {
                    'prior_distribution':arg_priors[key],
                    'node_function':None,
                    'arg_priors':{},
                    'plate':None}
        
    def create_model(self):
        
        distributions = {}
        graph = {}
        sorted_graph_keys = []
        nodes_created = []

        """
        Filling dictionary "graph"
        Nodes are the key and if a node has a logpi function the value is a list of its "parents"
        """
        for node in self.nodes:
            #if node does not have logpi function it has no parents
            if self.nodes[node]['node_function'] is None:
                graph[node] = []
            else:
                #temp list to set as value in graph dictionary
                cur_parents = []
                #getting the arguements to the current logpi function
                parents = inspect.getargspec(self.nodes[node]['node_function'])[0]
                #looping over the parents
                for i in range(len(parents)):
                    #if the current parent has a node append it to the the temp list as is
                    if parents[i] in self.model_args:
                        continue
                    elif parents[i] in self.nodes:
                        cur_parents.append(parents[i])
                    #if the current parent has a node but was created through add_prior_distributions append with the naming scheme betaname_node
                    elif parents[i] in self.nodes[node]['arg_priors']:
                        cur_parents.append(parents[i]+" ("+node+")")
                    else:
                        assert parents[i]+" prior distribution is not defined for "+node
                #setting the key value pair
                graph[node] = cur_parents
                
        """
        Order of creation steps:
        
        Logpi functions that use other logpi functions in their parameters must be calculated in the correct order.
        
        If a logpi function for a node uses a previous logpi function that has not yet been calculated the model will 
        not know what to do.
        
        Order of node creation is necessary and currently done in the following way:
        
        1. Compute and create the nodes whose parents are calculated. 
        - The graph dictionary has a key-value pair where the key is the node and the value is the list of parents for that node. 
        - I also have defined a list of nodes that have been created. So I take the keys whose parents are all in the created_nodes list and create those nodes.
        
        2. Once the nodes from step 1 have been created I remove those node keys from the dictionary and add the nodes to created nodes
        
        2. Repeat from step one until the length of the sorted list equals the number of keys in the dictionary "graph"
        
        **I need some error checks here for cyclic graphs
        
        """
        
        length = len(graph)
        iteration = 0
        #while all nodes are not yet created
        while length != len(nodes_created):
            iteration += 1
            #keys to nodes that have not yet been created
            to_create = []
            #loop over keys in graph
            for key in graph:
                #if all the keys are created then add key to to_create
                if all([k in nodes_created for k in graph[key]]):
                    to_create.append(key)
            #append to_create to sorted_graph
            #loop over a pre-computed order when running the model
            sorted_graph_keys.append(to_create)
            
            #delete keys in to_create from graph since they have been "created"
            #Also add to created_nodes
            for key in to_create:
                nodes_created.append(key)
                del graph[key]
        
        self.sorted_graph_keys=sorted_graph_keys

        """
        Need VMAP, foriloop from jax to speed up compile
        Need error handling
        """
        def model(rng_seed=random.PRNGKey(0), model_args=self.model_args):
            local_vals = {}
            iteration = 0
            with seed(rng_seed=rng_seed):
                for to_create in self.sorted_graph_keys:
                    iteration += 1

                    #only on iteration 1 can nodes with prior distributions be calculated since they have no parents
                    #currently the creation of plates is ugly using if statements
                    """
                    Would like to add vmap here:
                    """
                    if iteration == 1:
                        for node in to_create:
                            if (node in self.nodes) and (self.nodes[node]['prior_distribution'] is not None):
                                if self.nodes[node]['plate'] is not None:
                                    with ExitStack() as es:
                                        for plate in self.nodes[node]['plate']:
                                            es.enter_context(plate)
                                        local_vals[node] = numpyro.sample(node,self.nodes[node]['prior_distribution'])
                                else:
                                    local_vals[node] = numpyro.sample(node,self.nodes[node]['prior_distribution'])
                        #continue back up to the for loop since we should be finished with this iteration
                        continue

                    #loop over the cur_empty keys and create their nodes in the graph
                    for node in to_create:
                        #Since the prior distribution nodes have all been calculated there should be no keys in to_create that do not have a logpi function

                        #calculate the nodes using their functions and finding their parents throught the functions args
                        """
                        Also ugly way of doing plates here with if statements
                        """
                        if self.nodes[node]['plate'] is not None:
                            with ExitStack() as es:
                                for plate in self.nodes[node]['plate']:
                                    es.enter_context(plate)
                                #getting arguments of the node function.
                                parents = inspect.getargspec(self.nodes[node]['node_function'])[0]
                                #logpi params dictionary for putting in arguments programmatically
                                logpi_params = {}
                                #looping over parents
                                #some names aren't connect since betas/nodes not explicitly created through add_node have "_node" attached
                                for parent in parents:
                                    if parent in model_args:
                                        logpi_params[parent] = model_args[parent]
                                    elif parent in local_vals:
                                        logpi_params[parent] = local_vals[parent]
                                    elif parent+" ("+node+")" in local_vals:
                                        logpi_params[parent] = local_vals[parent+" ("+node+")"]
                                #compute logpi
                                #logpi must be a distribution
                                logpi = self.nodes[node]['node_function'](**logpi_params)
                                local_vals[node] = numpyro.sample(node,logpi)
                        else:
                            #same as above just no plate
                            parents = inspect.getargspec(self.nodes[node]['node_function'])[0]
                            #logpi params dictionary for putting in arguments programmatically
                            logpi_params = {}
                            #looping over parents
                            #some names aren't connect since betas/nodes not explicitly created through add_node have "_node" attached
                            for parent in parents:
                                if parent in model_args:
                                    logpi_params[parent] = model_args[parent]
                                elif parent in local_vals:
                                    logpi_params[parent] = local_vals[parent]
                                elif parent+" ("+node+")" in local_vals:
                                    logpi_params[parent] = local_vals[parent+" ("+node+")"]
                            #compute logpi
                            #logpi must be a distribution
                            logpi = self.nodes[node]['node_function'](**logpi_params)
                            local_vals[node] = numpyro.sample(node,logpi)
                        
            return_list = []
            for value in local_vals.values():
                return_list.append(value)
            for arg in model_args.values():
                return_list.append(arg)
                
            return tuple(return_list)
            
        return model   

"""
I would like to add compatability with HMC, SVI, ...
Error handling

"""
class NodeToSklearn(BaseEstimator, RegressorMixin):

	def __init__(self, prior_model, posterior_model,  target: str, num_samples: int, num_warmup: int, num_chains: int, step_size: float = 0.01, 
                 target_accept_prob: float = 0.85, numpyro_platform: str = 'cpu', host_device_count: int = 1, nodes_blocked_prior: list = [], nodes_blocked_posterior: list = []):
        
		self.prior_model = prior_model
		self.posterior_model = posterior_model
		self.target = target
		self.num_samples = num_samples
		self.num_warmup = num_warmup
		self.num_chains = num_chains
		self.step_size = step_size
		self.target_accept_prob = target_accept_prob
		self.numpyro_platform = numpyro_platform
		self.host_device_count = host_device_count
		self.nodes_blocked_prior = nodes_blocked_prior
		self.nodes_blocked_posterior = nodes_blocked_posterior
        
		numpyro.set_platform(numpyro_platform)
		numpyro.set_host_device_count(host_device_count) #number of chains

	def fit(self, X, y):
		#prior model / finding regression coefficients
		rng_key = random.PRNGKey(0)
        
		#get variables
		X_copy = X.copy()
		X_copy[self.target] = y

		data = {}
		for column in X_copy.columns:
			data[column] = np.array(X_copy[column].values)

		# block the nodes indicated in init
		hidden_model = block(self.prior_model, hide=self.nodes_blocked_prior)

		#condition model on data
		model_cond = numpyro.handlers.condition(hidden_model, data=data)

		#execute MCMC for prior model
		nuts_kernel = NUTS(model_cond, step_size = self.step_size, target_accept_prob = self.target_accept_prob)
		mcmc = MCMC(nuts_kernel, num_samples = self.num_samples, num_warmup = self.num_warmup, num_chains = self.num_chains)
		mcmc.run(rng_key, len(data[self.target]))
        
		#save mcmc object and prior samples      
		self.prior_mcmc = mcmc
		self.prior_samples = mcmc.get_samples()
        
		return self
    
	def get_prior_arviz_data(self):
		return az.from_numpyro(self.prior_mcmc)
    
	def get_prior_samples(self):
		return self.prior_samples
    
	def get_prior_summary(self):
		return self.prior_mcmc.print_summary()
  
		#posterior model
	def predict(self, X):

		rng_key = random.PRNGKey(0)
    
		#create conditioning data with pandas DataFrame columns in X
		data = {}
		for column in X.columns:
			data[column] = np.array(X[column].values)
            
		#create conditioning data with newly learned betas from prior_samples         
		for key in self.prior_samples:
			data[key] = self.prior_samples[key].mean(axis=0)
            
		# block the nodes indicated in init
		self.posterior_model = block(self.posterior_model, hide=self.nodes_blocked_posterior)

		#condition model
		model_cond_post = numpyro.handlers.condition(
			self.posterior_model, 
			data=data
			)

		#run MCMC
		nuts_kernel = NUTS(model_cond_post, step_size = self.step_size, target_accept_prob = self.target_accept_prob)
		mcmc = MCMC(nuts_kernel, num_samples = self.num_samples, num_warmup = self.num_warmup, num_chains = self.num_chains)
		mcmc.run(rng_key, len(data[list(X.columns)[0]]))
        
		#save mcmc object and posterior samples  
		self.posterior_mcmc = mcmc
		self.posterior_samples = mcmc.get_samples()        

		preds = onp.array(mcmc.get_samples()[self.target])
		return onp.mean(preds,axis=0)

	def get_posterior_arviz_data(self):
		return az.from_numpyro(self.posterior_mcmc)
    
	def get_posterior_samples(self):
		return self.posterior_samples
        
	def get_posterior_summary(self):
		return self.posterior_mcmc.print_summary()
    
	def get_params(self, deep=False):
		params = {'prior_model':self.prior_model,
                  'posterior_model':self.posterior_model,
                  'target':self.target,
                  'num_samples':self.num_samples,
                  'num_warmup':self.num_warmup,
                  'num_chains':self.num_chains,
                  'step_size':self.step_size,
                  'target_accept_prob':self.target_accept_prob,
                  'numpyro_platform':self.numpyro_platform,
                  'host_device_count':self.host_device_count
                 }
        
		return params