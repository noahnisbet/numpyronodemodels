# **Node Models**
---
**This package automatically builds large numpyro models through a class structure rather than the traditional pyro/numypro model function.**

<br>

# **Why does NodeModels exist?**

This package is for use with the probabalistic programming language NumPyro (https://github.com/pyro-ppl/numpyro). 

This package generates NumPyro models using a class object rather than a traditional model function. The goal of this class is to allow users to create large inference models in more organized way. Hopefully the class will cut down on lines of code, allow for quick changes when developing, but not remove any of the functionality of base NumPyro. 

I named the package NodeModels because the class has a strong emphasis on the directed acyclic graph representation of a causal inference model and is more intuitive if viewed from that perspective. We initially wanted to use a package like NetworkX to define the graph structure and still can, however I was only using .add_node() or .add_edge() in NetworkX so creating our own class structure seemed more straightforward.

For some background, the motivation for this class came from a recent project where I was using NumPyro. As we got deeper into the project, our model grew quickly. My team and I found that as these models grow they get complicated and it can become tedious in a research environment where we are constantly making small changes. So, we decided that it would be worth the time to create a class structure that could programmatically create NumPyro models in order to cut down on lines of code while also keeping better track of the small details in the model. I found this class to be very helpful in my personal project, so I decided to keep working on it and share it here.

# **What does the NodeModel class do?**

I will go over the basic parts of a NodeModel object below, what you can do, and briefly explain what the class is doing under the hood.

TODO:
Document each part of the class. Constructor, .add_node, node_function, and .create_model works.
Add picture for inputs to function.


# **Installation**
---
1. Make sure you have the following packages installed: 
jax, numpyro, funsor, numpy, sklearn, pandas and arviz. 
  
2. Clone the repo:

`git clone https://github.com/noahnisbet/numpyronodemodels.git`

3. From inside the numpyronodemodels directory execute:

`python setup.py install --user`

Optionally you can omit --user to install for all users.

After that simply import numpyronodemodels in your project.

`import numpyronodemodels`

<br>

# **Usage**
---
Below I will recreate the 8 schools examples used in the NumPyro documentation and NumPyro Github linked here:

GitHub:
https://github.com/pyro-ppl/numpyro

Documentation website:
https://num.pyro.ai/en/stable/index.html

**I also included the notebook with this example in the examples folder in this repository.**

A Simple Example - 8 Schools
<br>
Let us explore NumPyro using a simple example. We will use the eight schools example from Gelman et al., Bayesian Data Analysis: Sec. 5.5, 2003, which studies the effect of coaching on SAT performance in eight schools.

The data is:

```
J = 8
y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
```

# **Original Method:** (Look at NumPyro GitHub for deeper explanation)

```
import numpyro
import numpyro.distributions as dist

# Eight Schools example
def eight_schools(J, sigma, y=None):
  mu = numpyro.sample('mu', dist.Normal(0, 5))
  tau = numpyro.sample('tau', dist.HalfCauchy(5))
  with numpyro.plate('J', J):
    theta = numpyro.sample('theta', dist.Normal(mu, tau))
    numpyro.sample('obs', dist.Normal(theta, sigma), obs=y)

from jax import random
from numpyro.infer import MCMC, NUTS

nuts_kernel = NUTS(eight_schools)
mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
rng_key = random.PRNGKey(0)
mcmc.run(rng_key, J, sigma, y=y, extra_fields=('potential_energy',))
```

# Output:
```
sample: 100%|██████████| 1500/1500 [00:06<00:00, 231.37it/s, 31 steps of size 4.85e-02. acc. prob=0.99]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
        mu      4.30      3.44      4.23     -1.68      8.96    158.83      1.01
       tau      4.05      3.45      2.97      0.29      8.29     70.21      1.02
  theta[0]      6.41      5.66      5.74     -2.99     14.52    213.16      1.01
  theta[1]      5.16      4.92      4.98     -2.10     13.42    250.40      1.01
  theta[2]      3.70      5.47      3.73     -5.71     11.76    324.69      1.00
  theta[3]      4.83      5.32      4.52     -2.80     13.70    331.97      1.00
  theta[4]      3.49      4.95      3.68     -4.69     11.44    211.57      1.00
  theta[5]      3.96      5.10      4.24     -3.66     12.65    298.62      1.01
  theta[6]      6.75      5.52      6.05     -1.46     15.60    155.36      1.01
  theta[7]      5.07      5.31      4.80     -3.16     13.83    232.52      1.02

Number of divergences: 1
```

# **numpyro node models method**

<br>

### **Import and Constructor**

First import NodeModel from numpyronodemodels after installation.

After, create a NodeModel object and in the constructor pass in any constants as a key value pair in a dictionary where the key will be a parameter in any node_function you wish.

```
from numpyronodemodels import NodeModel

eight_schools = NodeModel(model_args = {'sigma':sigma})
```

<br>

### **Adding Nodes**

Although it does not matter which order we define nodes. We will define the "top level" nodes first. These are the nodes / random variables that do not depend on any other nodes. In this examples that would be mu and tau.

Simply use the .add_node function and pass in the name as a string and the prior distribution of that random variable.

```
eight_schools.add_node(name = 'mu',
                       prior_distribution = dist.Normal(0,5))

eight_schools.add_node(name = 'tau',
                       prior_distribution = dist.HalfCauchy(5))
```

<br>

Now we will define theta. Theta depends on the samples from mu and tau, so we will use a node function!

To do this we simply create a function where the parameters of this function are the names of other nodes or constants defined in the model and you return a numpyro distribution object.

At runtime the model will sample a value from the all of he parameter's distributions and pass them in as an arguments to this function.
```
def eight_school_theta_node_function(mu, tau, sigma):
    return dist.Normal(mu, tau)
```

Theta also has a plate so we will need to define the J plate like below. This object will be passed into the plate parameter as a tuple. The reason I chose to take in a tuple of plates is so the class can accept any number of plates in this tuple. This allows for nested plates.
```
J_plate = numpyro.plate('J',J)
```
Now we can define theta using add_node below.
```
eight_schools.add_node(name = 'theta',
                       node_function=eight_school_theta_node_function,
                       plate = (J_plate,))
```

Now we repeat a similar process for "obs", however I will not observe this node yet. Instead, I will use the numpyro handler "condition" after the model is created.
```
def eight_school_obs_node_function(theta, sigma):
    return dist.Normal(theta, sigma)

eight_schools.add_node(name = 'obs',
                       node_function=eight_school_obs_node_function,
                       plate = (J_plate,))
```

Now all of the nodes are defined and I can use eight_schools.create_model(). This will convert the NodeModel to a traditional numpyro model for use however you want.
```
numpyro_traditional_model = eight_schools.create_model()
```
One thing I would like to note is the render_model function does not work with numpyro models created with .create_model. The plate does not show in the rendering, however don't worry because it is there.
```
numpyro.render_model(numpyro_traditional_model)
```
<br>

### **Running MCMC**

Now we can run MCMC below and we will get the same output as a numpyro model defined normally.
```
from jax import random
from numpyro.infer import MCMC, NUTS

conditioning_data = {'obs':y}
conditioned_numpyro_traditional_model = numpyro.handlers.condition(numpyro_traditional_model, data=conditioning_data)

nuts_kernel = NUTS(conditioned_numpyro_traditional_model)
mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
rng_key = random.PRNGKey(0)
mcmc.run(rng_key, extra_fields=('potential_energy',))
mcmc.print_summary()
```
```
sample: 100%|██████████| 1500/1500 [00:05<00:00, 283.41it/s, 31 steps of size 4.85e-02. acc. prob=0.99]

                mean       std    median      5.0%     95.0%     n_eff     r_hat
        mu      4.30      3.44      4.23     -1.68      8.96    158.83      1.01
       tau      4.05      3.45      2.97      0.29      8.29     70.21      1.02
  theta[0]      6.41      5.66      5.74     -2.99     14.52    213.16      1.01
  theta[1]      5.16      4.92      4.98     -2.10     13.42    250.40      1.01
  theta[2]      3.70      5.47      3.73     -5.71     11.76    324.69      1.00
  theta[3]      4.83      5.32      4.52     -2.80     13.70    331.97      1.00
  theta[4]      3.49      4.95      3.68     -4.69     11.44    211.57      1.00
  theta[5]      3.96      5.10      4.24     -3.66     12.65    298.62      1.01
  theta[6]      6.75      5.52      6.05     -1.46     15.60    155.36      1.01
  theta[7]      5.07      5.31      4.80     -3.16     13.83    232.52      1.02

Number of divergences: 1
```

<br>

# **Related Links**
---

Pyro forum post:
https://forum.pyro.ai/t/show-numpyro-class-structure-for-generating-large-models/4485

Numpyro github:
https://github.com/pyro-ppl/numpyro

# **Feedback**
---
Please email me at noahnisbet3@gmail.com with any questions or feedback of any kind. If there is a bug or feature that I should implement, please email me! Also feel free to use GitHub's issues tab or feature requests tab. Thank you for checking out NumPyroNodeModels!
