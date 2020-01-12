# Neural_ODE
ODE learning using continuous in time back-propagation

## Serial_Training
Uses `autograd` to implement the Neural-ODE algorithm _Chen, Tian Qi, et al. "Neural ordinary differential equations." Advances in neural information processing systems. 2018._ in serial

## Parallel
Uses `autograd` as well as `mpi4py` to to run parallel trainings of the neural ODE with gradient information exchange at each epoch (will add a conditional statement to allow for update after a preset number of epochs) - implemented for a different time series

## JIT_GPU
Deployment of the NODE using JAX and its JIT module for deployment on CPU, GPU or TPU. Very convenient and good speed up.

## Fitting a dynamical system
<center>
	<img src="https://github.com/Romit-Maulik/Neural_ODE/blob/master/Figure_1.png" width="600" height="600"/>
</center>

## Progress to convergence
<center>
	<img src="https://github.com/Romit-Maulik/Neural_ODE/blob/master/Figure_2.png" width="600" height="400"/>
</center>