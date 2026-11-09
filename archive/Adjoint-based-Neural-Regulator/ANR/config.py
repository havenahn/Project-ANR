dt = 0.05 # Sampling time
N =  30  # Prediction horizon
#N = 150  #Prediction horizon for SQP-RTI

# Control input Constraints
a_min = -1
a_max =  1
w_min = -4
w_max = 4
# Total time of simulation and time steps
T_sim = 10
total_steps_sim = int(T_sim/dt)


beta =0.99
#HOCBF class-K gains
alpha0 = 3            
alpha1 = 3
