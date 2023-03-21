import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx

options = {'c1':0.5,'c2':0.3,'w':0.9}

optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=6,
options=options)

optimizer.optimize(fx.sphere, iters=1000)

x_max = 1
x_min = 0
my_bounds = (x_min, x_max)

bounds=my_bounds