from loc_lib.shared_tools import send_email
import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture

params = {'quantile': .3,
          'eps': .3,
          'damping': .9,
          'preference': -200,
          'n_neighbors': 10,
          'n_clusters': 3}

ap = cluster.AffinityPropagation(damping=params['damping'], preference=params['preference'])
