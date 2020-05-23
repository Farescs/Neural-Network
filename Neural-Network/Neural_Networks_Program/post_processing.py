import numpy as np
import pandas as p

class post_processing:

    def post_process(self, set_to_process, mean, stnd):
        unnormalized_set = (set_to_process * stnd) + mean[-1]
        return unnormalized_set

