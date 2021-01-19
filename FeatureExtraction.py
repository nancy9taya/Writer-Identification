from ComputeSlant_feature import *
from gaborFilter import *
from LBP import *


def getFeatureVector(segmentedLine):
    feature_vector=[]
    # first Feature :: AngleHistogram
    feature_vector.extend(AnglesHistogram(segmentedLine))
    # second feature:: connected components
    kernels= gabor_filter_bank()
    feature_vector.extend(calculate_gabor_featuers(segmentedLine ,kernels))
    """
    Call Here function
    """
    # third feature :: Disk Fractal
    return feature_vector

