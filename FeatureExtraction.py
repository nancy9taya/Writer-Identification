from ComputeSlant_feature import *
from gaborFilter import *
from DiskFractal_feature import *
from LBP import *
from ConnectedComponent import *


def getFeatureVector(segmentedLine):
    feature_vector=[]
    kernels= gabor_filter_bank()
    feature_vector.extend(calculate_gabor_featuers(segmentedLine.copy(),kernels))
    feature_vector.append(fractal_dimension(segmentedLine.copy()))
    feature_vector.extend(ConnectedComponents(segmentedLine.copy()))
    """
    Call Here function
    """

    # third feature :: Disk Fractal
    return feature_vector

