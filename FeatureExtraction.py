from ComputeSlant_feature import *
from gaborFilter import *
from DiskFractal_feature import *
from LBP import *
from ConnectedComponent import *


def getFeatureVector(segmentedLine):
    feature_vector=[]
    # first Feature :: AngleHistogram
    # feature_vector.extend(AnglesHistogram(segmentedLine.copy()))
    # second feature:: connected components
    kernels= gabor_filter_bank()
    feature_vector.extend(calculate_gabor_featuers(segmentedLine.copy(),kernels))
    # r=fractal_dimension(segmentedLine)
    feature_vector.append(fractal_dimension(segmentedLine.copy()))
    feature_vector.extend(ConnectedComponents(segmentedLine.copy()))
    # print("FEEEEEEEEEEEEEEEEEEETUREEEEEEEEEEEEEEEEEEEEEEEE")
    # print(feature_vector)
    # print(r)
    # exit(0)
    """
    Call Here function
    """
    # third feature :: Disk Fractal
    return feature_vector

