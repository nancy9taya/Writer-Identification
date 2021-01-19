from ComputeSlant_feature import *


def getFeatureVector(segmentedLine):
    feature_vector=[]
    # first Feature :: AngleHistogram
    feature_vector.extend(AnglesHistogram(segmentedLine))
    # second feature:: connected components
    """
    Call Here function
    """
    # third feature :: Disk Fractal
    return feature_vector

