from utils import *
from process import maps
import matplotlib.pyplot as plt
import math

patches = Container(maps[2])
trace = [math.log(cov[0]+cov[4]+cov[8]) for cov in patches['covariance']]




for cov in patches['covariance']:
	# print cov[0], cov[1], cov[2]
	# print cov[3], cov[4], cov[5]
	# print cov[6], cov[7], cov[8]
	 cov[0] + cov[4] + cov[8]