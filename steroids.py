import numpy as np
import math

iuf_cache = []

def iuf(users_orig):

	global iuf_cache

	if (len(iuf_cache) == 0):	
		users = users_orig	
		m = len(users)
		for i in range(1000):
			m_j = len([0 for u in users if u[i] > 0])
			if m_j == 0:
				# Do nothing
				continue
			iuf = math.log10(m/m_j)
			for u in users:
				u[i] *= iuf
		iuf_cache = users

	return iuf_cache

def case_amplification(a):
    p = 2.5
    return a * pow(np.abs(a), p-1)
