"""
 Author:: Nemali Aditya <aditya.nemali@dzne.de>
==================
Exceptions

==================
This module contains exceptions classes for raising errors
"""

class SquaredKernelError(ValueError):
	'''Kernel matrix is not squared'''
	def __init__(self,shape):
		message = 'K is not squared: %s' % str(shape)
		super().__init__(message)


class RegexError(ValueError):
	'''No files found with the specific filter option'''
	def __init__(self):
		message = 'The regex should be in form of "wc1*"'
		super().__init__(message)

class LingError(ValueError):
	'''No files found with the specific filter option'''
	def __init__(self):
		message = 'The kernel is not returning a positive definite matrix. Try increasing hyper-parameters values'
		super().__init__(message)