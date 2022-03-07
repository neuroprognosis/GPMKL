"""
 Author:: Nemali Aditya <aditya.nemali@dzne.de>
==================
Validation of matrices
==================
This module contains function that validations matrices
"""

def check_square(x):
    '''

    :param x: To check if x is a square matrix
    :return: true or false
    '''
    if x.shape[0] == x.shape[1]:
        return True
    else:
        return False

