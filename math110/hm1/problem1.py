import numpy

# Problem 1

# Computes u and v such that au + bv = d
def magicBox(a, b, q):
    u = int(q)
    v = int(numpy.floor(a * q / b))
    return [u, v]

# Computing d = gcd(a, b) using the Euclidian algorithm
def euclidianAlgorithm(a, b):
    if a < b:
        print("a must be bigger than b")
    quotientList = []
    r0 = a
    r1 = b
    r2 = 1
    while (True):
        r2 = r0 % r1
        q = numpy.floor(r0 / r1)
        if r2 == 0:
            print("[u, v]") 
            print(magicBox(a, b, q))
            return r1
        # shift values
        r0 = r1
        r1 = r2


print(euclidianAlgorithm(7, 17))
