from typing import List
import math

Vector = List[float]

def add(a:Vector,b:Vector) -> Vector:
    assert len(a) == len(b), "vectors must be the same length"

    return [x+y for x,y in zip(a,b)]

add([4,5],[8,8])

def substract(a:Vector,b:Vector) -> Vector:
    assert len(a) == len(b), "vectors must be the same length"

    return [x-y for x,y in zip(a,b)]

substract([4,5],[8,8])

def vector_sum(vectors:List[Vector]) ->  Vector:
    """ return element-wise sum of vectors"""

    assert vectors, "Vector must not be empty"

    len_vectors = len(vectors[0])
    assert all(len_vectors == len(v) for v in vectors), "vectors must be the same size"

    return [sum(row_i) for row_i in zip(*vectors)]

vector_sum([[1,4], [2,6], [3,8]])

def scalar_multiply(a:Vector, c:float) -> Vector:
    return [c*x for x in a]

def average(vectors:List[Vector]) -> Vector:
    assert vectors, "Vector must not be empty"

    len_vectors = len(vectors[0])
    assert all(len_vectors == len(v) for v in vectors), "vectors must be the same size"

    num_vectors = len(vectors)
    return [v_i/num_vectors for v_i in vector_sum(vectors)]

average([[1,4], [2,6], [3,8]])

def dot(a:Vector,b:Vector) -> float:       
    """ return dot multiplication of two vectors"""

    assert len(a) == len(b), "vectors must be the same size"

    return sum(x*y for x,y in zip(a,b))

dot([4,5],[8,8])

def sum_of_squares(v:Vector) -> float:
    return dot(v,v)

sum_of_squares([4,5])

def magnitude(v:Vector) -> float:
    return math.sqrt(sum_of_squares(v))

magnitude([6,8])

def distance(a:Vector, b:Vector) -> float:
    return magnitude(substract(a,b))

distance([1,1], [2,2])

#matrices
def make_matrix(n_rows,n_cols,entry_fn):
    return [[entry_fn(i,j) 
             for i in range(n_rows)] 
             for j in range(n_cols)]

def identity_matrix(n):
    return make_matrix(n,n, lambda i,j: 1 if i==j else 0)

identity_matrix(6)


def vector_mean(vectors: List[Vector]) -> Vector:
    n = len(vectors)
    return scalar_multiply(vector_sum(vectors), 1/n)