import numpy as np
import matplotlib.pyplot as plt
import math

def create_vector(a,b,c):
    a,b,c = a/c, b/c, c/c
    m1 = np.array([a,b,c])
    return m1

def cross_operation(v1,v2):
    result = np.cross(v1,v2)
    return result

def values(v,x):
    if v[1] == 0:
        eval = 0
    else:
        eval = (v[0]*x + v[2])/-v[1]
    return eval

def graph_vectors(vecs, cols, alpha = 1):

    plt.figure()
    plt.axvline(x=0, color='grey', zorder=0)
    plt.axhline(y=0, color='grey', zorder=0)
    
    plt.xlim(-5,5)
    plt.ylim(-5,5)

    for i in range(len(vecs)):
        x = np.concatenate([[0, 0], vecs[i]])
        plt.quiver([x[0]],
                [x[1]],
                [x[2]],
                [x[3]],
                angles='xy', scale_units='xy', scale=1, color=cols[i], alpha=alpha)

    plt.show()
def evaluate_line(v,start,end):
    start = int(start)
    end = int(end)
    x1 = range(start,end)
    y1 = [values(v,i) for i in x1]
    x1 = [i for i in range(start,end)]
    return y1,x1

def graph_cross_point(v,cross,cross2):
    x1 = range(-5,5)
    y1 = [values(v[0],i) for i in x1]

    x2 = range(-5,5)
    y2 = [values(v[1],i) for i in x2]

    x3 = range(-5,5)
    y3 = [values(v[2],i) for i in x3]

    x4 =range(-5,8)
    y4 =[values(v[3],i)for i in x4]

    plt.plot(x1,y1)
    plt.plot(x2,y2)
    plt.scatter(v[0][0],v[0][1], c = 'g', marker = 'o')
    plt.scatter(v[1][0],v[1][1], c = 'g', marker = 'o')
    # plt.plot(x3,y3)
    plt.plot(x4,y4)
    # plt.scatter(cross[0],cross[1], c = 'g', marker = 'o')
    # plt.scatter(cross2[0],cross2[1], c = 'g', marker = 'o')
    plt.show()

def ingresarM():
    variables = ['x', 'y']
    l = []
    for var in variables:
        l.append(int(input(f"{var}: ")))
    l.append(1)

def distance(m1,m2):
    x = math.pow((m1[0]-m2[0]),2)
    y = math.pow((m1[1]-m2[1]),2)

    sqrt = math.sqrt(x+y)
    return sqrt

def pend(v1):
    m = v1[0]/v1[1]

def cross_point(v1,v2):
    cross = cross_operation(v1,v2)
    cross =cross/cross[2]
    return cross
    # graph_vectors([v[0], v[1], v[2], v[3], v[4], cross], ['red', 'blue', 'orange', 'green', 'pink', 'black'])


if __name__  == '__main__':
    v1 = np.zeros(3)
    v2 = np.zeros(3)
    v3 = np.zeros(3)
    m1 = np.zeros(3)
    m2 = np.zeros(3)
    v1 = create_vector(1,0.1,3)
    v2 = create_vector(5,6,1)
    v3 = create_vector(3,1,1)
    m1 = cross_point(v1,v2)
    m2 = cross_point(v2,v3)
    print(m1)
    v = np.array([v1,v2,v3,m1,m2])
    d = distance(m1,m2)
    pend1 = pend(v1)
    print(f"Distancia entre intersecciones: {d} ")
    graph_cross_point(v,m1,m2)
    # graph_vectors([v[0], v[1], v[2], v[3], v[4],m1],['red', 'blue', 'orange', 'green', 'pink', 'black'])

    