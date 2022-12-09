import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
from random import uniform
import random
from math import sqrt
from sklearn.cluster import KMeans
import vectores as vct
import cv2

random.seed(0)

#Función que ayuda a obtener la media de un array
def avg(points):
    length = len(points[0])

    center = []
    for dimension in range(length):
        dim_sum = 0
        for p in points:
            dim_sum += p[dimension]
        #Obtenemos nuevos centros del cluster
        center.append(dim_sum / float(len(points)))

    return center


def get_centers(data_set, assignments):
    new_means = defaultdict(list)
    centers = []
    for assignment, point in zip(assignments, data_set):
        new_means[assignment].append(point)

    for count, points in new_means.items():
        centers.append(avg(points))

    return centers

#Asignamos los puntos al cluster más cercano
def assign_points(data_points, centers):
    assignments = []
    for point in data_points:
        shortest = float('inf')
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments

#Obtiene distancias entre puntos
def distance(a, b):
    length = len(a)

    _sum = 0
    for dimension in range(length):
        difference_sq = (a[dimension] - b[dimension]) ** 2
        _sum += difference_sq
    return sqrt(_sum)

#Genera los primeros centros de manera aleatoria obteniendo maximos y minimos de cada canal
def generate_k(data_set, k):
    centers = []
    length = len(data_set[0])
    min_max = defaultdict(int)

    for point in data_set:
        for i in range(length):
            val = point[i]
            min_key = 'min_%d' % i
            max_key = 'max_%d' % i
            if min_key not in min_max or val < min_max[min_key]:
                min_max[min_key] = val
            if max_key not in min_max or val > min_max[max_key]:
                min_max[max_key] = val

    for _k in range(k):
        rand_point = []
        for i in range(length):
            min_val = min_max['min_%d' % i]
            max_val = min_max['max_%d' % i]

            rand_point.append(uniform(min_val, max_val))

        centers.append(rand_point)

    return centers

#
def k_means(dataset, k):
    k_points = generate_k(dataset, k)
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    while assignments != old_assignments:
        new_centers = get_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)

    return new_centers, assignments

def print_img(img, name):
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.title(name)
    plt.show()

def img_segment(mat_img, kClusters):
    img_reds = mat_img[:,:,0]
    img_greens = mat_img[:,:,1]
    img_blues = mat_img[:,:,2]

    red = img_reds.reshape((-1, 1))
    green = img_greens.reshape((-1, 1))
    blue = img_blues.reshape((-1, 1))

    pixel_RGB = np.concatenate((red,green,blue), axis=1)

    centers, asignaciones = k_means(pixel_RGB, kClusters)

    m = red.shape
    for i in range(m[0]):
        red[i] = centers[asignaciones[i]][0]
        green[i] = centers[asignaciones[i]][1]
        blue[i] = centers[asignaciones[i]][2]
    red.shape = img_reds.shape
    green.shape = img_greens.shape
    blue.shape = img_blues.shape
    red = red[:, :, np.newaxis]
    green = green[:, :, np.newaxis]
    blue = blue[:, :, np.newaxis]

    k_image = np.concatenate((red,green,blue),axis=2)

    jit_color = list(map(int,centers[4]))
    for i in range(len(k_image)):
        for j in range(len(k_image[0])):
            comparison = k_image[i][j] == jit_color
            equalarrays = comparison.all()
            if equalarrays == False:
                k_image[i][j] = [0,0,0]
    return k_image

def obtain_coords(k_image):
    coords = []
    for i in range(len(k_image)):
        for j in range(len(k_image[0])):
            comparison = k_image[i][j] == [255,255,255]
            equalarrays = comparison.all()
            if equalarrays == True:
                coords.append([i,j])
    return coords

def k_means_scikit(coords):
    kmeans = KMeans(n_clusters=4).fit(coords)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    jit1 = []
    jit2 = []
    jit3 = []
    jit4 = []
    jits = [jit1,jit2,jit3,jit4]
    for i in range(len(coords)):
        jits[labels[i]].append(coords[i])
    return jits

def get_obj(img, objs):
    jits = []
    for i in range(len(objs)):
        jits.append(len(objs[i]))
    jits.sort()
    print(jits)
    jits_sel = [0,0]
    for j in range(len(objs)):
        if len(objs[j]) == jits[0]:
            jits_sel[1]=(objs[j])
        if len(objs[j]) == jits[3]:
            jits_sel[0]=(objs[j])

    x1,y1 = [],[]
    for i in range(len(jits_sel[0])):
        x = jits_sel[0][i][0]
        y = jits_sel[0][i][1]
        x1.append(x)
        y1.append(y)
        img[x][y] = 100
    maxX = max(x1)
    minX = min(x1)
    maxY = max(y1)
    minY = min(y1)
    esquinas = [[maxX,minY],[minX,maxY]]
    for m in esquinas:
        x = m[0]
        y = m[1]
    for i in range(len(jits_sel[1])):
        x = jits_sel[1][i][0]
        y = jits_sel[1][i][1]
    return img, esquinas, jits_sel[0],jits_sel[1]

def create_line(esquinas,img, jit_diag):
    v1 = esquinas[0]
    v2 = esquinas[1]
    v1 = vct.create_vector(v1[0],v1[1],1)
    v2 = vct.create_vector(v2[0],v2[1],1)
    vres = vct.cross_operation(v1,v2)
    valy,valx = vct.evaluate_line(vres,v2[0],v1[0])
    valy  = list(map(int,valy))
    cruces = []
    for m in range(len(valy)):
        comparison = img[valx[m]][valy[m]] == 100
        equalarrays = comparison.all()
        if equalarrays== True:
                img[valx[m]][valy[m]] =255
                cruces.append([valx[m],valy[m]])

    return [cruces[0],cruces[len(cruces)-1]]
    
def get_line_hor(img,jit):
    xs = []
    for i in range(len(jit)):
        xs.append(jit[i][1])
    p1 = xs.index(min(xs))
    p2 = xs.index(max(xs))

    return jit[p1], jit[p2]
    
    

if __name__ == '__main__':
    #Leemos nuestra imagen
    img = cv2.imread("Jit1.jpg")
    #Reescalamos la imagen para poder ejecutar en tiempo razonable
    width = int(img.shape[1] * 15 / 100)
    height = int(img.shape[0] * 15 / 100)
    dsize = (width,height)
    imgrs = cv2.resize(img,dsize)

    #Convertimos a tipo entero nuestra matriz
    mat_img = imgrs.astype(np.uint8)
    cv2.imshow("Original",imgrs)
    cv2.waitKey(0)

    #Segmentamos imagen en colores con kClusters = 5
    img_k = img_segment(mat_img,5)
    cv2.imshow("Segmentada en colores", img_k)
    cv2.waitKey(0)
    #Convertimos a escala de grises, aplicamos gauss y canny a nuestra imagen
    gris = cv2.cvtColor(img_k, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gris, (5,5), 0)
    canny = cv2.Canny(gauss, 50, 150)
    cv2.imshow("Jitomates solos", canny)
    cv2.waitKey(0)

    #Aplicamos KMeans a los contornos para obtener los 4 jitomates
    jit_coords = obtain_coords(canny)
    objs = k_means_scikit(jit_coords)

    #Obtenemos un cuadrado alrededor del jitomate diagonal
    img_sep,esquinas,jit_diag,jit_hor = get_obj(canny,objs)
    #Obtenemos una linea que atraviesa el jitomate y obtenemos los puntos donde toca el contorno del jitomate
    points_cru = create_line(esquinas, img_sep,jit_diag)
    #Obtenemos los extremos horizontales del jitomate horizontal
    point_hor1, point_hor2 = get_line_hor(img_sep,jit_hor)

    #Calculamos distancias entre extremos
    distance_1 = vct.distance(points_cru[0],points_cru[1])
    distance_2 = vct.distance(point_hor1,point_hor2)

    #Desplegamos datos
    print(f"Coordenadas extremos jitomate 1: ({points_cru[0][1]},{points_cru[0][0]}) y ({points_cru[1][1]},{points_cru[1][0]}) Distancia = {distance_1}")
    print(f"Coordenadas extremos jitomate 2: ({point_hor1[1]},{point_hor1[0]}) y ({point_hor2[1]},{point_hor2[0]}) Distancia = {distance_2}")
    #Desplegamos imagen y trazamos lineas
    cv2.line(imgrs, (points_cru[0][1],points_cru[0][0]),(points_cru[1][1],points_cru[1][0]),(143,65,73),2)
    cv2.line(imgrs, (point_hor1[1],point_hor1[0]),(point_hor2[1],point_hor2[0]),(143,65,73),2)
    cv2.imshow("Imagen final", imgrs)
    cv2.waitKey(0)