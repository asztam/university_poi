# ver 0.9
import csv
import numpy as np
import pyransac3d as py
import random
from matplotlib.pyplot import figure, show, title
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


# loading the coordinates of the points from .xyz file
def points_read():
    with open('LidarData_concat.xyz', 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for x, y, z in reader:
            yield x, y, z


# initialisation of the set of the points
points_set = []
for line in points_read():
    points_set.append(line)


# drawing cloud of the points
def draw_3d(points):
    fig = figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    x, y, z = zip(*points)

    # conversion of tuple type to the float type in order to make a 3-dimensional plot
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    z = np.array(z, dtype=float)

    title('Visualisation of points cloud')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter(x, y, z)

    show()


draw_3d(points_set)


# drawing clusters of the points produced by the K-Means algorithm
def drawKMeans_3d(points):
    clusterer = KMeans(n_clusters=3)  # clusterer object, list of tuples

    X_clus = np.array(points, dtype=float)  # conversion of list of tuples into array
    y_pred = clusterer.fit_predict(X_clus)  # grouping the points and labeling/assigning the points to the clusters

    blue = y_pred == 0  # cluster label
    red = y_pred == 1  # cluster label
    black = y_pred == 2  # cluster label

    fig = figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    title('KMeans Algorithm')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter(X_clus[blue, 0], X_clus[blue, 1], X_clus[blue, 2], c="b")
    ax.scatter(X_clus[red, 0], X_clus[red, 1], X_clus[red, 2], c="r")
    ax.scatter(X_clus[black, 0], X_clus[black, 1], X_clus[black, 2], c="k")

    show()

    threshold = 0.01
    thrs = 0.05
    # plane equation derivation - cluster blue
    Xb = np.array(X_clus[blue], dtype=float)
    ab = random.choice(Xb)  # three potential inliers
    bb = random.choice(Xb)
    cb = random.choice(Xb)

    vector_A = np.subtract(ab, cb)
    vector_B = np.subtract(bb, cb)

    vector_Ua = vector_A / np.linalg.norm(vector_A)  # norm of the vector
    vector_Ub = vector_B / np.linalg.norm(vector_B)
    vector_Uc = np.cross(vector_Ua, vector_Ub)  # cross product

    db = -np.sum(np.multiply(cb, vector_Uc))  # distance from 0xyz

    # how points fits to the derived plane?
    distance_all_points = (vector_Uc[0] * Xb[:, 0] + vector_Uc[1] * Xb[:, 1] + vector_Uc[2] * Xb[:, 2]) + db

    # points that 'fits' to the model, 'fit' threshold condition/definition -> group of inliers and their amount
    inliers = np.where(np.abs(distance_all_points) <= threshold)[0]  # returns indexes of these inliers
    model_size = len(inliers)  # lenght of inliers list

    print('\nCluster blue')
    print('plane normal vector values: ')
    print('a: ', vector_Uc[0], '\nb: ', vector_Uc[1], '\nc: ', vector_Uc[2], '\nd: ',
          db, )
    print('Number of inliers: ', model_size)

    # in this form it will be almost always the last case - accuracy should be defined (threshold)
    if vector_Uc[0] <= thrs and vector_Uc[1] <= thrs and vector_Uc[2] != 0:
        print('Plane is horizontal.', '\n')
    elif (vector_Uc[0] != 0 or vector_Uc[1] != 0) and vector_Uc[2] <= thrs:
        print('Plane is vertical.', '\n')
    else:
        print('Plane is not one dimensional.', '\n')

        # plane equation derivation - cluster red
    Xr = np.array(X_clus[red], dtype=float)
    ar = random.choice(Xr)  # three potential inliers
    br = random.choice(Xr)
    cr = random.choice(Xr)
    vector_A = np.subtract(ar, cr)
    vector_B = np.subtract(br, cr)
    vector_Ua = vector_A / np.linalg.norm(vector_A)  # norm of the vector
    vector_Ub = vector_B / np.linalg.norm(vector_B)
    vector_Uc = np.cross(vector_Ua, vector_Ub)  # cross product
    dr = -np.sum(np.multiply(cr, vector_Uc))  # distance from 0xyz
    # how points fits to the derived plane?
    distance_all_points = (vector_Uc[0] * Xr[:, 0] + vector_Uc[1] * Xr[:, 1] + vector_Uc[2] * Xr[:, 2]) + dr
    # points that 'fits' to the model, 'fit' threshold condition/definition -> group of inliers and their amount
    inliers = np.where(np.abs(distance_all_points) <= threshold)[0]  # returns indexes of these inliers
    model_size = len(inliers)  # lenght of inliers list
    print('Cluster red')
    print('plane normal vector values: ')
    print('a: ', vector_Uc[0], '\nb: ', vector_Uc[1], '\nc: ',
          vector_Uc[2], '\nd: ',
          dr, )
    print('Number of inliers: ', model_size)
    # in this form it will be almost always the last case - accuracy should be defined (threshold)
    if vector_Uc[0] <= thrs and vector_Uc[1] <= thrs and vector_Uc[2] != 0:
        print('Plane is horizontal.', '\n')
    elif (vector_Uc[0] != 0 or vector_Uc[1] != 0) and vector_Uc[2] <= thrs:
        print('Plane is vertical.', '\n')
    else:
        print('Plane is not one dimensional.', '\n')
    # plane equation derivation cluster black
    Xk = np.array(X_clus[black], dtype=float)
    a = random.choice(Xk)  # three potential inliers
    b = random.choice(Xk)
    c = random.choice(Xk)
    vector_A = np.subtract(a, c)
    vector_B = np.subtract(b, c)
    vector_Ua = vector_A / np.linalg.norm(vector_A)  # norm of the vector
    vector_Ub = vector_B / np.linalg.norm(vector_B)
    vector_Uc = np.cross(vector_Ua, vector_Ub)  # cross product
    d = -np.sum(np.multiply(c, vector_Uc))  # distance from 0xyz
    # how points fits to the derived plane?
    distance_all_points = (vector_Uc[0] * Xk[:, 0] + vector_Uc[1] * Xk[:, 1] + vector_Uc[2] * Xk[:, 2]) + d
    # points that 'fits' to the model, 'fit' threshold condition/definition -> group of inliers and their amount
    inliers = np.where(np.abs(distance_all_points) <= threshold)[0]  # returns indexes of these inliers
    model_size = len(inliers)  # lenght of inliers list
    print('Cluster black')
    print('plane normal vector coefficients: ')
    print('a: ', vector_Uc[0], '\nb: ', vector_Uc[1], '\nc: ', vector_Uc[2],
          '\nd: ',
          d, )
    print('Number of inliers: ', model_size)
    # in this form it will be almost always the last case - accuracy should be defined (threshold)
    if vector_Uc[0] <= thrs and vector_Uc[1] <= thrs and vector_Uc[2] != 0:
        print('Plane is horizontal.', '\n\n')
    elif (vector_Uc[0] != 0 or vector_Uc[1] != 0) and vector_Uc[2] <= thrs:
        print('Plane is vertical.', '\n\n')
    else:
        print('Plane is not one dimensional.', '\n\n')


drawKMeans_3d(points_set)


# ransac 3d algorithm
def ransac3d_plane(points):
    X = np.array(points, dtype=float)
    plane = py.Plane()
    best_eq, best_inliers = plane.fit(X, 0.01)
    print('\nransac3d plane all points')
    print('\n\nPlane normal vector coefficients:', '\na: ', best_eq[0], '\nb: ', best_eq[1], '\nc: ', best_eq[2],
          '\nd: ', best_eq[3], '\n')


ransac3d_plane(points_set)


# drawing clusters of the points produced by the DBSCAN algorithm
def drawDBSCAN_3d(points, eps, min_samples):
    X_clus = np.array(points, dtype=float)
    clusterer = DBSCAN(eps, min_samples=min_samples)
    y_pred = clusterer.fit_predict(X_clus)

    # black = y_pred == -1  # outliers - uncomment if desired on plot
    blue = y_pred == 0
    red = y_pred == 1
    black = y_pred == 2

    fig = figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    # ax.scatter(X_clus[black, 0], X_clus[black, 1], X_clus[black, 2], c="black") # outliers
    title('DBSCAN Algorithm')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter(X_clus[blue, 0], X_clus[blue, 1], X_clus[blue, 2], c="b")
    ax.scatter(X_clus[red, 0], X_clus[red, 1], X_clus[red, 2], c="r")
    ax.scatter(X_clus[black, 0], X_clus[black, 1], X_clus[black, 2], c="k")

    show()

    # I did not manage to group clusters by plane-fitting but I will do this ASAP
    # Xb = np.array(X_clus[blue], dtype=float)
    # plane = py.Plane()
    # best_eq, best_inliers = plane.fit(Xb, 0.01)
    # print('\nransac3d')
    # print('\nCluster blue')
    # print('\n\nPlane normal vector coefficients:', '\na: ', best_eq[0], '\nb: ', best_eq[1], '\nc: ', best_eq[2],
    #       '\nd: ', best_eq[3], '\n')


drawDBSCAN_3d(points_set, 50, 1000)
