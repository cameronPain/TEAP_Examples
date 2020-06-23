#!/usr/bin/env python3
#
# Defines a number of functions used to convert the output from the Neural network to a surface, and finally, a series of contours defined along the axis.
#
# cdp 20200303
#
import numpy as n
import matplotlib.pyplot as pyplot
from scipy.optimize import curve_fit as curve_fit
import os
import pydicom
from matplotlib.widgets import Slider
import scipy.ndimage as ndi
import time
import sys
from scipy.ndimage.morphology import binary_dilation
import numpy as np

def check_neighbouring_pixel_values(image,i,j, dialate = True):
    image_copy  = n.array(image)
    check_value = image[i,j]
    neighbours  = []
    if image_copy[i+1,j] == check_value:
        pass
    else:
        neighbours.append(True)
    if image_copy[i+1,j+1] == check_value:
        pass
    else:
        neighbours.append(True)
    if image_copy[i,j+1] == check_value:
        pass
    else:
        neighbours.append(True)
    if image_copy[i-1,j+1] == check_value:
        pass
    else:
        neighbours.append(True)
    if image_copy[i-1,j] == check_value:
        pass
    else:
        neighbours.append(True)
    if image_copy[i-1,j-1] == check_value:
        pass
    else:
        neighbours.append(True)
    if image_copy[i,j-1] == check_value:
        pass
    else:
        neighbours.append(True)
    if image_copy[i+1,j-1] == check_value:
        pass
    else:
        neighbours.append(True)
    if sum(neighbours) >= 5:
        image[i,j] = -1* check_value + 1
    else:
        pass
def smoothROI(image, dialate=True):
    for i in range(1,len(image)-1):
        for j in range(1,len(image[0])-1):
            check_neighbouring_pixel_values(image,i,j,dialate = dialate)




#An alternate smooth function which doesnt fill in the holes.
def check_neighbouring_pixel_values_nonVoid(image,i,j, dialate = True):
    if dialate:
        check_value = 0
    else:
        check_value = 1
    neighbours = []
    if image[i+1,j] == check_value:
        pass
    else:
        neighbours.append(True)
    if image[i+1,j+1] == check_value:
        pass
    else:
        neighbours.append(True)
    if image[i,j+1] == check_value:
        pass
    else:
        neighbours.append(True)
    if image[i-1,j+1] == check_value:
        pass
    else:
        neighbours.append(True)
    if image[i-1,j] == check_value:
        pass
    else:
        neighbours.append(True)
    if image[i-1,j-1] == check_value:
        pass
    else:
        neighbours.append(True)
    if image[i,j-1] == check_value:
        pass
    else:
        neighbours.append(True)
    if image[i+1,j-1] == check_value:
        pass
    else:
        neighbours.append(True)
    return neighbours
def smoothROI_KeepHoles(image): #It will search for pixels of value 0 and if they are surrounded by more than 5 1's, it will set it to 1. Search for pixels of value 1 and if theyre surrounded by more than 5 zeros it will set them to zero. Will act to round edges and remove small blobs.
    return_image = n.array(image)
    for i in range(1,len(image)-1):
        for j in range(1,len(image[0])-1):
            if image[i,j] == 1:
                check_array = check_neighbouring_pixel_values_nonVoid(image,i,j,dialate = False)
                if n.sum(check_array) >= 6:
                    return_image[i,j] = 0
            if image[i,j] == 0:
                check_array = check_neighbouring_pixel_values_nonVoid(image,i,j,dialate = True)
                if n.sum(check_array) >= 6:
                    return_image[i,j] = 1
    return return_image








def flip_contour(contour, flip_batch = 3): #Flip in batches of 3 as these are x,y,z contours.
    flipped_contour = []
    if len(contour)//flip_batch != len(contour)/flip_batch:
        print('Trying to flip the contour in batches of ' + str(flip_batch) + ', but the length of the contour is not divisible by this size. Try a new flip_batch')
        return contour
    else:
        for i in range(len(contour)//flip_batch):
            reverse_index = len(contour)//flip_batch - i
            batch         = contour[(reverse_index - 1)*flip_batch:reverse_index*flip_batch]
            for j in batch:
                flipped_contour.append(j)
    return flipped_contour

def get_ROI_seed(Image):
    z,y,x = n.shape(Image)
    seed_z         = n.argmax(Image) // n.product(n.shape(Image[0]))
    seed_x, seed_y = n.argmax(Image[seed_z]) % y , n.argmax(Image[seed_z]) // y
    return [seed_z, seed_y, seed_x]

def get_subtraction_matrix(step = 1):
    subtraction_matrix = []
    steps              = [-step,0,step]
    for i in steps:
        for j in steps:
            for k in steps:
                subtraction_matrix.append([i,j,k])
    subtraction_matrix.remove([0,0,0])
    subtraction_matrix = n.array(subtraction_matrix)
    return subtraction_matrix

def get_2d_subtraction_matrix(step = 1):
    subtraction_matrix = n.array([[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[1,-1],[-1,-1]])
    subtraction_matrix = n.array(subtraction_matrix)
    return subtraction_matrix


def get_3d_connected_coords(point, subtraction_matrix):
    connected_coords   = []
    for i in subtraction_matrix:
        connected_coords.append(list(n.add(point, i)))
    return connected_coords

def get_2d_ROI_seed(Image):
    y,x = n.shape(Image)
    seed_x, seed_y = n.argmax(Image) % y , n.argmax(Image) // y
    return [seed_y, seed_x]

def seedROI(image, seed_locations):
    seed_locations = list(seed_locations)
    template = n.zeros(n.shape(image))
    for seed in seed_locations:
        seed = list(seed)
        template[seed[0],seed[1]] = 1
        ROI_coords    = [seed]
        tested_coords = []
        start = time.time()
        for coord in ROI_coords:
            tested_coords.append(coord)
            
            if image[coord[0] + 1, coord[1]] == 1 and [coord[0] + 1, coord[1]] not in tested_coords and template[coord[0] + 1, coord[1]] == 0:
                template[coord[0] + 1, coord[1]] = 1
                ROI_coords.append([coord[0] + 1, coord[1]])
            
            if image[coord[0] + 1, coord[1] + 1] == 1 and [coord[0] + 1, coord[1] + 1] not in tested_coords and template[coord[0] + 1, coord[1] + 1] == 0:
                template[coord[0] + 1, coord[1] + 1 ] = 1
                ROI_coords.append([coord[0] + 1, coord[1] + 1])
            
            if image[coord[0] , coord[1] + 1] == 1 and [coord[0] , coord[1]+ 1] not in tested_coords and template[coord[0] , coord[1] + 1] == 0:
                template[coord[0] , coord[1] + 1 ] = 1
                ROI_coords.append([coord[0] , coord[1] + 1])
            
            if image[coord[0] - 1, coord[1] + 1 ] == 1 and [coord[0] - 1, coord[1] + 1] not in tested_coords and template[coord[0] -1, coord[1] + 1] == 0:
                template[coord[0] -1, coord[1] + 1 ] = 1
                ROI_coords.append([coord[0] -1, coord[1] + 1])
            
            if image[coord[0] - 1, coord[1] ] == 1 and [coord[0] - 1, coord[1]] not in tested_coords and template[coord[0] -1 , coord[1] ] == 0:
                template[coord[0] -1 , coord[1] ] = 1
                ROI_coords.append([coord[0] -1 , coord[1] ])
            
            if image[coord[0] -1 , coord[1] - 1] == 1 and [coord[0] - 1, coord[1] - 1] not in tested_coords and template[coord[0] -1 , coord[1] -1] == 0:
                template[coord[0] -1 , coord[1] -1 ] = 1
                ROI_coords.append([coord[0] -1 , coord[1] -1 ])
            
            if image[coord[0] , coord[1] - 1 ] == 1 and [coord[0], coord[1] - 1] not in tested_coords and template[coord[0] , coord[1] - 1] == 0:
                template[coord[0] , coord[1] - 1 ] = 1
                ROI_coords.append([coord[0] , coord[1] - 1 ])
            
            if image[coord[0] + 1 , coord[1] - 1] == 1 and [coord[0] + 1, coord[1] -1 ] not in tested_coords and template[coord[0] + 1 , coord[1] -1 ] == 0:
                template[coord[0] + 1 , coord[1] -1 ] = 1
                ROI_coords.append([coord[0] + 1 , coord[1] -1 ])
            
            ROI_coords.remove(coord)
            now = time.time()
            loopTime = (now - start)
    if tested_coords == [seed]:
        return n.zeros(n.shape(image))
    else:
        return template

def get_seedROI_3Dforlooped(image, seed_locations):
    seed_locations     = list(seed_locations)
    subtraction_matrix = get_subtraction_matrix()
    template           = n.zeros(n.shape(image))
    figure, (ax1, ax2, ax3) = pyplot.subplots(1,3,figsize=[15,9])
    for seed in seed_locations:
        seed = list(seed)
        template[seed[0], seed[1], seed[2]] = 1
        ROI_coords    = [seed]
        tested_coords = []
        start = time.time()
        ax    = ax1.imshow(n.sum(template, axis=0), cmap = pyplot.cm.binary)
        co    = ax2.imshow(n.sum(template, axis=1), cmap = pyplot.cm.binary)
        sa    = ax3.imshow(n.sum(template, axis=2), cmap = pyplot.cm.binary)
        pyplot.show(block=False)
        for coord in ROI_coords:
            tested_coords.append(coord)
            connected_coords     = get_3d_connected_coords(coord, subtraction_matrix)
            #print(connected_coords)
            new_connected_coords = [x for x in connected_coords if x not in tested_coords]
            for new_coord in new_connected_coords:
                z_coord, y_coord, x_coord = new_coord
                if image[z_coord, y_coord, x_coord] == 1 and template[z_coord, y_coord, x_coord] == 0:
                    template[z_coord, y_coord, x_coord] = 1
                    ROI_coords.append([z_coord, y_coord, x_coord])
        
            ROI_coords.remove(coord)
            now = time.time()
            loopTime = (now-start)
            ax.set_data(n.sum(template, axis=0))
            co.set_data(n.sum(template, axis=1))
            sa.set_data(n.sum(template, axis=2))
            figure.canvas.draw()
            print('loop time: ' + str(n.round(loopTime,2)) + ' seconds', end='\r')
    return template

def get_boundary(Image):
    boundary_kernel   = n.ones([3,3])
    boundary          = []
    print('     Collecting boundaries...')
    for slice in Image:
        boundary.append(n.multiply(slice, binary_dilation(slice == 0, boundary_kernel).astype(int)))
    boundary          = n.array(boundary)
    return boundary

def check_unit_neighbours(point, image, subtraction_matrix):
    check_points     = get_3d_connected_coords(point, subtraction_matrix)
    unit_neighbours = []
    bounds = n.shape(image)
    for i in range(len(check_points)):
        for j in range(len(check_points[0])):
            if check_points[i][j] > bounds[j] or check_points[i][j] < 0:
                continue
    for neighbours in check_points:
        if image[neighbours[0],neighbours[1]] > 0:
            unit_neighbours.append(neighbours)
    return unit_neighbours


def check_intersection_point(point, image):
    #print('############# check_intersection_point #############')
    #print('Point: ', point)
    unit_neighbours_0  = check_unit_neighbours(point, image, get_2d_subtraction_matrix())
    #print('First connected point: ', unit_neighbours_0)
    unit_neighbours_1  = check_unit_neighbours(unit_neighbours_0[0], image, get_2d_subtraction_matrix())
    #print('Second connected point: ', unit_neighbours_1)
    if unit_neighbours_1 == []:
        return False, None
    if point in unit_neighbours_1:
        unit_neighbours_1.remove(point)#Remove the seed point otherwise everything is an intersection point.
        if unit_neighbours_1 == []:
            return True, unit_neighbours_0[1]
    if unit_neighbours_1[0] == unit_neighbours_0[1]:
        return False, None
    else:
        return True, unit_neighbours_0[1]

def get_seed_for_contour_collection(image):
    y,x   = n.shape(image)
    k_hor = n.array([[0,0,0],[1,1,1],[0,0,0]])
    im_h  = ndi.convolve(image, k_hor)
    if n.amax(im_h) == 3:
        seed_x, seed_y = n.argmax(im_h) % x , n.argmax(im_h) // x
        return seed_x, seed_y
    else:
        k_ver = n.array([[0,1,0], [0,1,0], [0,1,0]])
        im_v = ndi.convolve(image, k_ver)
        if n.amax(im_v) == 3:
            seed_x, seed_y = n.argmax(im_v) % x , n.argmax(im_v) // x
            return seed_x, seed_y
        else:
            seed_x, seed_y = n.argmax(image) % x , n.argmax(image) // x
            return seed_x, seed_y


def remove_useless_intersection_pts(int_pts, point):
    while point in int_pts[1]:
        idx = int_pts[1].index(point)
        int_pts[0].pop(idx)
        int_pts[1].pop(idx)


def get_check_surface_2d(side_len):
    check_surface = n.zeros([side_len,side_len])
    check_surface[:,0]  = 1
    check_surface[0,:]  = 1
    check_surface[-1,:] = 1
    check_surface[:,-1] = 1
    return check_surface

def find_closest_unit(point, image, iter=8):
    #Note, the point has already been found to have no 8 connected neightbours. Define a square surface around the point, look for unit points on the surface. Calculate the distance. If there are two equal distances, go for the once closest to the seed?
    for i in range(iter-1):
        side_len      = (4 + (i*2+1))
        check_len     = (side_len - 1) //2
        check_surface = get_check_surface_2d(side_len)
        check         = n.multiply(check_surface, image[point[0]-check_len:point[0]+check_len+1, point[1]-check_len:point[1]+check_len+1])
        if n.amax(check) != 1:
            continue
        else:
            nof_points = n.sum(check).astype(int)
            points     = []
            for j in range(nof_points):
                unit_ix, unit_iy = n.argmax(check) % side_len , n.argmax(check) // side_len
                points.append([unit_ix, unit_iy])
            points    = n.array(points)
            distances = n.sum(n.square(points), axis=1)
            min_in    = n.where(distances == n.amin(distances))[0]
            if len(min_in) == 1:
                closest_pt = [point[0] + points[min_in[0]][1] - check_len, point[1] + points[min_in[0]][0] -check_len]
                return closest_pt
            else:
                closest_pt = [point[0] + points[min_in[0]][1] - check_len, point[1] + points[min_in[0]][0] -check_len]
                return closest_pt
        return None


def step_distance(point1, point0):
    x_steps = abs(point1[0] - point0[0])
    y_steps = abs(point1[1] - point0[1])
    return x_steps + y_steps

#Brute force means of dealing with the boundaries.
def collect_single_contour(image, z_coord = 0, check_contour = False):
    subtraction_matrix = get_2d_subtraction_matrix()
    start              = time.time()
    y,x                = n.shape(image)
    seed_x, seed_y     = get_seed_for_contour_collection(image)
    collected_points   = [[seed_y, seed_x]]
    collected_contour  = [seed_x, seed_y, z_coord]
    scan_points        = [[seed_y, seed_x]]
    intersection_pts   = [[],[]]
    for point in scan_points:
        if point in intersection_pts[1]:
            remove_useless_intersection_pts(intersection_pts, point)  #If we come back around and pick up one of our intersection points, we remove it from the collection of intersections.
        if check_contour:
            fig, ax = pyplot.subplots(1,1,figsize=(20,10))
            ax.imshow(image)
            pyplot.show()
        unit_neighbours = check_unit_neighbours(point,image, subtraction_matrix)
        if len(unit_neighbours) >1:
            is_int_point, return_pt = check_intersection_point(point, image)
            if is_int_point:
                intersection_pts[0].append(point)
                intersection_pts[1].append(return_pt)
            else:
                pass
        if point[0] != 0 and point[0] != y-1 and point[1] != 0 and point[1] != x-1:
            if image[point[0] + 1, point[1]] == 1 and [point[0] + 1, point[1]] not in collected_points:
                image[point[0], point[1]] = 0
                collected_contour.append(point[1])
                collected_contour.append(point[0] + 1)
                collected_contour.append(z_coord)
                collected_points.append([point[0] + 1, point[1]])
                scan_points.append([point[0] + 1, point[1]])
            elif image[point[0] , point[1] + 1] == 1 and [point[0] , point[1]+ 1] not in collected_points:
                image[point[0] , point[1]] = 0
                collected_contour.append(point[1]+1)
                collected_contour.append(point[0])
                collected_contour.append(z_coord)
                collected_points.append([point[0] , point[1] + 1 ])
                scan_points.append([point[0] , point[1] + 1 ])
            elif image[point[0] - 1, point[1] ] == 1 and [point[0] - 1, point[1]] not in collected_points:
                image[point[0] , point[1] ] = 0
                collected_contour.append(point[1]   )
                collected_contour.append(point[0] -1)
                collected_contour.append(z_coord)
                collected_points.append([point[0] -1 , point[1] ])
                scan_points.append([point[0] -1 , point[1] ])
            elif image[point[0] , point[1] - 1 ] == 1 and [point[0], point[1] - 1] not in collected_points:
                image[point[0] , point[1]  ] = 0
                collected_contour.append(point[1] - 1 )
                collected_contour.append(point[0])
                collected_contour.append(z_coord)
                collected_points.append([point[0] , point[1] - 1 ])
                scan_points.append([point[0] , point[1] - 1 ])
            elif image[point[0] + 1 , point[1] + 1] == 1 and [point[0] + 1 , point[1]+ 1] not in collected_points:
                image[point[0] , point[1]] = 0
                collected_contour.append(point[1]+1)
                collected_contour.append(point[0]+1)
                collected_contour.append(z_coord)
                collected_points.append([point[0] + 1 , point[1] + 1 ])
                scan_points.append([point[0] + 1, point[1] + 1 ])
            elif image[point[0] - 1, point[1] + 1 ] == 1 and [point[0] - 1, point[1] + 1] not in collected_points:
                image[point[0] , point[1] ] = 0
                collected_contour.append(point[1] +1)
                collected_contour.append(point[0] -1)
                collected_contour.append(z_coord)
                collected_points.append([point[0] -1 , point[1] + 1])
                scan_points.append([point[0] -1 , point[1] +1])
            elif image[point[0] + 1 , point[1] - 1 ] == 1 and [point[0] + 1, point[1] - 1] not in collected_points:
                image[point[0] , point[1]  ] = 0
                collected_contour.append(point[1] - 1 )
                collected_contour.append(point[0] + 1)
                collected_contour.append(z_coord)
                collected_points.append([point[0] + 1 , point[1] - 1 ])
                scan_points.append([point[0] + 1 , point[1] - 1 ])
            elif image[point[0] -1 , point[1] - 1] == 1 and [point[0] -1, point[1]-1 ] not in collected_points:
                image[point[0] , point[1]] = 0
                collected_contour.append(point[1]-1)
                collected_contour.append(point[0]-1)
                collected_contour.append(z_coord)
                collected_points.append([point[0]-1 , point[1] - 1 ])
                scan_points.append([point[0]-1, point[1] - 1 ])
            else:
                image[point[0], point[1]] = 0
                isConnected = point in get_3d_connected_coords([seed_y, seed_x], subtraction_matrix)
                if point == [seed_y, seed_x]: #It is a single isolated point.
                    return collected_contour
                if isConnected: #It is a closed loop.
                    image[point[0] , point[1]] = 0
                    collected_contour.append(point[1])
                    collected_contour.append(point[0])
                    collected_contour.append(z_coord)
                    return collected_contour
                elif len(intersection_pts[0])> 0:
                    steps_back = step_distance(intersection_pts[0][-1], point)
                    if steps_back < 10:
                        collected_contour.append(intersection_pts[0][-1][1])
                        collected_contour.append(intersection_pts[0][-1][0])
                        collected_contour.append(z_coord)
                        scan_points.append(intersection_pts[0].pop(-1))
                        intersection_pts[1].pop(-1)
                    else:
                        image[seed_x,seed_y] = 1
                        closest_pt = find_closest_unit(point, image, iter = n.amin([y - point[0]-1,point[0]-1, x-point[1]-1, point[1]-1]))
                        image[seed_x, seed_y] = 0
                        if closest_pt == [seed_y, seed_x]:
                            collected_contour.append(seed_x)
                            collected_contour.append(seed_y)
                            collected_contour.append(z_coord)
                            return collected_contour
                        elif closest_pt == None:
                            scan_points.append([seed_y, seed_x])
                        else:
                            collected_contour.append(closest_pt[1])
                            collected_contour.append(closest_pt[0])
                            collected_contour.append(z_coord)
                            scan_points.append(closest_pt)
                else:
                    scan_points.append([seed_y,seed_x])
        
        elif point[0] == y-1 and point[1] == x-1:
            if image[point[0] - 1, point[1] ] == 1 and [point[0] - 1, point[1]] not in collected_points:
                image[point[0] , point[1] ] = 0
                collected_contour.append(point[1]   )
                collected_contour.append(point[0] -1)
                collected_contour.append(z_coord)
                collected_points.append([point[0] -1 , point[1] ])
                scan_points.append([point[0] -1 , point[1] ])
            elif image[point[0] , point[1] - 1 ] == 1 and [point[0], point[1] - 1] not in collected_points:
                image[point[0] , point[1]  ] = 0
                collected_contour.append(point[1] - 1 )
                collected_contour.append(point[0])
                collected_contour.append(z_coord)
                collected_points.append([point[0] , point[1] - 1 ])
                scan_points.append([point[0] , point[1] - 1 ])
            else:
                image[point[0], point[1]] = 0
                isConnected = point in get_3d_connected_coords([seed_y, seed_x], subtraction_matrix)
                if isConnected: #It is a closed loop.
                    image[point[0] , point[1]] = 0
                    collected_contour.append(point[1])
                    collected_contour.append(point[0])
                    collected_contour.append(z_coord)
                    return collected_contour
                else: #It is a single connected point (such as at the edge of the image.)
                    try:
                        FLIP_BATCH = 3
                        flipped_contour = flip_contour(collected_contour, flip_batch = FLIP_BATCH)
                        joined_contour  = flipped_contour[:-FLIP_BATCH] + original_contour
                        return joined_contour
                    except:
                        scan_points.append([seed_y, seed_x]) #Start back at the seed and scan the other way.
                        original_contour  = collected_contour
                        collected_contour = []
        #This should suffice. If one single connected point is found, there must be another one. At the end, it will skip is connected. There will only be two points as it is a single line.


        elif point[0] == y-1 and point[1] == 0:
            if image[point[0] , point[1] + 1] == 1 and [point[0] , point[1]+ 1] not in collected_points:
                image[point[0] , point[1]] = 0
                collected_contour.append(point[1]+1)
                collected_contour.append(point[0])
                collected_contour.append(z_coord)
                collected_points.append([point[0] , point[1] + 1 ])
                scan_points.append([point[0] , point[1] + 1 ])
            elif image[point[0] - 1, point[1] ] == 1 and [point[0] - 1, point[1]] not in collected_points:
                image[point[0] , point[1] ] = 0
                collected_contour.append(point[1]   )
                collected_contour.append(point[0] -1)
                collected_contour.append(z_coord)
                collected_points.append([point[0] -1 , point[1] ])
                scan_points.append([point[0] -1 , point[1] ])
            else:
                image[point[0], point[1]] = 0
                isConnected = point in get_3d_connected_coords([seed_y, seed_x], subtraction_matrix)
                if isConnected: #It is a closed loop.
                    image[point[0] , point[1]] = 0
                    collected_contour.append(point[1])
                    collected_contour.append(point[0])
                    collected_contour.append(z_coord)
                    return collected_contour
                else: #It is a single connected point (such as at the edge of the image.)
                    try:
                        FLIP_BATCH = 3
                        flipped_contour = flip_contour(collected_contour, flip_batch = FLIP_BATCH)
                        joined_contour  = flipped_contour[:-FLIP_BATCH] + original_contour
                        return joined_contour
                    except:
                        scan_points.append([seed_y, seed_x]) #Start back at the seed and scan the other way.
                        original_contour  = collected_contour
                        collected_contour = []
        #This should suffice. If one single connected point is found, there must be another one. At the end, it will skip is connected. There will only be two points as it is a single line.


        elif point[0] == 0 and point[1] == x-1:
            if image[point[0] + 1, point[1]] == 1 and [point[0] + 1, point[1]] not in collected_points:
                image[point[0], point[1]] = 0
                collected_contour.append(point[1])
                collected_contour.append(point[0] + 1)
                collected_contour.append(z_coord)
                collected_points.append([point[0] + 1, point[1]])
                scan_points.append([point[0] + 1, point[1]])
            elif image[point[0] , point[1] - 1 ] == 1 and [point[0], point[1] - 1] not in collected_points:
                image[point[0] , point[1]  ] = 0
                collected_contour.append(point[1] - 1 )
                collected_contour.append(point[0])
                collected_contour.append(z_coord)
                collected_points.append([point[0] , point[1] - 1 ])
                scan_points.append([point[0] , point[1] - 1 ])
            else:
                image[point[0], point[1]] = 0
                isConnected = point in get_3d_connected_coords([seed_y, seed_x], subtraction_matrix)
                if isConnected: #It is a closed loop.
                    image[point[0] , point[1]] = 0
                    collected_contour.append(point[1])
                    collected_contour.append(point[0])
                    collected_contour.append(z_coord)
                    return collected_contour
                else: #It is a single connected point (such as at the edge of the image.)
                    try:
                        FLIP_BATCH = 3
                        flipped_contour = flip_contour(collected_contour, flip_batch = FLIP_BATCH)
                        joined_contour  = flipped_contour[:-FLIP_BATCH] + original_contour
                        return joined_contour
                    except:
                        scan_points.append([seed_y, seed_x]) #Start back at the seed and scan the other way.
                        original_contour  = collected_contour
                        collected_contour = []
#This should suffice. If one single connected point is found, there must be another one. At the end, it will skip is connected. There will only be two points as it is a single line.


        elif point[0] == 0 and point[1] == 0:
            if image[point[0] + 1, point[1]] == 1 and [point[0] + 1, point[1]] not in collected_points:
                image[point[0], point[1]] = 0
                collected_contour.append(point[1])
                collected_contour.append(point[0] + 1)
                collected_contour.append(z_coord)
                collected_points.append([point[0] + 1, point[1]])
                scan_points.append([point[0] + 1, point[1]])
            elif image[point[0] , point[1] + 1] == 1 and [point[0] , point[1]+ 1] not in collected_points:
                image[point[0] , point[1]] = 0
                collected_contour.append(point[1]+1)
                collected_contour.append(point[0])
                collected_contour.append(z_coord)
                collected_points.append([point[0] , point[1] + 1 ])
                scan_points.append([point[0] , point[1] + 1 ])
            else:
                image[point[0], point[1]] = 0
                isConnected = point in get_3d_connected_coords([seed_y, seed_x], subtraction_matrix)
                if isConnected: #It is a closed loop.
                    image[point[0] , point[1]] = 0
                    collected_contour.append(point[1])
                    collected_contour.append(point[0])
                    collected_contour.append(z_coord)
                    return collected_contour
                else: #It is a single connected point (such as at the edge of the image.)
                    try:
                        FLIP_BATCH = 3
                        flipped_contour = flip_contour(collected_contour, flip_batch = FLIP_BATCH)
                        joined_contour  = flipped_contour[:-FLIP_BATCH] + original_contour
                        return joined_contour
                    except:
                        scan_points.append([seed_y, seed_x]) #Start back at the seed and scan the other way.
                        original_contour  = collected_contour
                        collected_contour = []
#This should suffice. If one single connected point is found, there must be another one. At the end, it will skip is connected. There will only be two points as it is a single line.


        elif point[0] == 0:
            if image[point[0] + 1, point[1]] == 1 and [point[0] + 1, point[1]] not in collected_points:
                image[point[0], point[1]] = 0
                collected_contour.append(point[1])
                collected_contour.append(point[0] + 1)
                collected_contour.append(z_coord)
                collected_points.append([point[0] + 1, point[1]])
                scan_points.append([point[0] + 1, point[1]])
            elif image[point[0] , point[1] + 1] == 1 and [point[0] , point[1]+ 1] not in collected_points:
                image[point[0] , point[1]] = 0
                collected_contour.append(point[1]+1)
                collected_contour.append(point[0])
                collected_contour.append(z_coord)
                collected_points.append([point[0] , point[1] + 1 ])
                scan_points.append([point[0] , point[1] + 1 ])
            elif image[point[0] , point[1] - 1 ] == 1 and [point[0], point[1] - 1] not in collected_points:
                image[point[0] , point[1]  ] = 0
                collected_contour.append(point[1] - 1 )
                collected_contour.append(point[0])
                collected_contour.append(z_coord)
                collected_points.append([point[0] , point[1] - 1 ])
                scan_points.append([point[0] , point[1] - 1 ])
            else:
                image[point[0], point[1]] = 0
                isConnected = point in get_3d_connected_coords([seed_y, seed_x], subtraction_matrix)
                if isConnected: #It is a closed loop.
                    image[point[0] , point[1]] = 0
                    collected_contour.append(point[1])
                    collected_contour.append(point[0])
                    collected_contour.append(z_coord)
                    return collected_contour
                else: #It is a single connected point (such as at the edge of the image.)
                    try:
                        FLIP_BATCH = 3
                        flipped_contour = flip_contour(collected_contour, flip_batch = FLIP_BATCH)
                        joined_contour  = flipped_contour[:-FLIP_BATCH] + original_contour
                        return joined_contour
                    except:
                        scan_points.append([seed_y, seed_x]) #Start back at the seed and scan the other way.
                        original_contour  = collected_contour
                        collected_contour = []
#This should suffice. If one single connected point is found, there must be another one. At the end, it will skip is connected. There will only be two points as it is a single line.


        elif point[0] == y-1:
            if image[point[0] , point[1] + 1] == 1 and [point[0] , point[1]+ 1] not in collected_points:
                image[point[0] , point[1]] = 0
                collected_contour.append(point[1]+1)
                collected_contour.append(point[0])
                collected_contour.append(z_coord)
                collected_points.append([point[0] , point[1] + 1 ])
                scan_points.append([point[0] , point[1] + 1 ])
            elif image[point[0] - 1, point[1] ] == 1 and [point[0] - 1, point[1]] not in collected_points:
                image[point[0] , point[1] ] = 0
                collected_contour.append(point[1]   )
                collected_contour.append(point[0] -1)
                collected_contour.append(z_coord)
                collected_points.append([point[0] -1 , point[1] ])
                scan_points.append([point[0] -1 , point[1] ])
            elif image[point[0] , point[1] - 1 ] == 1 and [point[0], point[1] - 1] not in collected_points:
                image[point[0] , point[1]  ] = 0
                collected_contour.append(point[1] - 1 )
                collected_contour.append(point[0])
                collected_contour.append(z_coord)
                collected_points.append([point[0] , point[1] - 1 ])
                scan_points.append([point[0] , point[1] - 1 ])
            else:
                image[point[0], point[1]] = 0
                isConnected = point in get_3d_connected_coords([seed_y, seed_x], subtraction_matrix)
                if isConnected: #It is a closed loop.
                    image[point[0] , point[1]] = 0
                    collected_contour.append(point[1])
                    collected_contour.append(point[0])
                    collected_contour.append(z_coord)
                    return collected_contour
                else: #It is a single connected point (such as at the edge of the image.)
                    try:
                        FLIP_BATCH = 3
                        flipped_contour = flip_contour(collected_contour, flip_batch = FLIP_BATCH)
                        joined_contour  = flipped_contour[:-FLIP_BATCH] + original_contour
                        return joined_contour
                    except:
                        scan_points.append([seed_y, seed_x]) #Start back at the seed and scan the other way.
                        original_contour  = collected_contour
                        collected_contour = []
#This should suffice. If one single connected point is found, there must be another one. At the end, it will skip is connected. There will only be two points as it is a single line.


        elif point[1] == 0:
            if image[point[0] + 1, point[1]] == 1 and [point[0] + 1, point[1]] not in collected_points:
                image[point[0], point[1]] = 0
                collected_contour.append(point[1])
                collected_contour.append(point[0] + 1)
                collected_contour.append(z_coord)
                collected_points.append([point[0] + 1, point[1]])
                scan_points.append([point[0] + 1, point[1]])
            elif image[point[0] , point[1] + 1] == 1 and [point[0] , point[1]+ 1] not in collected_points:
                image[point[0] , point[1]] = 0
                collected_contour.append(point[1]+1)
                collected_contour.append(point[0])
                collected_contour.append(z_coord)
                collected_points.append([point[0] , point[1] + 1 ])
                scan_points.append([point[0] , point[1] + 1 ])
            elif image[point[0] - 1, point[1] ] == 1 and [point[0] - 1, point[1]] not in collected_points:
                image[point[0] , point[1] ] = 0
                collected_contour.append(point[1]   )
                collected_contour.append(point[0] -1)
                collected_contour.append(z_coord)
                collected_points.append([point[0] -1 , point[1] ])
                scan_points.append([point[0] -1 , point[1] ])
            else:
                image[point[0], point[1]] = 0
                isConnected = point in get_3d_connected_coords([seed_y, seed_x], subtraction_matrix)
                if isConnected: #It is a closed loop.
                    image[point[0] , point[1]] = 0
                    collected_contour.append(point[1])
                    collected_contour.append(point[0])
                    collected_contour.append(z_coord)
                    return collected_contour
                else: #It is a single connected point (such as at the edge of the image.)
                    try:
                        FLIP_BATCH = 3
                        flipped_contour = flip_contour(collected_contour, flip_batch = FLIP_BATCH)
                        joined_contour  = flipped_contour[:-FLIP_BATCH] + original_contour
                        return joined_contour
                    except:
                        scan_points.append([seed_y, seed_x]) #Start back at the seed and scan the other way.
                        original_contour  = collected_contour
                        collected_contour = []
#This should suffice. If one single connected point is found, there must be another one. At the end, it will skip is connected. There will only be two points as it is a single line.


        elif point[1] == x-1:
            if image[point[0] + 1, point[1]] == 1 and [point[0] + 1, point[1]] not in collected_points:
                image[point[0], point[1]] = 0
                collected_contour.append(point[1])
                collected_contour.append(point[0] + 1)
                collected_contour.append(z_coord)
                collected_points.append([point[0] + 1, point[1]])
                scan_points.append([point[0] + 1, point[1]])
            elif image[point[0] - 1, point[1] ] == 1 and [point[0] - 1, point[1]] not in collected_points:
                image[point[0] , point[1] ] = 0
                collected_contour.append(point[1]   )
                collected_contour.append(point[0] -1)
                collected_contour.append(z_coord)
                collected_points.append([point[0] -1 , point[1] ])
                scan_points.append([point[0] -1 , point[1] ])
            elif image[point[0] , point[1] - 1 ] == 1 and [point[0], point[1] - 1] not in collected_points:
                image[point[0] , point[1]  ] = 0
                collected_contour.append(point[1] - 1 )
                collected_contour.append(point[0])
                collected_contour.append(z_coord)
                collected_points.append([point[0] , point[1] - 1 ])
                scan_points.append([point[0] , point[1] - 1 ])
            else:
                image[point[0], point[1]] = 0
                isConnected = point in get_3d_connected_coords([seed_y, seed_x], subtraction_matrix)
                if isConnected: #It is a closed loop.
                    image[point[0] , point[1]] = 0
                    collected_contour.append(point[1])
                    collected_contour.append(point[0])
                    collected_contour.append(z_coord)
                    return collected_contour
                else: #It is a single connected point (such as at the edge of the image.)
                    try:
                        FLIP_BATCH = 3
                        flipped_contour = flip_contour(collected_contour, flip_batch = FLIP_BATCH)
                        joined_contour  = flipped_contour[:-FLIP_BATCH] + original_contour
                        return joined_contour
                    except:
                        scan_points.append([seed_y, seed_x]) #Start back at the seed and scan the other way.
                        original_contour  = collected_contour
                        collected_contour = []
#This should suffice. If one single connected point is found, there must be another one. At the end, it will skip is connected. There will only be two points as it is a single line.


def get_contour_library(image):
    contour_library      = [ [] for i in range(len(image))]
    for i in range(len(image)):
        slice            = image[i]
        if n.amax(slice) == 0:
            continue
        else:
            while n.amax(slice) == 1:
                slice_seed = get_2d_ROI_seed(slice)
                contour_library[i].append(collect_single_contour(slice, z_coord = i))
    return contour_library

def remove_zero_padding(contour_library, pad_image):# removes entries in the contour_library which correspond to a padding slice in the pad_image.
    print('     Starting remove_zero_padding()')
    print('     Contour library length: ', len(contour_library))
    print('     Pad Image length: ', len(pad_image))
    new_contour_library = []
    for i in range(len(pad_image)):
        if n.amax(pad_image[i]) != 0:
            new_contour_library.append(contour_library[i])
    print('     end remove_zero_padding()')
    return new_contour_library

def remove_zero_padding_image(boundary_image, pad_image):
    new_boundary_image = []
    for i in range(len(pad_image)):
        if n.amax(pad_image[i]) != 0:
            new_boundary_image.append(boundary_image[i])
    return n.array(new_boundary_image)


def create_dicom_coordinates_contour(contour_library, Header, Flipped = False):
    print('     starting create_dicom_coordinates_contour()')
    start = time.time()
    print('     Getting pixel scale and translations...')
    z_step, y_step, x_step = float((Header[-1].ImagePositionPatient[2] - Header[0].ImagePositionPatient[2])/float(len(Header))), float(Header[0].PixelSpacing[0]), float(Header[0].PixelSpacing[1])
    #z_step                 = ((z_step)**2)**0.5
    z_vector = []
    for i in [0,1,2]:
        z_vector.append(Header[-1].ImagePositionPatient[i] - Header[0].ImagePositionPatient[i])
    z_vector = n.array(z_vector)
    z_vector = -1*n.divide(z_vector, n.linalg.norm(z_vector))#In units of mm
    rx, ry, rz, cx, cy, cz = Header[0].ImageOrientationPatient
    sx, sy, sz             = z_vector
    #CoB_Matrix             = n.array([[rx*x_step, cx*y_step, 0], [ry*x_step, cy*y_step, 0 ], [rz*x_step, cz*y_step, 1*z_step]])
    CoB_Matrix             = n.array([[rx*x_step, cx*y_step, sx*z_step], [ry*x_step, cy*y_step, sy*z_step ], [rz*x_step, cz*y_step, sz*z_step]])
    if Flipped:
        z_step = abs(z_step)
    else:
        z_step = -1 * abs(z_step)
        pass
    #The desired result is [origin[0] + x_step * C00, origin[1] + y_step * C01, origin[2] + z_step * C02, origin[0] + x_step * C10, .....] where Cxy is the yth coordinate of the xth contour point.
    # Make the following vectors, [origin[0], origin[1], origin[2], origin[0], origin[1], origin[2],....], [x_step, y_step, z_step, x_step, y_step, z_step, ....]
    x0, y0, z0 = Header[0].ImagePositionPatient
    origin     = [float(x0), float(y0), float(z0)]
    step       = [x_step, y_step, z_step]
    print('     Dicom header origin: ', origin, ' mm')
    print('     Pixel scale factors: ', step, ' mm/px')
    dicom_contour_library = [[] for i in range(len(contour_library))]
    print('     Scanning contour library...')
    print('     Dicom contour Lib len: '    , len(dicom_contour_library))
    print('     Original Contour Lib len  :', len(contour_library))
    for i in range(len(contour_library)):
        if len(contour_library[i]) == 0:
            continue
        for j in range(len(contour_library[i])):
            nof_points = len(contour_library[i][j])//3 #it is x,y,z coords so doing integer division will just ensure an int is returned.
            dicom_contour = []
            for k in range(nof_points):#Get an adequately long step_vec and origin_vec to do the calculation as said above.
                #
                image_point   = n.array([[contour_library[i][j][k*3] , contour_library[i][j][1 + k*3] , 1 + contour_library[i][j][2 + k*3]]]).transpose()
                patient_point = n.add(n.dot(CoB_Matrix,image_point).transpose(), origin)
                dicom_contour.append(patient_point[0,0])
                dicom_contour.append(patient_point[0,1])
                dicom_contour.append(patient_point[0,2])
            # end for
            #scaled_points     = n.multiply(n.array([step_vec]), n.array(contour_library[i][j]))
            #translated_points = n.add(n.array(origin_vec), scaled_points)
            #contour = list(translated_points)
            dicom_contour_library[i].append([dicom_contour])
        # end for
    #end for
    print('     dicom_contour_library after scanning: ', len(dicom_contour_library))
    end = time.time()
    print('     end create_dicom_coordinates_contour()')
    print('     time elapsed: ', n.round(end - start, 3), ' seconds')
    return dicom_contour_library


def concatenate_contours(contour_library):
    concantenated_contour_library = []
    for slice in contour_library:
        if slice == []:
            concantenated_contour_library.append([])
            continue
        else:
            concat = []
            for i in slice:
                concat = concat + i
            concantenated_contour_library.append(concat)
    return concantenated_contour_library





def thresholdPrediction(i,threshold):
    if i < threshold:
        return 0
    else:
        return 1
_thresholdPrediction = n.vectorize(thresholdPrediction)

def main():
    #Neural network prediction
    Image       = n.load('liver_prediction_example.npy')
    #Threshold mask
    thresholdImage = _thresholdPrediction(Image, 0.4)
    #Boundary of threshold mask
    boundary    = get_boundary(thresholdImage)
    print('\n starting contour function: ')
    #Converting to contours
    contour_library = get_contour_library(boundary)
    #concatenating contours.
    concat_library  = concatenate_contours(contour_library)

import argparse
if __name__ == '__main__' :
  usage = 'Written by Cameron Pain. Opens a dicom file.'
  parser = argparse.ArgumentParser(description = usage)
  args = parser.parse_args()
  main()
#end if
