# Author: Vincent Whannou de Dravo

import logging
import sys
import numpy as np
from PIL import Image, ImageFilter
#import os
#import math
#import re

ANGLE_PRIME = 95273  # for LUTs
RADIUS_PRIME = 29537  # for LUTs

STRESS, STRASS, MAXENV, MINENV = range(4)

CHANNEL0, CHANNEL1, CHANNEL2 = range(3)


def strassCore(img, R, filename="", Ni=5, Ns=5, type=STRASS):
    '''
    strassCore(img, R, filename="", Ni=5, Ns=5, type=STRASS)

    Description: STRASS uses in core a Monte Carlo process simulation to compute 
    the output pixel of a hazy scene image.  
    
    img: The original RGB image matrix
    R: The image radius (image width)
    filename: the original image filename with a hazy scene, used to derive the output image filename
    Ni: Number of iterations, five by default in the original STRASS code
    Ns: Number of sample Ns= 5 here since the processing is slow with Python, 100 by default in the original C++ code
    type: which concept to apply to the output (STRASS by default)
    '''

    res_v = np.copy(img)
    res_r = np.copy(img)
    res_h = np.copy(img)

    res_v = res_v.astype('float') / 255.0
    res_r = res_r.astype('float') / 255.0
    res_h = res_h.astype('float') /255.0

    logging.info(f'output filename preparation: {filename} {Ns} {Ni}')
    pos = filename.find(".png")
    pivot = filename[:pos]

    logging.info(f"filename without extension: {pivot}")

    lut_cos = np.random.rand(ANGLE_PRIME) * 2 * np.pi
    lut_sin = np.random.rand(ANGLE_PRIME) * 2 * np.pi
    radiuses = np.random.rand(RADIUS_PRIME)

    angle_no, radius_no = 0, 0
    best_min, best_max, cumul = 0.0, 0.0, 0.0


    for x in range(res_h.shape[0]):
        for y in range(res_h.shape[1]):
            for c in range(res_h.shape[2]):
                res_v[x, y, c] = 0.0
                res_r[x, y, c] = 0.0
                for it in range(Ni):
                    #print(f'current pix val: {res_h[x, y, c]}')
                    best_min, best_max = res_h[x, y, c], res_h[x, y, c]
                    cumul = 0.0
                    for s in range(Ns):
                        while True:
                            angle_no = (angle_no + 1) % ANGLE_PRIME
                            radius_no = (radius_no + 1) % RADIUS_PRIME
                            u = int(x + radiuses[radius_no] * R * np.cos(lut_cos[angle_no]))
                            v = int(y + radiuses[radius_no] * R * np.sin(lut_sin[angle_no]))
                            if 0 <= u < res_h.shape[0] and 0 <= v < res_h.shape[1]:
                                break
                        cumul += res_h[u, v, c]
                        best_min = min(best_min, res_h[u, v, c])
                        best_max = max(best_max, res_h[u, v, c])
                    cumul /= Ns
                    range_ = best_max - best_min
                    if range_ == 0:
                        s = 0.5
                    elif res_h[x, y, c] - cumul < 0:
                        s = 0
                    else:
                        s = (res_h[x, y, c] - cumul) / range_
                    res_v[x, y, c] += s
                    # Uncomment this if you use max env or min env concepts
                    #res_r[x, y, c] += range_
                res_v[x, y, c] /= Ni
                # Uncomment this if you use max env or min env concepts
                #res_r[x, y, c] /= Ni

    # STRASS Concept
    if type == STRASS:
        logging.info(f'pixel at (0,0): {res_v[0,0,0]}')
        logging.info(f'strass res array type: {res_v.dtype}')
        logging.info(f'strass res array max: {np.max(res_v)}')
        logging.info(f'strass res array min: {np.min(res_v)}')
        # The STRASS Dehazing algorithm output
        return res_v

    # MAX ENVELOP Concept 
    elif type == MAXENV:
        env = np.copy(img)
        for x in range(env.shape[0]):
            for y in range(env.shape[1]):
                for z in range(env.shape[2]):
                    for v in range(env.shape[3]):
                        env[x, y, z, v] = res_h[x, y, z, v] + (1 - res_v[x, y, z, v]) * res_r[x, y, z, v]
        return env
    
    # MIN ENVELOP Concept
    elif type == MINENV:
        env = np.copy(img)
        for x in range(env.shape[0]):
            for y in range(env.shape[1]):
                for z in range(env.shape[2]):
                    for v in range(env.shape[3]):
                        env[x, y, z, v] = res_h[x, y, z, v] - res_v[x, y, z, v] * res_r[x, y, z, v]
        return env
    

# naive paralellism
def strassCore2(img, R, filename="", Ni=5, Ns=5, type=STRASS):
    '''
    strassCore2(img, R, filename="", Ni=5, Ns=5, type=STRASS)

    Description: STRASS uses in core a Monte Carlo process simulation to compute 
    the output pixel of a hazy scene image.  
    
    :param: img: The original RGB image matrix
    :param: R: The image radius (image width)
    :param: filename: the original image filename with a hazy scene, used to derive the output image filename
    :param: Ni: Number of iterations, five by default in the original STRASS code
    :param: Ns: Number of sample Ns= 5 here since the processing is slow with Python, 100 by default in the original C++ code
    :param: type: which concept to apply to the output (STRASS by default)
    '''

    res_v = np.copy(img)
    res_r = np.copy(img)
    res_h = np.copy(img)

    res_v = res_v.astype('float') / 255.0
    res_r = res_r.astype('float') / 255.0
    res_h = res_h.astype('float') /255.0

    logging.info(f'output filename preparation: {filename} {Ns} {Ni}')
    pos = filename.find(".png")
    pivot = filename[:pos]

    logging.info(f"filename without extension: {pivot}")

    lut_cos = np.random.rand(ANGLE_PRIME) * 2 * np.pi
    lut_sin = np.random.rand(ANGLE_PRIME) * 2 * np.pi
    radiuses = np.random.rand(RADIUS_PRIME)

    angle_no, radius_no = 0, 0
    best_min0, best_max0, cumul0 = 0.0, 0.0, 0.0
    best_min1, best_max1, cumul1 = 0.0, 0.0, 0.0
    best_min2, best_max2, cumul2 = 0.0, 0.0, 0.0

    for x in range(res_h.shape[0]):
        for y in range(res_h.shape[1]):
            #for c in range(res_h.shape[2]):
                res_v[x, y, CHANNEL0] = 0.0
                res_v[x, y, CHANNEL1] = 0.0
                res_v[x, y, CHANNEL2] = 0.0
                    
                res_r[x, y, CHANNEL0] = 0.0
                res_r[x, y, CHANNEL1] = 0.0
                res_r[x, y, CHANNEL2] = 0.0
                for it in range(Ni):
                    #print(f'current pix val: {res_h[x, y, c]}')
                    best_min0, best_max0 = res_h[x, y, CHANNEL0], res_h[x, y, CHANNEL0]
                    best_min1, best_max1 = res_h[x, y, CHANNEL1], res_h[x, y, CHANNEL1]
                    best_min2, best_max2 = res_h[x, y, CHANNEL2], res_h[x, y, CHANNEL2]
                    cumul0, cumul1, cumul2 = 0.0, 0.0, 0.0
                    for s in range(Ns):
                        while True:
                            angle_no = (angle_no + 1) % ANGLE_PRIME
                            radius_no = (radius_no + 1) % RADIUS_PRIME
                            u = int(x + radiuses[radius_no] * R * np.cos(lut_cos[angle_no]))
                            v = int(y + radiuses[radius_no] * R * np.sin(lut_sin[angle_no]))
                            if 0 <= u < res_h.shape[0] and 0 <= v < res_h.shape[1]:
                                break
                        cumul0 += res_h[u, v, CHANNEL0]
                        cumul1 += res_h[u, v, CHANNEL1]
                        cumul2 += res_h[u, v, CHANNEL2]

                        best_min0 = min(best_min0, res_h[u, v, CHANNEL0])
                        best_min1 = min(best_min1, res_h[u, v, CHANNEL1])
                        best_min2 = min(best_min2, res_h[u, v, CHANNEL2])

                        best_max0 = max(best_max0, res_h[u, v, CHANNEL0])
                        best_max1 = max(best_max1, res_h[u, v, CHANNEL1])
                        best_max2 = max(best_max2, res_h[u, v, CHANNEL2])

                    cumul0 /= Ns
                    range_0 = best_max0 - best_min0

                    cumul1 /= Ns
                    range_1 = best_max1 - best_min1

                    cumul2 /= Ns
                    range_2 = best_max2 - best_min2

                    # CHANNEL 0
                    if range_0 == 0:
                        s0 = 0.5
                    elif res_h[x, y, CHANNEL0] - cumul0 < 0:
                        s0 = 0
                    else:
                        s0 = (res_h[x, y, CHANNEL0] - cumul0) / range_0
                    res_v[x, y, CHANNEL0] += s0

                    # CHANNEL1
                    if range_1 == 0:
                        s1 = 0.5
                    elif res_h[x, y, CHANNEL1] - cumul1 < 0:
                        s1 = 0
                    else:
                        s1 = (res_h[x, y, CHANNEL1] - cumul1) / range_1
                    res_v[x, y, CHANNEL1] += s1

                    # CHANNEL 2
                    if range_2 == 0:
                        s2 = 0.5
                    elif res_h[x, y, CHANNEL2] - cumul2 < 0:
                        s2 = 0
                    else:
                        s2 = (res_h[x, y, CHANNEL2] - cumul2) / range_2
                    res_v[x, y, CHANNEL2] += s2
                    # Uncomment this if you use max env or min env concepts
                    #res_r[x, y, c] += range_
                res_v[x, y, CHANNEL0] /= Ni
                res_v[x, y, CHANNEL1] /= Ni
                res_v[x, y, CHANNEL2] /= Ni
                # Uncomment this if you use max env or min env concepts
                #res_r[x, y, c] /= Ni

    # STRASS Concept
    if type == STRASS:
        logging.info(f'pixel at (0,0): {res_v[0,0,0]}')
        logging.info(f'strass res array type: {res_v.dtype}')
        logging.info(f'strass res array max: {np.max(res_v)}')
        logging.info(f'strass res array min: {np.min(res_v)}')
        # The STRASS Dehazing algorithm output
        return res_v

    # MAX ENVELOP Concept 
    elif type == MAXENV:
        env = np.copy(img)
        for x in range(env.shape[0]):
            for y in range(env.shape[1]):
                for z in range(env.shape[2]):
                    for v in range(env.shape[3]):
                        env[x, y, z, v] = res_h[x, y, z, v] + (1 - res_v[x, y, z, v]) * res_r[x, y, z, v]
        return env
    
    # MIN ENVELOP Concept
    elif type == MINENV:
        env = np.copy(img)
        for x in range(env.shape[0]):
            for y in range(env.shape[1]):
                for z in range(env.shape[2]):
                    for v in range(env.shape[3]):
                        env[x, y, z, v] = res_h[x, y, z, v] - res_v[x, y, z, v] * res_r[x, y, z, v]
        return env
    
def strass(img, R, filename="", Ni=5, Ns=5, equ=False, stretch=False, presat=0.0, bypass=False, postsat=0.0, gray=False):

    '''
    strass(img, R, filename="", Ni=5, Ns=5, equ=False, stretch=False, presat=0.0, bypass=False, postsat=0.0, gray=False):
    
    '''
    if equ:
        for c in range(img.shape[2]):
            img[:, :, c] = np.histogram(img[:, :, c], bins=256)[0]

    if stretch:
        for c in range(img.shape[2]):
            ch = img[:, :, c]
            tmp = np.min(ch)
            ch -= tmp
            tmp = np.max(ch)
            ch /= tmp#!usr/bin/python3

    if presat:
        img = np.clip(img, 0, 255)#!/usr/bin/env
        img = np.dstack((img[:, :, 0], img[:, :, 1]**presat, img[:, :, 2]))
        img = np.clip(img, 0, 255).astype(np.uint8)

    #res = np.copy(img)
    if not bypass:
        res = strassCore2(img, R, filename, Ni, Ns)
    res *= 255
    res = res.astype(np.uint8)
    

    if postsat:
        img = np.clip(img, 0, 255)
        img = np.dstack((img[:, :, 0], img[:, :, 1]**postsat, img[:, :, 2]))
        img = np.clip(img, 0, 255).astype(np.uint8)

    return res

def main():
    if len(sys.argv) < 7:
        print("Usage: python strass.py <input_image_path> <radius> <output_filename> <Ni> <Ns> <type>")
        return

    input_image_path = sys.argv[1]
    radius = int(sys.argv[2])
    output_filename = sys.argv[3]
    Ni = int(sys.argv[4])
    Ns = int(sys.argv[5])
    type = int(sys.argv[6])

    img = np.array(Image.open(input_image_path))

    result = strassCore2(img, radius, output_filename, Ni, Ns, type)

     # Smooth the result using PIL Bilateral Filter function
    smoothed_result = Image.fromarray((result * 255).astype(np.uint8))
    ##smoothed_result = smoothed_result.filter(ImageFilter.GaussianBlur(radius=2))
    
    if Ns < 5:
        smoothed_result = smoothed_result.filter(ImageFilter.SMOOTH_MORE)

    # Save the smoothed result as an image
    smoothed_result.save(output_filename)


if __name__ == "__main__":
    main()