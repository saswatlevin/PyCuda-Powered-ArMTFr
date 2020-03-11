import cv2              # OpenCV
import hashlib          # For SHA256
import secrets          # For genRelocVec()
import numpy as np      # Naturally needed
import CONFIG as cfg    # Module with Debug flags and other constants
import os               # Path setting & File Counting
from random import randint
from time import perf_counter

#PyCUDA Import
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

#os.chdir(cfg.PATH)

# Generate and return rotation vector of length n containing values < m
def genRelocVec(m, n, logfile, ENC=True):
    # Initialize constants
    if ENC:
        secGen = secrets.SystemRandom()
        a = secGen.randint(2,cfg.PERMINTLIM)
        b = secGen.randint(2,cfg.PERMINTLIM)
        c = 1 + a*b
        x = secGen.uniform(0.0001,1.0)
        y = secGen.uniform(0.0001,1.0)
        offset = secGen.randint(1,cfg.PERMINTLIM)
        # Log parameters for decryption
        with open(logfile, 'a+') as f:
            f.write(str(a) +"\n")
            f.write(str(b) +"\n")
            f.write(str(x) +"\n")
            f.write(str(y) +"\n")
            f.write(str(offset) + "\n")
    else:
        with open(logfile, "r") as f:
            fl = f.readlines()
            a = int(fl[0])
            b = int(fl[1])
            c = 1 + a*b
            x = float(fl[2])
            y = float(fl[3])
            offset = int(fl[4])
    unzero = 0.0000001

    # Skip first <offset> values
    for i in range(offset):
        x = (x + a*y)%1 + unzero
        y = (b*x + c*y)%1 + unzero
    
    # Start writing intermediate values
    ranF = np.zeros((n),dtype=np.float)
    for i in range(n//2):
        x = (x + a*y)%1 + unzero
        y = (b*x + c*y)%1 + unzero
        ranF[2*i] = x
        ranF[2*i+1] = y
    
    # Generate relocation vector
    exp = 10**14
    vec = np.zeros((n),dtype=np.uint16)
    for i in range(n):
        vec[i] = np.uint16((ranF[i]*exp)%m)
    return vec

def getFractal(N, fracID=-1):
    timer = perf_counter()
    '''
    # Read/Write fractal filename based on mode
    if fracID==-1:
        fileCount = len(os.listdir(cfg.FRAC))
        fracID = (randint(0,N) % fileCount) + 1
        with open(cfg.LOG, 'a+') as f:
            f.write(str(fracID)+"\n")

    #Read the file, resize it, then XOR
    filename = cfg.FRAC + str(fracID) + ".png"
    fractal = cv2.imread(filename, 1)
    timer = perf_counter() - timer
    return cv2.resize(fractal,(N,N)), timer
    '''
    fractal =  cv2.imread("Gradient.png")
    return cv2.resize(fractal,(N,N)), perf_counter() - timer

def interImageWrite(gpuImg, name, size, dim):
    timer = perf_counter()
    imgArr = np.zeros(size,dtype=np.uint8)
    cuda.memcpy_dtoh(imgArr, gpuImg)
    imgTemp = (np.reshape(imgArr,dim)).astype(np.uint8)
    filename = name + ".png"
    cv2.imwrite(filename, imgTemp)
    return perf_counter() - timer

mod = SourceModule("""
    #include <stdint.h>
    __global__ void Enc_GenCatMap(uint8_t *in, uint8_t *out, uint16_t *colRotate, uint16_t *rowRotate)
    {
        int colShift = colRotate[blockIdx.y];
        int rowShift = rowRotate[(blockIdx.x + colShift)%gridDim.x];
        int InDex    = ((gridDim.y)*blockIdx.x + blockIdx.y) * 3  + threadIdx.x;
        int OutDex   = ((gridDim.y)*((blockIdx.x + colShift)%gridDim.x) + (blockIdx.y + rowShift)%gridDim.y) * 3  + threadIdx.x;
        out[OutDex]  = in[InDex];
    }

    __global__ void Dec_GenCatMap(uint8_t *in, uint8_t *out, uint16_t *colRotate, uint16_t *rowRotate)
    {
        int colShift = colRotate[blockIdx.y];
        int rowShift = rowRotate[(blockIdx.x + colShift)%gridDim.x];
        int OutDex   = ((gridDim.y)*blockIdx.x + blockIdx.y) * 3  + threadIdx.x;
        int InDex    = ((gridDim.y)*((blockIdx.x + colShift)%gridDim.x) + (blockIdx.y + rowShift)%gridDim.y) * 3  + threadIdx.x;
        out[OutDex]  = in[InDex];
    }

    __global__ void FracXOR(uint8_t *in, uint8_t *out, uint8_t *fractal)
    {
        int idx = blockIdx.x * 3 + threadIdx.x;
        out[idx] = in[idx]^fractal[idx];
    } 

    __global__ void WarmUp()
    {
        return;
    } 
  """)
