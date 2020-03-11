import os                   # Path setting and file-retrieval
import cv2                  # OpenCV
import numpy as np          # See above
import CONFIG as cfg        # Debug flags and constants
import CoreFunctions as cf  # Common functions
from shutil import rmtree   # Directory removal
from time import perf_counter
from random import randint

#PyCUDA Import
import pycuda.driver as cuda
import pycuda.autoinit

#os.chdir(cfg.PATH)

def PreProcess():
    # Initiliaze misc_timer
    if cfg.DEBUG_IMAGES:
        misc_timer = np.zeros(7)
    else:
        misc_timer = np.zeros(6)

    misc_timer[0] = perf_counter()
    # Check if ./images directory exists
    """if not os.path.exists(cfg.SRC):
        print("Input directory does not exist!")
        raise SystemExit(0)
    else:
        if os.path.isfile(cfg.ENC_OUT):
            os.remove(cfg.ENC_OUT)
        if os.path.isfile(cfg.DEC_OUT):
            os.remove(cfg.DEC_OUT)"""
        
    # Check if ./temp directory exists
    """if os.path.exists(cfg.TEMP):
        rmtree(cfg.TEMP)
    os.makedirs(cfg.TEMP)
    misc_timer[0] = perf_counter() - misc_timer[0]

    misc_timer[1] = perf_counter()"""
    # Open Image
    img = cv2.imread(cfg.ENC_IN, 1)
    img=cv2.resize(img,(50,50),interpolation=cv2.INTER_LANCZOS4)
    if img is None:
        print("File does not exist!")
        raise SystemExit(0)
    # Pad Image so no. of rows and columns is even
    for i in range(2):
        dim_list = list(img.shape)
        if dim_list[i]&1==1:
            dim_list[i] = 1
            line = np.empty(dim_list,dtype=np.uint8)
            img = np.concatenate((img, line), axis=i)
    misc_timer[1] = perf_counter() - misc_timer[1]

    # Write original dimensions to file
    misc_timer[2] = perf_counter()
    dim = img.shape
    with open(cfg.LOG, 'w+') as f:
        f.write(str(dim[0]) + "\n")
        f.write(str(dim[1]) + "\n")
    misc_timer[2] = perf_counter() - misc_timer[2]
    return img, img.shape, misc_timer

# Driver function
def Encrypt():
    #Initialize perf_timer
    perf_timer = np.zeros(4)
    overall_time = perf_counter()
    
    # Read image & clear temp directories
    img, dim, misc_timer = PreProcess()

    # Resize image for Arnold Mapping
    misc_timer[3] = perf_counter()
    if dim[0]!=dim[1]:
        N = max(dim[0], dim[1])
        img = cv2.resize(img,(N,N), interpolation=cv2.INTER_CUBIC)
        dim = img.shape

    # Calculate no. of rounds
    rounds = randint(8,16)
    misc_timer[3] = perf_counter() - misc_timer[3]
    
    # Flatten image to vector,transfer to GPU
    temp_timer = perf_counter()
    imgArr = np.asarray(img).reshape(-1)
    gpuimgIn = cuda.mem_alloc(imgArr.nbytes)
    gpuimgOut = cuda.mem_alloc(imgArr.nbytes)
    cuda.memcpy_htod(gpuimgIn, imgArr)
    misc_timer[1] += perf_counter() - temp_timer

    # Warm-Up GPU for accurate benchmarking
    if cfg.DEBUG_TIMER:
        funcTemp = cf.mod.get_function("WarmUp")
        funcTemp(grid=(1,1,1), block=(1,1,1))

    # Log no. of rounds of ArMapping
    temp_timer = perf_counter()
    """with open(cfg.LOG, 'a+') as f:
        f.write(str(rounds)+"\n")"""
    misc_timer[2] += perf_counter() - temp_timer

    # Perform Arnold Mapping
    perf_timer[0] = perf_counter()
    """for i in range (max(rounds,5)):
        func(gpuimgIn, gpuimgOut, grid=(dim[0],dim[1],1), block=(3,1,1))
        gpuimgIn, gpuimgOut = gpuimgOut, gpuimgIn"""
    perf_timer[0] = perf_counter() - perf_timer[0]

    if cfg.DEBUG_IMAGES:
        misc_timer[6] += cf.interImageWrite(gpuimgIn, "IN_1", len(imgArr), dim)

    # Fractal XOR Phase
    temp_timer = perf_counter()
    fractal, misc_timer[4] = cf.getFractal(dim[0])
    fracArr  = np.asarray(fractal).reshape(-1)
    gpuFrac = cuda.mem_alloc(fracArr.nbytes)
    cuda.memcpy_htod(gpuFrac, fracArr)
    func = cf.mod.get_function("FracXOR")
    misc_timer[4] = perf_counter() - temp_timer

    perf_timer[1] = perf_counter()
    func(gpuimgIn, gpuimgOut, gpuFrac, grid=(dim[0]*dim[1],1,1), block=(3,1,1))
    perf_timer[1] = perf_counter() - perf_timer[1]

    gpuimgIn, gpuimgOut = gpuimgOut, gpuimgIn

    if cfg.DEBUG_IMAGES:
        misc_timer[6] += cf.interImageWrite(gpuimgIn, "IN_2", len(imgArr), dim)

    # Permutation: ArMap-based intra-row/column rotation
    perf_timer[2] = perf_counter()
    U = cf.genRelocVec(dim[0],dim[1],cfg.P1LOG, ENC=True) # Col-rotation | len(U)=n, values from 0->m
    V = cf.genRelocVec(dim[1],dim[0],cfg.P2LOG, ENC=True) # Row-rotation | len(V)=m, values from 0->n
    perf_timer[2] = perf_counter() - perf_timer[2]
    
    # Transfer rotation-vectors to GPU
    misc_timer[5] = perf_counter()
    gpuU = cuda.mem_alloc(U.nbytes)
    gpuV = cuda.mem_alloc(V.nbytes)
    cuda.memcpy_htod(gpuU, U)
    cuda.memcpy_htod(gpuV, V)
    func = cf.mod.get_function("Enc_GenCatMap")
    misc_timer[5] = perf_counter() - misc_timer[5]

    # Perform permutation
    perf_timer[3] = perf_counter()
    for i in range(cfg.PERM_ROUNDS):
        func(gpuimgIn, gpuimgOut, gpuU, gpuV, grid=(dim[0],dim[1],1), block=(3,1,1))
        gpuimgIn, gpuimgOut = gpuimgOut, gpuimgIn
    perf_timer[3] = perf_counter() - perf_timer[3]

    if cfg.DEBUG_IMAGES:
        misc_timer[6] += cf.interImageWrite(gpuimgIn, "IN_3", len(imgArr), dim)

    # Transfer vector back to host and reshape into encrypted output
    temp_timer = perf_counter()
    cuda.memcpy_dtoh(imgArr, gpuimgIn)
    img = (np.reshape(imgArr,dim)).astype(np.uint8)
    cv2.imwrite(cfg.ENC_OUT, img)
    misc_timer[1] += perf_counter() - temp_timer
    
    # Print timing statistics
    if cfg.DEBUG_TIMER:
        overall_time = perf_counter() - overall_time
        perf = np.sum(perf_timer)
        misc = np.sum(misc_timer)

        print("\nTarget: {} ({}x{})".format(cfg.ENC_IN, dim[1], dim[0]))    

        print("\nPERF. OPS: \t{0:9.7f}s ({1:5.2f}%)".format(perf, perf/overall_time*100))
        #print("ArMap Kernel:\t{0:9.7f}s ({1:5.2f}%)".format(perf_timer[0], perf_timer[0]/overall_time*100))   
        print("XOR Kernel: \t{0:9.7f}s ({1:5.2f}%)".format(perf_timer[1], perf_timer[1]/overall_time*100))
        #print("Shuffle Gen: \t{0:9.7f}s ({1:5.2f}%)".format(perf_timer[2], perf_timer[2]/overall_time*100))
        print("Perm. Kernel:\t{0:9.7f}s ({1:5.2f}%)".format(perf_timer[3], perf_timer[3]/overall_time*100))

        print("\nMISC. OPS: \t{0:9.7f}s ({1:5.2f}%)".format(misc, misc/overall_time*100))
        print("Dir. Cleanup:\t{0:9.7f}s ({1:5.2f}%)".format(misc_timer[0], misc_timer[0]/overall_time*100)) 
        print("I/O:\t\t{0:9.7f}s ({1:5.2f}%)".format(misc_timer[1], misc_timer[1]/overall_time*100))
        print("Logging:\t{0:9.7f}s ({1:5.2f}%)".format(misc_timer[2], misc_timer[2]/overall_time*100))
        print("ArMap Misc:\t{0:9.7f}s ({1:5.2f}%)".format(misc_timer[3], misc_timer[3]/overall_time*100)) 
        print("FracXOR Misc:\t{0:9.7f}s ({1:5.2f}%)".format(misc_timer[4], misc_timer[4]/overall_time*100)) 
        print("Permute Misc:\t{0:9.7f}s ({1:5.2f}%)".format(misc_timer[5], misc_timer[5]/overall_time*100))

        if cfg.DEBUG_IMAGES:
            print("Debug Images:\t{0:9.7f}s ({1:5.2f}%)".format(misc_timer[6], misc_timer[6]/overall_time*100))

        print("\nNET TIME:\t{0:7.5f}s\n".format(overall_time))
    
Encrypt()
cv2.waitKey(0)
cv2.destroyAllWindows()
