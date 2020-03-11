import cv2                  # OpenCV
import numpy as np          # See above
import CONFIG as cfg        # Debug flags and constants
import CoreFunctions as cf  # Common functions
from os import chdir        # Path-setting
from time import perf_counter

#PyCUDA Import
import pycuda.driver as cuda
import pycuda.autoinit

#os.chdir(cfg.PATH)

# Driver function
def Decrypt():
    #Initialize Timers
    if cfg.DEBUG_IMAGES:
        misc_timer = np.zeros(6)
    else:
        misc_timer = np.zeros(5)

    perf_timer = np.zeros(5)
    overall_time = perf_counter()

    # Read input image
    misc_timer[0] = overall_time
    img = cv2.imread(cfg.ENC_OUT, 1)
    if img is None:
        print("File does not exist!")
        raise SystemExit(0)
    dim = img.shape

    misc_timer[1] = perf_counter()
    # Read log file
    with open(cfg.LOG, "r") as f:
        width = int(f.readline())
        height = int(f.readline())
        #rounds = int(f.readline())
        #fracID = int(f.readline())
    misc_timer[1] = perf_counter() - misc_timer[1]
    
    # Flatten image to vector and send to GPU
    imgArr  = np.asarray(img).reshape(-1)
    gpuimgIn  = cuda.mem_alloc(imgArr.nbytes)
    gpuimgOut = cuda.mem_alloc(imgArr.nbytes)
    cuda.memcpy_htod(gpuimgIn, imgArr)
    misc_timer[0] = perf_counter() - misc_timer[0] - misc_timer[1]

    # Warm-Up GPU for accurate benchmarking
    if cfg.DEBUG_TIMER:
        funcTemp = cf.mod.get_function("WarmUp")
        funcTemp(grid=(1,1,1), block=(1,1,1))
    
    # Inverse Permutation: Intra-row/column rotation
    perf_timer[0] = perf_counter()
    U = cf.genRelocVec(dim[0],dim[1],cfg.P1LOG, ENC=False) # Col-rotation | len(U)=n, values from 0->m
    V = cf.genRelocVec(dim[1],dim[0],cfg.P2LOG, ENC=False) # Row-rotation | len(V)=m, values from 0->n
    perf_timer[0] = perf_counter() - perf_timer[0]
    
    misc_timer[2] = perf_counter()
    gpuU = cuda.mem_alloc(U.nbytes)
    gpuV = cuda.mem_alloc(V.nbytes)
    cuda.memcpy_htod(gpuU, U)
    cuda.memcpy_htod(gpuV, V)
    func = cf.mod.get_function("Dec_GenCatMap")
    misc_timer[2] = perf_counter() - misc_timer[2]

    perf_timer[1] = perf_counter()
    for i in range(cfg.PERM_ROUNDS):
        func(gpuimgIn, gpuimgOut, gpuU, gpuV, grid=(dim[0],dim[1],1), block=(3,1,1))
        gpuimgIn, gpuimgOut = gpuimgOut, gpuimgIn
    perf_timer[1] = perf_counter() - perf_timer[1]

    if cfg.DEBUG_IMAGES:
        misc_timer[5] += cf.interImageWrite(gpuimgIn, "OUT_1", len(imgArr), dim)

    # Inverse Fractal XOR Phase
    temp_timer = perf_counter()
    fractal, misc_timer[3] = cf.getFractal(dim[0])
    fracArr  = np.asarray(fractal).reshape(-1)
    gpuFrac = cuda.mem_alloc(fracArr.nbytes)
    cuda.memcpy_htod(gpuFrac, fracArr)
    func = cf.mod.get_function("FracXOR")
    misc_timer[3] = perf_counter() - temp_timer

    perf_timer[2] = perf_counter()
    func(gpuimgIn, gpuimgOut, gpuFrac, grid=(dim[0]*dim[1],1,1), block=(3,1,1))
    perf_timer[2] = perf_counter() - perf_timer[2]

    gpuimgIn, gpuimgOut = gpuimgOut, gpuimgIn

    if cfg.DEBUG_IMAGES:
        misc_timer[5] += cf.interImageWrite(gpuimgIn, "OUT_2", len(imgArr), dim)

    # Ar Phase: Cat-map Iterations
    misc_timer[4] = perf_counter()
    """imgShuffle = np.arange(start=0, stop=len(imgArr)/3, dtype=np.uint32)
    gpuShuffIn = cuda.mem_alloc(imgShuffle.nbytes)
    gpuShuffOut = cuda.mem_alloc(imgShuffle.nbytes)
    cuda.memcpy_htod(gpuShuffIn, imgShuffle)
    func = cf.mod.get_function("ArMapTable")"""
    misc_timer[4] = perf_counter() - misc_timer[4]

    # Recalculate mapping to generate lookup table
    perf_timer[3] = perf_counter()
    """for i in range(rounds):
        func(gpuShuffIn, gpuShuffOut, grid=(dim[0],dim[1],1), block=(1,1,1))
        gpuShuffIn, gpuShuffOut = gpuShuffOut, gpuShuffIn"""
    perf_timer[3] = perf_counter() - perf_timer[3]

    # Apply mapping
    """gpuShuffle = gpuShuffIn
    func = cf.mod.get_function("ArMapTabletoImg")"""

    perf_timer[4] = perf_counter()
    #func(gpuimgIn, gpuimgOut, gpuShuffle, grid=(dim[0]*dim[1],1,1), block=(3,1,1))
    perf_timer[4] = perf_counter() - perf_timer[4]

    if cfg.DEBUG_IMAGES:
        misc_timer[5] += cf.interImageWrite(gpuimgOut, "OUT_3", len(imgArr), dim)

    # Transfer vector back to host and reshape into original dimensions if needed
    temp_timer = perf_counter()
    cuda.memcpy_dtoh(imgArr, gpuimgIn)
    img = (np.reshape(imgArr,dim)).astype(np.uint8)

    if height!=width:
        img = cv2.resize(img,(height,width),interpolation=cv2.INTER_CUBIC)
        dim = img.shape

    cv2.imwrite(cfg.DEC_OUT, img)
    misc_timer[0] += perf_counter() - temp_timer

    # Print timing statistics
    if cfg.DEBUG_TIMER:
        overall_time = perf_counter() - overall_time
        perf = np.sum(perf_timer)
        misc = np.sum(misc_timer)

        print("\nTarget: {} ({}x{})".format(cfg.ENC_IN, dim[1], dim[0]))

        print("\nPERF. OPS: \t{0:9.7f}s ({1:5.2f}%)".format(perf, perf/overall_time*100))
        #print("Shuffle Gen:   \t{0:9.7f}s ({1:5.2f}%)".format(perf_timer[0], perf_timer[0]/overall_time*100))
        print("Perm. Kernel:  \t{0:9.7f}s ({1:5.2f}%)".format(perf_timer[1], perf_timer[1]/overall_time*100))
        print("XOR Kernel:   \t{0:9.7f}s ({1:5.2f}%)".format(perf_timer[2], perf_timer[2]/overall_time*100))
        #print("LUT Kernel:\t{0:9.7f}s ({1:5.2f}%)".format(perf_timer[3], perf_timer[3]/overall_time*100))
        #print("Mapping Kernel:\t{0:9.7f}s ({1:5.2f}%)".format(perf_timer[4], perf_timer[4]/overall_time*100))
        
        print("\nMISC. OPS: \t{0:9.7f}s ({1:5.2f}%)".format(misc, misc/overall_time*100))
        print("I/O:\t\t{0:9.7f}s ({1:5.2f}%)".format(misc_timer[0], misc_timer[0]/overall_time*100)) 
        print("Log Read:\t{0:9.7f}s ({1:5.2f}%)".format(misc_timer[1], misc_timer[1]/overall_time*100))
        print("Permute Misc:\t{0:9.7f}s ({1:5.2f}%)".format(misc_timer[2], misc_timer[2]/overall_time*100)) 
        print("FracXOR Misc:\t{0:9.7f}s ({1:5.2f}%)".format(misc_timer[3], misc_timer[3]/overall_time*100)) 
        #print("LUT Misc:\t{0:9.7f}s ({1:5.2f}%)".format(misc_timer[4], misc_timer[4]/overall_time*100)) 

        if cfg.DEBUG_IMAGES:
            print("Debug Images:\t{0:9.7f}s ({1:5.2f}%)".format(misc_timer[5], misc_timer[5]/overall_time*100))

        print("\nNET TIME:\t{0:7.5f}s\n".format(overall_time))

Decrypt()
cv2.waitKey(0)
cv2.destroyAllWindows()