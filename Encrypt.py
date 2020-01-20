import cv2                  #OpenCV
import os                   #Path setting and file-retrieval
import glob                 #File counting
import random               #Obviously neccessary
import numpy as np          #See above
import CONFIG               #Module with Debug flags and other constants
import time                 #Literally just timing

os.chdir(CONFIG.PATH)

# Function to equalize Luma channel
def HistEQ(img_in):
    # Convert to LAB Space
    img_lab = cv2.cvtColor(img_in, cv2.COLOR_BGR2LAB)

    # Equalize L(Luma) channel
    img_lab[:,:,0] = cv2.equalizeHist(img_lab[:,:,0])

    # Convert back to RGB Space
    img_out = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    print(os.getcwd())
    if CONFIG.DEBUG_HISTEQ:
        cv2.imshow('Input image', img_in)
        cv2.imshow('Histogram equalized', img_out)
    return img_out

#Generate 64-bit Integer hash based on rough horizontal gradient
def gradientHash(img, hashSize=8):
    # Resize image; add extra column to compute the horizontal gradient
    imgBW = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgScaled = cv2.resize(imgBW,(hashSize+1,hashSize))

    # Compute horizontal gradient b/w adjacent pixels
    diffMat = imgScaled[:, 1:] > imgScaled[:, :-1]

    # Convert the gradient image to a hash
    return sum([2**i for (i, v) in enumerate(diffMat.flatten()) if v])

# Arnold's Cat Map
def ArCatMap(img_in, num):
    dim = img_in.shape
    N = dim[0]
    img_out = np.zeros([N, N, dim[2]])

    for x in range(N):
        for y in range(N):
            img_out[x][y] = img_in[(2*x+y)%N][(x+y)%N]

    if CONFIG.DEBUG_CATMAP:
        cv2.imwrite("catmap\\iteration " + str(num) + ".png", img_out)
    return img_out

# Mersenne-Twister Intra-Column-Shuffle
def MTShuffle(img_in, imghash):
    mask = 2**CONFIG.MASK_BITS - 1   # Default: 8 bits
    temphash = imghash
    dim = img_in.shape
    N = dim[0]
    for j in range(N):
        random.seed(temphash & mask)
        random.shuffle(img_in[:, j])
        temphash = temphash>>CONFIG.MASK_BITS
        if temphash==0:
            temphash = imghash
    return img_in

#XOR Image with a Fractal
def FracXor(imghash):
    #Open the image
    img_in = cv2.imread("4mtshuffle.png", 1)

    #Select a file for use based on hash
    fileCount = len(glob.glob1("fractals","*.png"))
    fracID = (imghash % fileCount) + 1
    filename = "fractals\\" + str(fracID) + ".png"
    print(imghash)
    print(fileCount)
    print(fracID)
    #Read the file, resize it, then XOR
    fractal = cv2.imread(filename, 1)
    dim = img_in.shape
    fractal = cv2.resize(fractal,(dim[0],dim[1]))
    img_out = cv2.bitwise_xor(img_in,fractal)

    return img_out

# Driver function
def Encrypt():
    filename = "raytracer480.png"
    img = cv2.imread(filename, 1)
    dim = img.shape

    timer = np.zeros(5)
    overall_time = time.time()
    timer[0] = overall_time

    # Check image dimensions, perform HisEQ if valid and proceed
    if dim[0]!=dim[1]:
        N = min(dim[0], dim[1])
        img = cv2.resize(img,(N,N))
    imgEQ = HistEQ(img)
    timer[0] = time.time() - timer[0]
    cv2.imwrite("2histeq.png", imgEQ)
    imgAr = imgEQ

    timer[1] = time.time()
    # Compute hash of imgEQ and write to text file
    imghash = gradientHash(imgEQ)
    timer[1] = time.time() - timer[1]
    f = open("hash.txt","w+")
    f.write(str(imghash))
    f.close()

    #Clear catmap debug files
    if CONFIG.DEBUG_CATMAP:
        files = os.listdir("catmap")
        for f in files:
            os.remove(os.path.join("catmap", f))

    timer[2] = time.time()
    # Ar Phase: Cat-map Iterations
    for i in range (1, random.randint(CONFIG.AR_MIN_ITER,CONFIG.AR_MAX_ITER)):
        imgAr = ArCatMap(imgAr, i)
    timer[2] = time.time() - timer[2]
    cv2.imwrite("3catmap.png", imgAr)

    timer[3] = time.time()
    # MT Phase: Intra-column pixel shuffle
    imgMT = MTShuffle(imgAr, imghash)
    timer[3] = time.time() - timer[3]
    cv2.imwrite("4mtshuffle.png", imgMT)

    timer[4] = time.time()
    # Fractal XOR Phase
    imgFr = FracXor(imghash)
    timer[4] = time.time() - timer[4]
    cv2.imwrite("5imgfractal.png", imgFr)
    overall_time = time.time() - overall_time

    print("Pre-processing completed in " + str(timer[0]) +"s")
    print("Hashing completed in " + str(timer[1]) +"s")
    print("Arnold Mapping completed in " + str(timer[2]) +"s")
    print("MT Shuffle completed in " + str(timer[3]) +"s")
    print("Fractal XOR completed in " + str(timer[4]) +"s")
    print("Encryption took " + str(np.sum(timer)) + "s out of " + str(overall_time) + "s of net execution time")
    
Encrypt()
cv2.waitKey(0)
cv2.destroyAllWindows()