import cv2 as cv
import numpy as np
import multiprocessing as mp
import time

img = cv.imread("larger_fixed.png")

def SerialGrayScale(img):
    row, col = img.shape[0:2]
    for i in range(row):
        for j in range(col):
            img[i, j] = 0.2126*img[i, j][-1] + 0.7152*img[i, j][1] + 0.0722*img[i, j][0]
    return img

def ParallelGrayScale(img, n):
    splitImg = np.array_split(img, n, axis=0)

    with mp.Pool(processes=n) as pool:
        processedImg = pool.map(SerialGrayScale, splitImg)
    
    combined = np.vstack(processedImg)
    return combined

if __name__ == "__main__":
    start = time.time()
    serialGray = SerialGrayScale(img.copy())
    end = time.time()
    print(f"Serial Time elapsed: {end-start}s")

    start = time.time()
    parallelGray = ParallelGrayScale(img.copy(), 6)
    end = time.time()
    print(f"Parallel Time elapsed: {end-start}")

    imgResized = cv.resize(img, (500, 500))
    serialGrayResized = cv.resize(serialGray, (500, 500))
    parallelGrayResized = cv.resize(parallelGray, (500, 500))

    combined = cv.hconcat([imgResized, serialGrayResized, parallelGrayResized])

    cv.imshow("result", combined)

    cv.waitKey(0)
    cv.destroyAllWindows()
