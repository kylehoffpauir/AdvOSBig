import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import matplotlib.pyplot as plt

# converted from: https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python
from matplotlib import patches


def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    # Height is bigger. Heigth becomes 20 pixels.
    nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
    if (nwidth == 0):  # rare case but minimum is 1 pixel
        nwidth = 1
        # resize and sharpen
    img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
    wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
    newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    # newImage.save("sample.png

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [np.float32((255 - x) * 1.0 / 255.0) for x in tv]
    print(tva)
    return tva


def makeMNIST(imgPath):
    image = Image.open(imgPath)
    bw_image = image.convert(mode='L') #L is 8-bit black-and-white image mode
    bw_image = ImageEnhance.Contrast(bw_image).enhance(1.5)
    SIZE = 30
    samples = [] #array to store cut images
    for digit, y in enumerate(range(0, bw_image.height, SIZE)):
        #print('Cutting digit:', digit)
        cuts=[]
        for x in range(0, bw_image.width, SIZE):
            cut = bw_image.crop(box=(x, y, x+SIZE, y+SIZE))
            cuts.append(cut)
    samples.append(cuts)

    def resize_and_center(sample, new_size=28):
        inv_sample = ImageOps.invert(sample)
        bbox = inv_sample.getbbox()
        crop = inv_sample.crop(bbox)
        delta_w = new_size - crop.size[0]
        delta_h = new_size - crop.size[1]
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        return ImageOps.expand(crop, padding)

    resized_samples = []
    for row in samples:
        resized_samples.append([resize_and_center(sample) for sample in row])


    binary_samples = np.array([[sample.getdata() for sample in row] for row in resized_samples])
    binary_samples = binary_samples.reshape(len(resized_samples)*len(resized_samples[0]), 28, 28)
    return binary_samples



