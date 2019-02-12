import skimage
import skimage.io
import skimage.transform
import numpy as np
import tensorflow as tf
import scipy
import scipy.interpolate

def show_image(img):
    skimage.io.imshow(img)
    skimage.io.show()

# [height, width, depth]
def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    print( "Original Image Shape: ", img.shape)
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (256, 256))
    print( "Resize Image Shape: ", resized_img.shape)
    return resized_img


def load_image_big(path, crop=True):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    print( "Original Image Shape: ", img.shape)
    # we crop image from center
    if crop:
        short_edge = min(img.shape[:2])
        yy = int((img.shape[0] - short_edge) / 2)
        xx = int((img.shape[1] - short_edge) / 2)
        crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    else:
        crop_img = img

    #resized_img = skimage.transform.resize(crop_img, (256, 256))

    print( "Resize Image Shape: ", crop_img.shape)
    return crop_img


def white_noise(sizeW, sizeH):
    white = np.random.rand(sizeH, sizeW, 3)
    return white



def crappy_noise(sizeW, sizeH, mu=0.5, sigma=0.1):

    a = np.zeros((sizeW, sizeH, 3))
    for i in range(sizeW):
        for j in range(sizeH):

            a[i][j] = np.random.normal(mu, sigma, 3)

    rescale = np.minimum(np.maximum(0, a), 1)
    return rescale




def less_crappy_noise(origin, sizeW = 256, sizeH = 256):
    ch1 = origin[:,:,0]
    ch2 = origin[:,:,1]
    ch3 = origin[:,:,2]
    h1,b1 = skimage.exposure.histogram(ch1)
    h2,b2 = skimage.exposure.histogram(ch2)
    h3,b3 = skimage.exposure.histogram(ch3)
    """
    N = max(np.amax(h1), np.amax(h2), np.amax(h3))
    h1 = h1/ N
    h2 = h2 / N
    h3 = h2/ N
    b1 = b1/255
    b2 = b2/255
    b3 = b3/255
    """
    s1 = np.sum(h1)
    s2 = np.sum(h2)
    s3 = np.sum(h3)
    m1 = np.sum(np.multiply(h1,b1))/s1
    m2 = np.sum(np.multiply(h2,b2))/s2
    m3 = np.sum(np.multiply(h3,b3))/s3
    v1 = sum(np.power(h1 - m1, 2)) / len(h1)
    v2 = sum(np.power(h2 - m2, 2)) / len(h2)
    v3 = sum(np.power(h3 - m3, 2)) / len(h3)
    print(m1)
    print(v1)
    n1 = np.random.normal(m1, np.sqrt(v1), (sizeW, sizeH))
    n2 = np.random.normal(m2, np.sqrt(v2), (sizeW, sizeH))
    n3 = np.random.normal(m3, np.sqrt(v3), (sizeW, sizeH))
    noise = np.stack((n1,n2,n3), 2)

    noise = noise / 255
    noise = np.maximum(0., np.minimum(1., noise))

    return noise


def his_noise(origin,sizeW=256,sizeH=256):
    his,b = np.histogram(origin,sizeW,normed=True)

    calculate = origin.reshape(sizeW*sizeH,3)
    lis = []
    for x in calculate:
        n_bin = x[0]*sizeW*sizeH+x[1]*sizeW+x[2]
        lis.append(n_bin)
    print(np.shape(lis))

    long = np.array(lis)
    his,b = np.histogram(long,sizeW*sizeH*sizeW,normed=True)
    b = np.zeros(sizeW*sizeH*sizeW+1,dtype=np.float32)
    for i in range(0,len(b)):
        b[i]=i


    cum_values = np.zeros(b.shape)
    cum_values[1:] = np.cumsum(his*np.diff(b))
    inv_cdf = scipy.interpolate.interp1d(cum_values, b,kind='nearest',bounds_error=True)
    rand = np.random.uniform(0.,1.,sizeW*sizeH)
    answer = inv_cdf(rand)
    ko = []
    for i in answer:
        R = i%sizeW
        G = (i-R)%(sizeW*sizeH)/sizeW
        B = (i-sizeW*G-R)/(sizeW*sizeH)
        ko.append([B,G,R])
    answer = np.array(ko).reshape(sizeW,sizeH,3)/255.

    answer = np.maximum(0., np.minimum(1., answer))

    return answer
# r = np.asarray(uniform_hist(org_image[:,:,i].ravel()))




def uniform_hist(X):
    '''
    Maps data distribution onto uniform histogram

    :param X: data vector
    :return: data vector with uniform histogram
    '''
    Z = [(x, i) for i, x in enumerate(X)]
    Z.sort()
    n = len(Z)
    Rx = [0]*n
    start = 0 # starting mark
    for i in range(1, n):
        if Z[i][0] != Z[i-1][0]:
            for j in range(start, i):
                Rx[Z[j][1]] = float(start+1+i)/2.0;
            start = i
    for j in range(start, n):
        Rx[Z[j][1]] = float(start+1+n)/2.0;
    return np.asarray(Rx) / float(len(Rx))
def histogram_matching(org_image, match_image, grey=False, n_bins=100):
    '''
    Matches histogram of each color channel of org_image with histogram of match_image
    :param org_image: image whose distribution should be remapped
    :param match_image: image whose distribution should be matched
    :param grey: True if images are greyscale
    :param n_bins: number of bins used for histogram calculation
    :return: org_image with same histogram as match_image
    '''

    if grey:
        hist, bin_edges = np.histogram(match_image.ravel(), bins=n_bins, density=True)
        cum_values = np.zeros(bin_edges.shape)
        cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
        inv_cdf = scipy.interpolate.interp1d(cum_values, bin_edges,bounds_error=True)
        r = np.asarray(uniform_hist(org_image.ravel()))
        r[r>cum_values.max()] = cum_values.max()
        matched_image = inv_cdf(r).reshape(org_image.shape)
    else:
        matched_image = np.zeros_like(org_image)
        for i in range(3):
            hist, bin_edges = np.histogram(match_image[:,:,i].ravel(), bins=n_bins, density=True)
            cum_values = np.zeros(bin_edges.shape)
            cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
            inv_cdf = scipy.interpolate.interp1d(cum_values, bin_edges,bounds_error=True)
            r = np.asarray(uniform_hist(org_image[:,:,i].ravel()))
            r[r>cum_values.max()] = cum_values.max()
            matched_image[:,:,i] = inv_cdf(r).reshape(org_image[:,:,i].shape)

    return matched_image


def gkern(kernlen=5, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(scipy.stats.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel
'''
def testGaussian():
    tmp = plt.imshow(gkern(5), interpolation='none')
    plt.show()
'''