import numpy as np
import scipy
import tensorflow as tf
import skimage
import vgg19
import vgg16
import utils
import time
import os
import sys
import shutil
import datetime
import matplotlib
import matplotlib.pyplot as plt


import pickle



import json # ?
import argparse
from skimage.transform import pyramid_gaussian

"""
Refactored version of the original texture synthesis code
"""
def plotting(loss, iterations, outputPath):
    #Save the number of iterations
    tmp = os.path.join(outputPath, "iterations.npy")
    with open(tmp, 'wb') as fp:
        pickle.dump(iterations, fp)

    #Save the loss over all iterations
    tmp = os.path.join(outputPath, "loss.npy")
    with open(tmp, 'wb') as fp:
        pickle.dump(loss, fp)

def main():
    # Define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="Path to style image")
    parser.add_argument("-c", "--content", help="Path to content image")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("-n", "--noise", help="Path to noise image")
    parser.add_argument("-z", "--noiseType", help="Noise type (0 histogram, 1 crappy, 2 less crappy, 3 white)", type=int, default=0)
    parser.add_argument("-t", "--iterations", help="Max number of iterations to run", type=int)
    parser.add_argument("-w", "--width", help="Width of output image", type=int)
    parser.add_argument("-l", "--height", help="Height of output", type=int)
    parser.add_argument("-p", "--pyramid", help="Use multiscale image pyramid", type=int, default=0)
    parser.add_argument("-a", "--wLayers", help="Array for the weights of each layer for style, should look like 1,0,0.5,2,0 ", default='1,1')
    parser.add_argument("-d", "--wLayersContent", help="Array for the weights of each layer for content, should look like 1,0,0.5,2,0 ", default='0,0,1')
    parser.add_argument("-b", "--wPyramidStyle", help="Array for the weights of each pyramid level, should look like 1,0,0.5,2,0", default='1,1')
    parser.add_argument("-q", "--betaStyle", help="Weight for the content loss", type=float, default='1')
    parser.add_argument("-y", "--wPyramidContent", help="Array for the weights of each pyramid level of the content, should look like 1,0,0.5,2,0", default='1,1')
    args = parser.parse_args()

    styleImage = None

    if args.output:
        output_directory = args.output
    else:
        output_directory = "result"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if args.image:
        styleImage = utils.load_image_big(args.image, crop=False) # [0, 1)

    if args.content:
        contentImage = utils.load_image_big(args.content, crop=False) # [0, 1)

    else:
        print("Image not defined")
        return

    # Iterations
    if args.iterations:
        iter = args.iterations
    else:
        iter = 20

    if args.betaStyle:
        beta = args.betaStyle
    else:
        beta = 1

    # Dimensions for the output image
    if args.width:
        width = args.width
    else:
        width = np.shape(contentImage)[0]

    if args.height:
        height = args.height
    else:
        height = np.shape(contentImage)[1]

    #Clear the output folder
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    try:
        filesToRemove = [f for f in os.listdir(output_directory)]
        for f in filesToRemove:
            if os.path.isfile(output_directory):
                os.remove(os.path.join(output_directory, f))
            else:
                shutil.rmtree(os.path.join(output_directory, f))

        #if os.path.isfile(output_directory):
        #    os.unlink(output_directory)
        #elif os.path.isdir(output_directory): shutil.rmtree(output_directory)
    except Exception as e:
        print(e)

    # Define noise
    noiseImage = None
    if args.noise:
        noiseImage = utils.load_image_big(args.noise, crop=False)
    else:
        if args.noiseType == 0:
            noise = utils.his_noise(contentImage, sizeW = np.shape(contentImage)[0], sizeH = np.shape(contentImage)[1])
        elif args.noiseType == 1:
            noise = utils.crappy_noise(width, height)
        elif args.noiseType == 2:
            noise = utils.less_crappy_noise(contentImage, width, height,)
        else:
            noise = utils.white_noise(height, width)

        skimage.io.imsave(os.path.join(output_directory, "his_noise.png"), noise)
        noiseImage = noise

    print( "Noise Image Shape: ", noiseImage.shape)

    #If 4-D, the shape is [batch_size, height, width, channels]
    noiseImage = noiseImage.reshape((1, np.shape(noiseImage)[0], np.shape(noiseImage)[1], 3)).astype("float32")

    # Define image
    skimage.io.imsave(os.path.join(output_directory, "originStyle.png"), styleImage)
    styleImage = styleImage.reshape((1, np.shape(styleImage)[0], np.shape(styleImage)[1], 3))

    skimage.io.imsave(os.path.join(output_directory, "originContent.png"), contentImage)
    contentImage = contentImage.reshape((1, np.shape(contentImage)[0], np.shape(contentImage)[1], 3))



    with open(os.path.join(output_directory, "Parameters.txt"), "w") as text_file:
        a = datetime.datetime.now()
        a = a.strftime("%Y-%m-%d    %H:%M%:%S")
        print(a)
        print("Texture Synthesis Experiment", file=text_file)
        print("Time = {}".format(a), file=text_file)
        for arg in vars(args):
            print(arg, "\t\t\t", getattr(args, arg))
            print(arg, "\t\t\t", getattr(args, arg), file=text_file)

    #Weights for the loss function
    weightsLayers = args.wLayers.split(',')
    weightsLayers = list(map(float, weightsLayers))
    weightsPyramid = args.wPyramidStyle.split(',')
    weightsPyramid = list(map(float, weightsPyramid))
    weightsPyramidContent = args.wPyramidContent.split(',')
    weightsPyramidContent = list(map(float, weightsPyramidContent))
    weightsLayersContent = args.wLayersContent.split(',')
    weightsLayersContent = list(map(float, weightsLayersContent))

    print(weightsLayers)
    print('\n')
    print(weightsPyramid)


    # Run synthesis
    #styleImage = np array of style image
    #noiseImage = np array of noise
    start = time.time()
    run_tensorflow(styleImage, noiseImage, contentImage, output_directory, args.pyramid, weightsLayers, weightsLayersContent,
                   weightsPyramid, weightsPyramidContent, iter, beta)
    elapsed = time.time() - start

    with open(os.path.join(output_directory, "Parameters.txt"), "a") as text_file:
        print("\n")
        tmp = time.strftime("%H:%M%:%S", time.gmtime(elapsed))
        print("Elapsed time = %s"%tmp )
        print("Elapsed time = %s"%tmp , file=text_file)

def run_tensorflow(image, noiseImage, contentImage, output_directory, depth, weightsLayers, weightsLayersContent,
                   weightsPyramid, weightsPyramidContent, iter, betaPar, vgg_class = vgg19.Vgg19):

    print('Begin execution of run_tensorflow')
    print(np.shape(image))

    # Variable for storing the style image
    style = tf.get_variable(name = "style_image", dtype = tf.float64, initializer=image, trainable=False)
    style = tf.cast(style, tf.float32)
    noise = tf.get_variable(name = "noise_image", dtype = tf.float32, initializer=tf.constant(noiseImage), trainable=True)
    content = tf.get_variable(name = "content_image", dtype = tf.float64, initializer=tf.constant(contentImage), trainable=False)
    content = tf.cast(content, tf.float32)
    #noise = tf.cast(noise, tf.float32)

    styleList = [style]
    noiseList = [noise]
    contentList = [content]
    fpassListContent = []
    fpassListstyle = [] #list of vgg objects
    fpassListNoise = []
    outListstyle = [] #list of output layer of vgg objects
    outListNoise = []
    outListContent = []

    ## TODO ##
    # move the pyramid code to a funciton
    #it recieves the styleList and namescope name, returns the updated list
    with tf.name_scope('build_pyramid_style'):
        gaussKerr = tf.get_variable(initializer = np.reshape(utils.gkern(5), (5,5,1,1)), trainable=False, dtype='float64', name='gauss_kernel')
        gaussKerr = tf.cast(gaussKerr, tf.float32)
        downsamp_filt = tf.get_variable(initializer = np.reshape(np.array([[1.,0.],[0.,0.]]), (2,2,1,1)), trainable=False, dtype='float64', name='downsample_filter')
        downsamp_filt = tf.cast(downsamp_filt, tf.float32)

        for i in range(depth):
            with tf.name_scope('cycle%d'%(i)):
                [tR, tG, tB] = tf.unstack(styleList[i], num=3, axis=3)
                tR = tf.expand_dims(tR, 3)
                tG = tf.expand_dims(tG, 3)
                tB = tf.expand_dims(tB, 3)

                #convolve each input image with the gaussian filter
                tR_gauss = tf.nn.conv2d(tR, gaussKerr, strides=[1, 1, 1, 1], padding='SAME')
                tG_gauss = tf.nn.conv2d(tG, gaussKerr, strides=[1, 1, 1, 1], padding='SAME')
                tB_gauss = tf.nn.conv2d(tB, gaussKerr, strides=[1, 1, 1, 1], padding='SAME')

                tR_downs = tf.nn.conv2d(tR_gauss, downsamp_filt, strides=[1, 2, 2, 1], padding='SAME')
                tG_downs = tf.nn.conv2d(tG_gauss, downsamp_filt, strides=[1, 2, 2, 1], padding='SAME')
                tB_downs = tf.nn.conv2d(tB_gauss, downsamp_filt, strides=[1, 2, 2, 1], padding='SAME')

                tmp = tf.concat([tR_downs, tG_downs, tB_downs], axis=3)

                styleList.append(tmp)

    ## TODO ##
    ## Find out what to do with the reuse
    with tf.name_scope('build_pyramid_noise'):
        #gaussKerr = tf.get_variable(initializer = np.reshape(utils.gkern(5), (5,5,1,1)), trainable=False, dtype='float64', name='gauss_kernel')
        #gaussKerr = tf.cast(gaussKerr, tf.float32, reuse=True)
        #downsamp_filt = tf.get_variable(initializer = np.reshape(np.array([[1.,0.],[0.,0.]]), (2,2,1,1)), trainable=False, dtype='float64', name='downsample_filter')
        #downsamp_filt = tf.cast(downsamp_filt, tf.float32, reuse=True)

        for i in range(depth):
            with tf.name_scope('cycle%d'%(i)):
                [tR, tG, tB] = tf.unstack(noiseList[i], num=3, axis=3)
                tR = tf.expand_dims(tR, 3)
                tG = tf.expand_dims(tG, 3)
                tB = tf.expand_dims(tB, 3)

                #convolve each input image with the gaussian filter
                tR_gauss = tf.nn.conv2d(tR, gaussKerr, strides=[1, 1, 1, 1], padding='SAME')
                tG_gauss = tf.nn.conv2d(tG, gaussKerr, strides=[1, 1, 1, 1], padding='SAME')
                tB_gauss = tf.nn.conv2d(tB, gaussKerr, strides=[1, 1, 1, 1], padding='SAME')

                tR_downs = tf.nn.conv2d(tR_gauss, downsamp_filt, strides=[1, 2, 2, 1], padding='SAME')
                tG_downs = tf.nn.conv2d(tG_gauss, downsamp_filt, strides=[1, 2, 2, 1], padding='SAME')
                tB_downs = tf.nn.conv2d(tB_gauss, downsamp_filt, strides=[1, 2, 2, 1], padding='SAME')

                tmp = tf.concat([tR_downs, tG_downs, tB_downs], axis=3)

                noiseList.append(tmp)

    with tf.name_scope('build_pyramid_content'):
        #gaussKerr = tf.get_variable(initializer = np.reshape(utils.gkern(5), (5,5,1,1)), trainable=False, dtype='float64', name='gauss_kernel')
        #gaussKerr = tf.cast(gaussKerr, tf.float32, reuse=True)
        #downsamp_filt = tf.get_variable(initializer = np.reshape(np.array([[1.,0.],[0.,0.]]), (2,2,1,1)), trainable=False, dtype='float64', name='downsample_filter')
        #downsamp_filt = tf.cast(downsamp_filt, tf.float32, reuse=True)

        for i in range(depth):
            with tf.name_scope('cycle%d'%(i)):
                [tR, tG, tB] = tf.unstack(contentList[i], num=3, axis=3)
                tR = tf.expand_dims(tR, 3)
                tG = tf.expand_dims(tG, 3)
                tB = tf.expand_dims(tB, 3)

                #convolve each input image with the gaussian filter
                tR_gauss = tf.nn.conv2d(tR, gaussKerr, strides=[1, 1, 1, 1], padding='SAME')
                tG_gauss = tf.nn.conv2d(tG, gaussKerr, strides=[1, 1, 1, 1], padding='SAME')
                tB_gauss = tf.nn.conv2d(tB, gaussKerr, strides=[1, 1, 1, 1], padding='SAME')

                tR_downs = tf.nn.conv2d(tR_gauss, downsamp_filt, strides=[1, 2, 2, 1], padding='SAME')
                tG_downs = tf.nn.conv2d(tG_gauss, downsamp_filt, strides=[1, 2, 2, 1], padding='SAME')
                tB_downs = tf.nn.conv2d(tB_gauss, downsamp_filt, strides=[1, 2, 2, 1], padding='SAME')

                tmp = tf.concat([tR_downs, tG_downs, tB_downs], axis=3)

                contentList.append(tmp)


    # fpassList is a list fo vgg instances
    # here we run the build method for each instance and
    # store the output (last layer) on outList
    with tf.name_scope('forward_pass_style'):
        for j in range(len(styleList)):
            with tf.name_scope('cycle%d'%(j)):
                fpassListstyle.append(vgg_class())
                out = fpassListstyle[j].build(styleList[j])
                outListstyle.append(out)

    with tf.name_scope('forward_pass_noise'):
        for j in range(len(styleList)):
            with tf.name_scope('cycle%d'%(j)):
                fpassListNoise.append(vgg_class())
                out = fpassListNoise[j].build(noiseList[j])
                outListNoise.append(out)

    with tf.name_scope('forward_pass_content'):
        for j in range(len(contentList)):
            with tf.name_scope('cycle%d'%(j)):
                fpassListContent.append(vgg_class())
                out = fpassListContent[j].build(contentList[j])
                outListContent.append(out)


    ###################################################
    ## Loss function
    with tf.name_scope('lossStyle'):
        #Check that there are as many weigthLayers
        assert len(weightsLayers) >= fpassListstyle[0].getLayersCount()
        assert len(weightsPyramid) >= len(fpassListstyle)

        loss_style = 0.0
        #for i in range(0,5): #layers
        for j in range(len(fpassListstyle)): #pyramid levels
            with tf.name_scope('cyclePyramid%d'%(j)):
                loss_pyra = 0.0
                for i in range(0, fpassListstyle[0].getLayersCount()): #layers
                    with tf.name_scope('cycleLayer%d'%(i)):
                        origin = fpassListstyle[j].conv_list[i]
                        new = fpassListNoise[j].conv_list[i]
                        shape = origin.get_shape().as_list()
                        N = shape[3]    #number of channels (filters)
                        M = shape[1] * shape[2] #width x height
                        F = tf.reshape(origin, (-1, N), name='CreateF_style') #N x M
                        Gram_o = (tf.matmul(tf.transpose(F, name='transpose_style'), F, name='Gram_style') / (N * M))
                        F_t = tf.reshape(new, (-1, N), name='CreateF_noise')
                        Gram_n = tf.matmul(tf.transpose(F_t, name='transpose_noise'), F_t, name='Gram_noise') / (N * M)
                        loss = tf.nn.l2_loss((Gram_o - Gram_n), name='lossGramsubstraction') / 4
                        loss = tf.scalar_mul(weightsLayers[i], loss)
                        loss_pyra = tf.add(loss_pyra, loss)
                loss_pyra = tf.scalar_mul(weightsPyramid[j], loss_pyra)
                loss_style = tf.add(loss_style, loss_pyra)
        tf.summary.scalar("loss_style", loss_style)

    with tf.name_scope('lossContent'):
        #Check that there are as many weigthLayers
        assert len(weightsLayersContent) >= fpassListContent[0].getLayersCount()
        assert len(weightsPyramidContent) >= len(fpassListContent)

        loss_content = 0.0
        #for i in range(0,5): #layers
        for j in range(len(fpassListContent)): #pyramid levels
            with tf.name_scope('cyclePyramid%d'%(j)):
                loss_pyra = 0.0
                for i in range(0, fpassListContent[0].getLayersCount()): #layers
                    with tf.name_scope('cycleLayer%d'%(i)):
                        con = fpassListContent[j].conv_list[i]
                        new = fpassListNoise[j].conv_list[i]
                        shape = con.get_shape().as_list()
                        N = shape[3]    #number of channels (filters)
                        M = shape[1] * shape[2] #width x height
                        P = tf.reshape(con, (-1, N), name='CreateF_content') #N x M
                        #Gram_o = (tf.matmul(tf.transpose(F, name='transpose_style'), F, name='Gram_style') / (N * M))
                        F = tf.reshape(new, (-1, N), name='CreateF_noise')
                        #Gram_n = tf.matmul(tf.transpose(F_t, name='transpose_noise'), F_t, name='Gram_noise') / (N * M)
                        loss = tf.nn.l2_loss((F - P), name='lossGramsubstraction') / 2
                        loss = tf.scalar_mul(weightsLayersContent[i], loss)
                        loss_pyra = tf.add(loss_pyra, loss)
                loss_pyra = tf.scalar_mul(weightsPyramidContent[j], loss_pyra)
                loss_content = tf.add(loss_content, loss_pyra)
        tf.summary.scalar("loss_content", loss_content)

    #betaPar = 0.5
    alpha = tf.constant(1, dtype=tf.float32, name="alpha")
    beta = tf.constant(betaPar, dtype=tf.float32, name="beta")

    loss_sum = tf.scalar_mul(loss_content, alpha) + tf.scalar_mul(loss_style, beta)

    train_step = tf.train.AdamOptimizer(0.01).minimize(loss_sum, var_list=[noise])
    #train_step = tf.train.AdagradOptimizer(0.01).minimize(loss_sum, var_list=[noise])


    restrict = tf.maximum(0., tf.minimum(1., noise), name="Restrict_noise")
    r_noise = noise.assign(restrict)



    tmpFile = os.path.join(output_directory, "tensor/")
    if not os.path.exists(tmpFile):
        os.makedirs(tmpFile)

    #https://www.tensorflow.org/api_docs/python/tf/contrib/opt/ScipyOptimizerInterface
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            loss_sum, var_to_bounds={noise: (0, 1)}, method='L-BFGS-B', options={'maxiter': iter})

    #trainOP = optimizer.minimze

    summary_writer = tf.summary.FileWriter(tmpFile, tf.get_default_graph())

    merged_summary_op = tf.summary.merge_all()

    Iterations = iter
    counter = 0
    temp_loss = 0
    allLoss = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        optimizer.minimize(sess)


        #tmp = fpassListContent[0].eval()
        #tf.summary.image('content', tmp, 3)

        answer = noise.eval()
        answer = answer.reshape(np.shape(answer)[1], np.shape(answer)[2], 3)
        skimage.io.imsave(os.path.join(output_directory, "final_texture_noHistMatch.png"), answer)
        answer = (utils.histogram_matching(answer, image)*255.).astype('uint8')
        skimage.io.imsave(os.path.join(output_directory, "final_texture.png"), answer)

        #Save the pyramid
        for w in range(1, len(noiseList)):
            outputPyramid = noiseList[w].eval()
            tmp = outputPyramid.reshape(np.shape(outputPyramid)[1], np.shape(outputPyramid)[2], 3)
            tmp = (utils.histogram_matching(tmp, image)*255.).astype('uint8')
            skimage.io.imsave(os.path.join(output_directory, "final_texture_pyra%s.png"%(str(w))), tmp)

        #Some plotting
        #plotting(allLoss, iter, output_directory)

#########################################################3


if __name__ == "__main__":
    main()




#########################################################################################


