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
    parser.add_argument("-i", "--image", help="Path to image")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("-n", "--noise", help="Path to noise image")
    parser.add_argument("-z", "--noiseType", help="Noise type (0 histogram, 1 crappy, 2 less crappy, 3 white)", type=int, default=0)
    parser.add_argument("-t", "--iterations", help="Max number of iterations to run", type=int)
    parser.add_argument("-w", "--width", help="Width of output image", type=int)
    parser.add_argument("-l", "--height", help="Height of output", type=int)
    parser.add_argument("-p", "--pyramid", help="Use multiscale image pyramid", type=int, default=0)
    parser.add_argument("-a", "--wLayers", help="Array for the weights of each layer, should look like 1,0,0.5,2,0 ", default='1,1')
    parser.add_argument("-b", "--wPyramid", help="Array for the weights of each pyramid level, should look like 1,0,0.5,2,0", default='1,1')
    args = parser.parse_args()

    inputImage = None

    if args.output:
        output_directory = args.output
    else:
        output_directory = "result"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if args.image:
        inputImage = utils.load_image_big(args.image, crop=False) # [0, 1)

    else:
        print("Image not defined")
        return

    # Iterations
    if args.iterations:
        iter = args.iterations
    else:
        iter = 20

    # Dimensions for the output image
    if args.width:
        width = args.width
    else:
        width = np.shape(inputImage)[0]

    if args.height:
        height = args.height
    else:
        height = np.shape(inputImage)[1]

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
        noiseImage = utils.load_image_big(args.noise)
    else:
        if args.noiseType == 0:
            noise = utils.his_noise(inputImage, sizeW = np.shape(inputImage)[1], sizeH = np.shape(inputImage)[0])
        elif args.noiseType == 1:
            noise = utils.crappy_noise(width, height)
        elif args.noiseType == 2:
            noise = utils.less_crappy_noise(inputImage, width, height,)
        else:
            noise = utils.white_noise(height, width)

        skimage.io.imsave(os.path.join(output_directory, "his_noise.png"), noise)
        noiseImage = noise

    #If 4-D, the shape is [batch_size, height, width, channels]
    noiseImage = noiseImage.reshape((1, np.shape(noiseImage)[0], np.shape(noiseImage)[1], 3)).astype("float32")

    # Define image
    skimage.io.imsave(os.path.join(output_directory, "origin.png"), inputImage)
    inputImage = inputImage.reshape((1, np.shape(inputImage)[0], np.shape(inputImage)[1], 3))

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
    weightsPyramid = args.wPyramid.split(',')
    weightsPyramid = list(map(float, weightsPyramid))

    print(weightsLayers)
    print('\n')
    print(weightsPyramid)


    # Run synthesis
    #inputImage = np array of target image
    #noiseImage = np array of noise
    start = time.time()
    run_tensorflow(inputImage, noiseImage, output_directory, args.pyramid, weightsLayers, weightsPyramid, iter)
    elapsed = time.time() - start

    with open(os.path.join(output_directory, "Parameters.txt"), "a") as text_file:
        print("\n")
        tmp = time.strftime("%H:%M%:%S", time.gmtime(elapsed))
        print("Elapsed time = %s"%tmp )
        print("Elapsed time = %s"%tmp , file=text_file)

def run_tensorflow(image, noiseImage, output_directory, depth, weightsLayers, weightsPyramid, iter, vgg_class = vgg19.Vgg19):

    print('Begin execution of run_tensorflow')
    print(np.shape(image))

    # Variable for storing the target image
    target = tf.get_variable(name = "target_image", dtype = tf.float64, initializer=image, trainable=False)
    target = tf.cast(target, tf.float32)
    noise = tf.get_variable(name = "noise_image", dtype = tf.float32, initializer=tf.constant(noiseImage), trainable=True)
    #noise = tf.cast(noise, tf.float32)

    targetList = [target]
    noiseList = [noise]
    fpassListTarget = [] #list of vgg objects
    fpassListNoise = []
    outListTarget = [] #list of output layer of vgg objects
    outListNoise = []

    ## TODO ##
    # move the pyramid code to a funciton
    #it recieves the targetList and namescope name, returns the updated list
    with tf.name_scope('build_pyramid_target'):
        gaussKerr = tf.get_variable(initializer = np.reshape(utils.gkern(5), (5,5,1,1)), trainable=False, dtype='float64', name='gauss_kernel')
        gaussKerr = tf.cast(gaussKerr, tf.float32)
        downsamp_filt = tf.get_variable(initializer = np.reshape(np.array([[1.,0.],[0.,0.]]), (2,2,1,1)), trainable=False, dtype='float64', name='downsample_filter')
        downsamp_filt = tf.cast(downsamp_filt, tf.float32)

        for i in range(depth):
            with tf.name_scope('cycle%d'%(i)):
                [tR, tG, tB] = tf.unstack(targetList[i], num=3, axis=3)
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

                targetList.append(tmp)

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


    # fpassList is a list fo vgg instances
    # here we run the build method for each instance and
    # store the output (last layer) on outList
    with tf.name_scope('forward_pass_target'):
        for j in range(len(targetList)):
            with tf.name_scope('cycle%d'%(j)):
                fpassListTarget.append(vgg_class())
                out = fpassListTarget[j].build(targetList[j])
                outListTarget.append(out)

    with tf.name_scope('forward_pass_noise'):
        for j in range(len(targetList)):
            with tf.name_scope('cycle%d'%(j)):
                fpassListNoise.append(vgg_class())
                out = fpassListNoise[j].build(noiseList[j])
                outListNoise.append(out)


    ###################################################
    ## Loss function
    with tf.name_scope('lossFunction'):
        #Check that there are as many weigthLayers
        assert len(weightsLayers) >= fpassListTarget[0].getLayersCount()
        assert len(weightsPyramid) >= len(fpassListTarget)

        loss_sum = 0.0
        #for i in range(0,5): #layers
        for j in range(len(fpassListTarget)): #pyramid levels
            with tf.name_scope('cyclePyramid%d'%(i)):
                loss_pyra = 0.0
                for i in range(0, fpassListTarget[0].getLayersCount()): #layers
                    with tf.name_scope('cycleLayer%d'%(i)):
                        origin = fpassListTarget[j].conv_list[i]
                        new = fpassListNoise[j].conv_list[i]
                        shape = origin.get_shape().as_list()
                        N = shape[3]    #number of channels (filters)
                        M = shape[1] * shape[2] #width x height
                        F = tf.reshape(origin, (-1, N), name='CreateF_target') #N x M
                        Gram_o = (tf.matmul(tf.transpose(F, name='transpose_target'), F, name='Gram_target') / (N * M))
                        F_t = tf.reshape(new, (-1, N), name='CreateF_noise')
                        Gram_n = tf.matmul(tf.transpose(F_t, name='transpose_noise'), F_t, name='Gram_noise') / (N * M)
                        loss = tf.nn.l2_loss((Gram_o - Gram_n), name='lossGramsubstraction') / 2
                        loss = tf.scalar_mul(weightsLayers[i], loss)
                        loss_pyra = tf.add(loss_pyra, loss)
                loss_pyra = tf.scalar_mul(weightsPyramid[j], loss_pyra)
                loss_sum = tf.add(loss_sum, loss_pyra)
        tf.summary.scalar("loss_sum", loss_sum)

    yolo = tf.Variable(np.zeros((20,20)), name='yolo')
    yolo2 = tf.get_variable("big_matrix", shape=(784, 10), initializer=tf.zeros_initializer())

    print(yolo)
    print(yolo2)
    print(noise)
    dummy = tf.get_variable(name = "dummy", dtype = tf.float64, initializer=np.zeros((5,5)), trainable=True)
    train_step = tf.train.AdamOptimizer(0.01).minimize(loss_sum, var_list=[noise])
    #train_step = tf.train.AdagradOptimizer(0.01).minimize(loss_sum, var_list=[noise])


    restrict = tf.maximum(0., tf.minimum(1., noise), name="Restrict_noise")
    r_noise = noise.assign(restrict)



    tmpFile = os.path.join(output_directory, "tensor/")
    if not os.path.exists(tmpFile):
        os.makedirs(tmpFile)




    summary_writer = tf.summary.FileWriter(tmpFile, tf.get_default_graph())

    merged_summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        Iterations = iter
        counter = 0
        temp_loss = 0
        allLoss = []


        for i in range(0, Iterations):
            a = sess.run([train_step])
            print(type(a))
            sess.run([r_noise])
            print(np.shape(r_noise))

            if i == 0:
                temp_loss = loss_sum.eval()
            if i%10==0:
                loss = loss_sum.eval()
                if loss > temp_loss:
                    counter+=1
                sys.stdout.write('\r')
                sys.stdout.write("[%-50s] %d/%d ,loss=%e" % ('='*int(i*50/iter), i, iter, loss))
                sys.stdout.flush()
                temp_loss = loss
            if i%10 == 0 and i!=0 and i <= 200:
                answer = noise.eval()
                answer = answer.reshape(np.shape(answer)[1], np.shape(answer)[2], 3)
                #answer = (answer*255).astype('uint8')
                answer = (utils.histogram_matching(answer, image)*255.).astype('uint8')
                # print('Mean = ', np.mean(answer))
                filename = os.path.join(output_directory, "%safter.jpg"%(str(i)))
                skimage.io.imsave(filename, answer)

                #Save the pyramid
                for w in range(1, len(noiseList)):
                    outputPyramid = noiseList[w].eval()
                    tmp = outputPyramid.reshape(np.shape(outputPyramid)[1], np.shape(outputPyramid)[2], 3)
                    tmp = (utils.histogram_matching(tmp, image)*255.).astype('uint8')
                    filename = os.path.join(output_directory, "%safter%spyra.jpg"%(str(i), str(w)))
                    skimage.io.imsave(filename, tmp)
            if i%200 == 0 and i!=0 and i>200:
                answer = noise.eval()
                answer = answer.reshape(np.shape(answer)[1], np.shape(answer)[2], 3)
                #answer = (answer*255).astype('uint8')
                answer = (utils.histogram_matching(answer, image)*255.).astype('uint8')
                # print('Mean = ', np.mean(answer))
                filename = os.path.join(output_directory, "%safter.jpg"%(str(i)))
                skimage.io.imsave(filename, answer)


                #Save the pyramid
                for w in range(1, len(noiseList)):
                    outputPyramid = noiseList[w].eval()
                    tmp = outputPyramid.reshape(np.shape(outputPyramid)[1], np.shape(outputPyramid)[2], 3)
                    tmp = (utils.histogram_matching(tmp, image)*255.).astype('uint8')
                    filename = os.path.join(output_directory, "%safter%spyra.jpg"%(str(i), str(w)))
                    skimage.io.imsave(filename, tmp)

            #allLoss.append(loss_sum.eval())
            allLoss.append(temp_loss)


            if counter > 3000:
                print('\n','Early Stop!')
                break


            summary_str = sess.run(merged_summary_op)
            summary_writer.add_summary(summary_str, 1)


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
        plotting(allLoss, iter, output_directory)

#########################################################3


if __name__ == "__main__":
    main()




#########################################################################################


