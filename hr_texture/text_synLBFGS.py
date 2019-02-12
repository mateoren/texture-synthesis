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
#Downsampling function
def downSample(x):
  # x will be a numpy array with the contents of the placeholder below
    return x[:, ::2, ::2]



def run_synthesis_pyramid(tex, images, proc_img, iterations, output_directory, weightsLayers, weightsPyramid, vgg_class=vgg19.Vgg19):
    config = tf.ConfigProto()

    gaussKerr = tf.cast(tf.Variable(np.reshape(utils.gkern(5), (5,5,1,1)), trainable=False, dtype='float64'), tf.float32)

    #os.environ["CUDA_VISIBLE_DEVICES"]="0"
    #exit(0)
    with tf.Session(config=config) as sess:
        vggs = [vgg_class() for i in range(len(images))]
        vggs2 = [vgg_class() for i in range(len(images))]

        for i in range(len(images)):
            with tf.name_scope("origin" + str(i)):
                vggs[i].build(images[i])
            with tf.name_scope("new" + str(i)):
                vggs2[i].build(tex[i])

        #Check that there are as many weigthLayers
        assert len(weightsLayers) == vggs[0].getLayersCount()
        assert len(weightsPyramid) == len(images)

        loss_sum = 0.0
        #for i in range(0,5): #layers
        for j in range(len(images)): #pyramid levels
            for i in range(0, vggs[0].getLayersCount()): #layers
                origin = vggs[j].conv_list[i]
                new = vggs2[j].conv_list[i]
                shape = origin.get_shape().as_list()
                N = shape[3]
                M = shape[1] * shape[2]
                F = tf.reshape(origin, (-1, N))
                Gram_o = (tf.matmul(tf.transpose(F), F) / (N * M))
                F_t = tf.reshape(new, (-1, N))
                Gram_n = tf.matmul(tf.transpose(F_t), F_t) / (N * M)
                loss = tf.nn.l2_loss((Gram_o - Gram_n)) / 2
                loss = tf.scalar_mul(weightsLayers[i], loss)
                loss_sum = tf.add(loss_sum, loss)
        #train_step=tf.train.AdamOptimizer(0.01).minimize(loss_sum, var_list=[tex[0]])

        '''
        train_step = tf.contrib.opt.ScipyOptimizerInterface(
                loss,
                method='L-BFGS-B',
                var_list=[tex[0]]
                )
        '''



        restrict = tf.maximum(0., tf.minimum(1., tex[0]))
        r_tex = tex[0].assign(restrict)

        #sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())

        '''
        train_step = tf.contrib.opt.ScipyOptimizerInterface(
                loss_sum,
                method='L-BFGS-B',
                options={'maxiter': iterations},
                var_list=[tex[0]]
                ).minimize(sess)
        '''

        train_step = tf.contrib.opt.ScipyOptimizerInterface(
                loss_sum,
                method='L-BFGS-B',
                options={'maxiter': iterations},
                var_list=[tex[0]]
                ).minimize()

        for j in range(len(tex) - 1):
            # Pyramid in TF
            [tR, tG, tB] = tf.unstack(tex[j], num=3, axis=3)
            tR = tf.expand_dims(tR, 3)
            tG = tf.expand_dims(tG, 3)
            tB = tf.expand_dims(tB, 3)

            #convolve each input image with the gaussian filter
            tR_gauss = tf.nn.conv2d(tR, gaussKerr, strides=[1, 1, 1, 1], padding='SAME')
            tG_gauss = tf.nn.conv2d(tG, gaussKerr, strides=[1, 1, 1, 1], padding='SAME')
            tB_gauss = tf.nn.conv2d(tB, gaussKerr, strides=[1, 1, 1, 1], padding='SAME')

            #tmpR = tf.py_func(downSample, tR_gauss, tf.float32)
            #tmpG = tf.py_func(downSample, tG_gauss, tf.float32)
            #tmpB = tf.py_func(downSample, tB_gauss, tf.float32)

            tmp = tf.stack([tR_gauss, tG_gauss, tB_gauss], axis=3)
            tmp = tf.concat([tR_gauss, tG_gauss, tB_gauss], axis=3)

            print("<<<<<<<<<<<<<<<HYPNOTOAD>>>>>>>>>>>>>>>>>>>>>>>>")
            print(tmp)
            print(tmp.get_shape())
            print(tmp.get_shape().as_list()[1:])
            print("<<<<<<<<<<<<<<<HYPNOTOAD>>>>>>>>>>>>>>>>>>>>>>>>")

            newTmp = tf.py_func(downSample, [tmp], tf.float32)

            print("<<<<<<<<<<<<<<<HYPNOTOAD>>>>>>>>>>>>>>>>>>>>>>>>")
            #print(newTmp)
            #print(newTmp.get_shape())
            #print(newTmp.get_shape().as_list()[1:])
            print("<<<<<<<<<<<<<<<HYPNOTOAD>>>>>>>>>>>>>>>>>>>>>>>>")
            yolo = tex[j + 1].assign(newTmp)




        Iteration = iterations
        counter = 0
        temp_loss = 0
        allLoss = []
        for i in range(0, Iteration):
            print('ITERATION'+str(i))
            #sess.run(train_step.minimize(sess, fetches=[tex[0]]))
            sess.run(train_step)
            sess.run(r_tex)
            sess.run(yolo)

            if i == 0:
                temp_loss = loss_sum.eval()
            if i%10==0:
                loss = loss_sum.eval()
                if loss > temp_loss:
                    counter+=1
                sys.stdout.write('\r')
                sys.stdout.write("[%-50s] %d/%d ,loss=%e" % ('='*int(i*50/Iteration), i, Iteration, loss))
                sys.stdout.flush()
                temp_loss = loss
            if i%200 == 0 and i!=0:
                answer = tex[0].eval()
                answer = answer.reshape(np.shape(answer)[1], np.shape(answer)[2], 3)
                #answer = (answer*255).astype('uint8')
                answer = (utils.histogram_matching(answer, proc_img)*255.).astype('uint8')
                # print('Mean = ', np.mean(answer))
                filename = os.path.join(output_directory, "%safter.jpg"%(str(i)))
                skimage.io.imsave(filename, answer)

                #Save the pyramid
                for w in range(1, len(tex)):
                    outputPyramid = tex[w].eval()
                    tmp = outputPyramid.reshape(np.shape(outputPyramid)[1], np.shape(outputPyramid)[2], 3)
                    tmp = (utils.histogram_matching(tmp, proc_img)*255.).astype('uint8')
                    filename = os.path.join(output_directory, "%safter%spyra.jpg"%(str(i), str(w)))
                    skimage.io.imsave(filename, tmp)
            #allLoss.append(loss_sum.eval())
            allLoss.append(temp_loss)


            if counter > 3000:
                print('\n','Early Stop!')
                break

            '''
            answer = tex[0].eval()
            #print(answer)
            pyramid = create_pyramids((answer.reshape(np.shape(answer)[1], np.shape(answer)[2], 3)*255).astype('uint8'), levels)


            im2 = [i.reshape((1, np.shape(i)[0], np.shape(i)[1], 3)) for i in pyramid]
            im2 = [tf.cast(tf.convert_to_tensor(i, dtype="float64"), tf.float32) for i in im2]
            #t_pyramid = tuple(tf.convert_to_tensor(np.reshape(i, (1, np.shape[0], np.shape[1], 3))) for i in pyramid)
            #t_pyramid = tuple(tf.convert_to_tensor(i) for i in im2)

            #print(t_pyramid[0].get_shape())
            #print("**********************")

            for j in range(1,len(im2)):
                sess.run(tex[j].assign(im2[j]))

            #print(pyramid)
            '''

        answer = tex[0].eval()
        answer = answer.reshape(np.shape(answer)[1], np.shape(answer)[2], 3)
        answer = (utils.histogram_matching(answer, proc_img)*255.).astype('uint8')
        skimage.io.imsave(os.path.join(output_directory, "final_texture.jpg"), answer)

        #Save the pyramid
        for w in range(1, len(tex)):
            outputPyramid = tex[w].eval()
            tmp = outputPyramid.reshape(np.shape(outputPyramid)[1], np.shape(outputPyramid)[2], 3)
            tmp = (utils.histogram_matching(tmp, proc_img)*255.).astype('uint8')
            skimage.io.imsave(os.path.join(output_directory, "final_texture_pyra%s.jpg"%(str(w))), tmp)

        #Some plotting
        plotting(allLoss, iterations, output_directory)


def plotting(loss, iterations, outputPath):
    #Save the number of iterations
    tmp = os.path.join(outputPath, "iterations.npy")
    with open(tmp, 'wb') as fp:
        pickle.dump(iterations, fp)

    #Save the loss over all iterations
    tmp = os.path.join(outputPath, "loss.npy")
    with open(tmp, 'wb') as fp:
        pickle.dump(loss, fp)


def run_synthesis(tex, images, proc_img, iterations, output_directory, weightsLayers, vgg_class=vgg19.Vgg19):
    config = tf.ConfigProto()

    os.environ["CUDA_VISIBLE_DEVICEs"]="0"

    with tf.Session(config=config) as sess:
        vgg = vgg_class()
        vgg2 = vgg_class()
        with tf.name_scope("origin"):
            vgg.build(images)
        with tf.name_scope("new"):
            vgg2.build(tex)

        #Check that there are as many weigthLayers
        assert len(weightsLayers) == vgg.getLayersCount()

        ## Caculate the Loss according to the paper
        loss_sum = 0.
        for i in range(0, vgg.getLayersCount()):
            origin = vgg.conv_list[i]
            new = vgg2.conv_list[i]
            shape = origin.get_shape().as_list()
            N = shape[3]
            M = shape[1]*shape[2]
            F = tf.reshape(origin,(-1,N))
            Gram_o = (tf.matmul(tf.transpose(F),F)/(N*M))
            F_t = tf.reshape(new,(-1,N))
            Gram_n = tf.matmul(tf.transpose(F_t),F_t)/(N*M)
            loss = tf.nn.l2_loss((Gram_o-Gram_n))/2
            loss = tf.scalar_mul(weightsLayers[i], loss)
            loss_sum = tf.add(loss_sum,loss)
        train_step=tf.train.AdamOptimizer(0.01).minimize(loss_sum, var_list=[tex])

        restrict = tf.maximum(0., tf.minimum(1., tex))
        r_tex = tex.assign(restrict)

        #sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())

        Iteration = iterations
        counter = 0
        temp_loss = 0
        allLoss = []
        for i in range(0, Iteration):
            sess.run(train_step)
            sess.run(r_tex)
            if i == 0:
                temp_loss = loss_sum.eval()
            if i%10==0:
                loss = loss_sum.eval()
                if loss > temp_loss:
                    counter+=1
                sys.stdout.write('\r')
                sys.stdout.write("[%-50s] %d/%d ,loss=%e" % ('='*int(i*50/Iteration), i, Iteration, loss))
                sys.stdout.flush()
                temp_loss = loss
            if i%200 == 0 and i!=0:
                answer = tex.eval()
                answer = answer.reshape(np.shape(answer)[1], np.shape(answer)[2], 3)
                answer = (answer*255).astype('uint8')
                # print('Mean = ', np.mean(answer))
                filename = os.path.join(output_directory, "%safter.jpg"%(str(i)))
                skimage.io.imsave(filename, answer)
            if counter > 3000:
                print('\n','Early Stop!')
                break
            #allLoss.append(loss_sum.eval())
            allLoss.append(temp_loss)

        answer = tex.eval()
        answer = answer.reshape(np.shape(answer)[1], np.shape(answer)[2],3)
        answer = (utils.histogram_matching(answer, proc_img)*255.).astype('uint8')
        skimage.io.imsave(os.path.join(output_directory, "final_texture.jpg"), answer)

        #Some plotting
        plotting(allLoss, iterations, output_directory)

def create_pyramids(image, levels):
    """
    Dummy pyramid
    """
    #image = image[:128, :128]
    pyr = tuple(pyramid_gaussian(image, max_layer = levels, downscale = 2))
    # #print(pyr)
    # p = []
    # for j in range(len(pyr)):
    #     print(np.shape(pyr[j]))
    #
    #
    # for i in range(len(pyr)):
    #     print(np.shape(pyr[i]))
    #     p.append(np.reshape(pyr[i], (1, np.shape(pyr[i])[0], np.shape(pyr[i])[1], 3)))
    return pyr


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

    proc_img = None

    if args.output:
        output_directory = args.output
    else:
        output_directory = "result"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if args.image:
        proc_img = utils.load_image_big(args.image) # [0, 1)

    else:
        print("Image not defined")
        return

    # Iterations
    if args.iterations:
        iter = args.iterations
    else:
        iter = 20;

    # Dimensions for the output image
    if args.width:
        width = args.width
    else:
        width = np.shape(proc_img)[0]

    if args.height:
        height = args.height
    else:
        height = np.shape(proc_img)[1]

    #Clear the outpu folder
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    try:
        filesToRemove = [f for f in os.listdir(output_directory)]
        for f in filesToRemove:
            os.remove(os.path.join(output_directory, f))

        #if os.path.isfile(output_directory):
        #    os.unlink(output_directory)
        #elif os.path.isdir(output_directory): shutil.rmtree(output_directory)
    except Exception as e:
        print(e)

    # Define noise
    init = None
    if args.noise:
        init = utils.load_image_big(args.noise)
    else:
        if args.noiseType == 0:
            noise = utils.his_noise(proc_img, sizeW = np.shape(proc_img)[1], sizeH = np.shape(proc_img)[0])
        elif args.noiseType == 1:
            noise = utils.crappy_noise(width, height)
        elif args.noiseType == 2:
            noise = utils.less_crappy_noise(proc_img, width, height,)
        else:
            noise = utils.white_noise(width, height)

        skimage.io.imsave(os.path.join(output_directory, "his_noise.png"), noise)
        init = noise

    batch2 = init.reshape((1, np.shape(init)[0], np.shape(init)[1], 3)).astype("float32")
    tex = tf.Variable(batch2)

    # Define image
    skimage.io.imsave(os.path.join(output_directory, "origin.png"), proc_img)
    batch1 = proc_img.reshape((1, np.shape(proc_img)[0], np.shape(proc_img)[1], 3))
    images = tf.cast(tf.Variable(batch1, trainable=False, dtype='float64'), tf.float32)

    #original_im_size = 256

    images_ = [i.reshape((1, np.shape(i)[0], np.shape(i)[1], 3)) for i in create_pyramids(proc_img, args.pyramid)]
    images_ = [tf.cast(tf.Variable(i, trainable=False, dtype="float64"), tf.float32) for i in images_]
    noises = [tf.Variable(i) for i in [j.reshape((1, np.shape(j)[1], np.shape(j)[1], 3)).astype("float32") for j in create_pyramids(init, args.pyramid)]]

    # Save arguments to log
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
    weightsLayers = list(map(int, weightsLayers))
    weightsPyramid = args.wPyramid.split(',')
    weightsPyramid = list(map(int, weightsPyramid))

    print(weightsLayers)
    print('\n')
    print(weightsPyramid)

    #utils.testGaussian()


    # Run synthesis
    start = time.time()
    if args.pyramid > 0:
        run_synthesis_pyramid(noises, images_, proc_img, iter, output_directory, weightsLayers, weightsPyramid) # Using VGG19
    else:
        run_synthesis(tex, images, proc_img, iter, output_directory) # Using VGG19
    elapsed = time.time() - start

    with open(os.path.join(output_directory, "Parameters.txt"), "a") as text_file:
        print(r"\n")
        tmp = time.strftime("%H:%M%:%S", time.gmtime(elapsed))
        print("Elapsed time = %s"%tmp )
        print("Elapsed time = %s"%tmp , file=text_file)


if __name__ == "__main__":
    main()
