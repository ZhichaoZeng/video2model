##
# @file main.py
# @brief the inferface of our work
#       use SMPLify SMPL and DeepCut
# @author Zhichao Zeng
# @version 1.0.0
# @date 2016-11-17

import os
#os.chdir('/home/zhichaozeng/Research/SMPLify/smplify_public/code' )

#print os.getcwd()

#import fit_3d
import sys
sys.path.append('../')


from os.path import join, exists, abspath, dirname
from os import makedirs
import logging
import cPickle as pickle
from time import time
from glob import glob

import cv2
import numpy as np
import chumpy as ch

from smpl_webuser.serialization import load_model


import fit_3d





##
# @brief set up paths to image and joint data
#
# @param img_dir    image path
# @param data_dir   joint data path
# @param out_dir    output file directory
# @param use_interpenetration   boolean, if True enables the interpentration term
# @param n_betas    number of shape coefficients considered during optimization
# @param flength    camera focal length ( an estimate )
# @param pix-thsh   threshold( in pixel ) if the distance between shoulder joints in 2D is lower than pix_thsh, the body orientation as ambiguous( so a fit is run on both the estimated one and its flip)
# @param use_neutral    boolean, is True enables uses the neutral gender SMPL model
# @param viz    boolean, if True enables visualization during optimization
#
# @return params={'cam_t','f','pose','betas'} a dictionary with camera translation camera focus length, SMPL model parameters
def call_fit_3d(img_dir,
                data_dir,
                out_dir,
                use_neutral=False,
                viz=True,
                use_interpenetration=True,
                n_betas=10,
                flength=5000.,
                pix_thsh=25.):
    if not exists( out_dir ):
        makedirs( out_dir )
    # Render degrees: List of degrees in azimuth to render the final fit.
    # Note that rendering many views can take a while.
    do_degrees = [0.]

    sph_regs = None

    model_female = load_model( MODEL_FEMALE_PATH )
    model_male = load_model( MODEL_MALE_PATH )
    model_neutral = load_model( MODEL_NEUTRAL_PATH )
    sph_regs_female = np.load( SPH_REGS_FEMALE_PATH )
    sph_regs_male = np.load( SPH_REGS_MALE_PATH )
    sph_regs_neutral = np.load( SPH_REGS_NEUTRAL_PATH )

    imFileNames = glob( img_dir + '/*.png' )

    for file_name in imFileNames :
        img = cv2.imread( file_name )
        img_name = os.path.basename( file_name )
        out_path = os.path.join( out_dir, img_name.replace( '_vis.png','.pkl') )

        #load image
        img = cv2.imread( file_name )
        if img.ndim == 2:
           _LOGGER.warn("The image is grayscale!")
           img = np.dstack((img, img, img))



        #load 2D joints
        jointFileName = file_name.replace('vis.png','pose.npz')
        if not os.path.exists( jointFileName ):
            continue
        est = np.load( jointFileName )['pose']
        joints = est[:2,:].T
        conf = est[2,:]

#        gender = raw_input( 'input the gender of the model( male, female or neutral )' )
        gender = 'male'
        if gender == 'female':
            model = model_female
            sph_regs = sph_regs_female
        elif gender == 'male':
            model = model_male
            sph_regs = sph_regs_male
        else :
            model = model_neutral
            sph_regs = sph_regs_neutral


        params, vis = fit_3d.run_single_fit(
            img,
            joints,
            conf,
            model,
            regs=sph_regs,
            n_betas=n_betas,
            flength=flength,
            pix_thsh=pix_thsh,
            scale_factor=2,
            viz=viz,
            do_degrees=do_degrees)
        if viz:
            import matplotlib.pyplot as plt
            plt.ion()
            plt.show()
            plt.subplot(121)
            plt.imshow(img[:, :, ::-1])
            if do_degrees is not None:
                for di, deg in enumerate(do_degrees):
                    plt.subplot(122)
                    plt.cla()
                    plt.imshow(vis[di])
                    plt.draw()
                    plt.title('%d deg' % deg)
                    plt.pause(1)
#            raw_input('Press any key to continue...')

        with open(out_path, 'w') as outf:
            pickle.dump(params, outf)

        # This only saves the first rendering.
        if do_degrees is not None:
            cv2.imwrite(out_path.replace('.pkl', '.png'), vis[0])
        
    return params



        











if __name__ == "__main__":
    # 'models'is in the 'code/' directory 
    MODEL_DIR = join( abspath( dirname( os.getcwd()) ), 'models' )

    #Model path:
    MODEL_NEUTRAL_PATH = join(
        MODEL_DIR, 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
    MODEL_FEMALE_PATH = join(
        MODEL_DIR, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
    MODEL_MALE_PATH = join(MODEL_DIR,
                           'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
    #Model path of DeepCut
    model_def = '/home/zhichaozeng/DeepCut/deepcut-cnn/models/deepercut/ResNet-152.prototxt'
    model_bin = '/home/zhichaozeng/DeepCut/deepcut-cnn/models/deepercut/ResNet-152.caffemodel'


    use_interpenetration = True
    if use_interpenetration:
        # paths to the npz files storing the regressors for capsules
        SPH_REGS_NEUTRAL_PATH = join(MODEL_DIR,
                                     'regressors_locked_normalized_hybrid.npz')
        SPH_REGS_FEMALE_PATH = join(MODEL_DIR,
                                    'regressors_locked_normalized_female.npz')
        SPH_REGS_MALE_PATH = join(MODEL_DIR,
                                  'regressors_locked_normalized_male.npz')

    img_dir = '/home/zhichaozeng/Research/Datas/Videos/20161220/cuttedImage/2D_pose'
    data_dir = '/home/zhichaozeng/Research/Datas/Videos/20161220/cuttedImage/2D_pose'
    out_dir = '/home/zhichaozeng/Research/Datas/Videos/20161220/cuttedImage/2D_pose/result'

    #call the DeepCut to get the pose
   #pose = estimate_pose( image, model_def, model_bin, scales )

    call_fit_3d( img_dir, data_dir, out_dir )

