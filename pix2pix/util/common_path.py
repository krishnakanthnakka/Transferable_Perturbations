import os
'''Things that we can change'''
###################################################
# siam_model_ = 'siamrpn_r50_l234_dwxcorr'
siam_model_ = 'siamrpn_r50_l234_dwxcorr_otb'
###################################################
dataset_name_ = 'OTB100'
# dataset_name_ = 'VOT2018'
# dataset_name_ = 'LaSOT'
##################################################
# video_name_ = 'CarScale' # worser(inaccurate scale estimation)
# video_name_ = 'Bolt' # fail earlier(distractor)
# video_name_ = 'Doll' # unstable
# video_name_ = 'ants1'
# video_name_ = 'airplane-1'
video_name_ = ''
#########################################################################################
'''change to yours'''
project_path_ = '//cvlabdata1/home/krishna/CSA'
dataset_root_ = '//cvlabdata1/home/krishna/CSA/'  # make links for used datasets
train_set_path_ = '/cvlabdata1/home/krishna/CSA'
