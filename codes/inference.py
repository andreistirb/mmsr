import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model

import numpy as np

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader specific for inference
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

model = create_model(opt)
for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    for data in test_loader:
        need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True

        # insert loop for covering the whole image

        # print(data["LQ"].shape)
        batch, channels, height, width = data["LQ"].shape
        input_size = 192

        width_length = int(width / input_size)
        height_length = int(height / input_size)
        #print(height_length, width_length)

        full_input = data["LQ"]
        final_image = np.zeros((height*16, width*16, channels))

        img_path = data['LQ_path'][0]
        final_img_name = osp.splitext(osp.basename(img_path))[0]

        overlap_offset = 10 # in pixels

        for i in range(height_length + 1):
            for j in range(width_length + 1):

                start_i = i * input_size - i * overlap_offset
                if start_i < 0:
                    start_i = 0
                stop_i = start_i + input_size
                if stop_i > height:
                    stop_i = height

                start_j = j * input_size - j * overlap_offset
                if start_j < 0:
                    start_j = 0
                stop_j = start_j + input_size

                if stop_j > width:
                    stop_j = width

                # print("Height start: {} stop: {}".format(start_i, stop_i))
                # print("Width start: {} stop: {}".format(start_j, stop_j))
                data["LQ"] = full_input[:,:,start_i:stop_i,start_j:stop_j]
                #print(data["LQ"].shape)

                model.feed_data(data, need_GT=need_GT)

                img_name = osp.splitext(osp.basename(img_path))[0] + "_{}_{}".format(i,j)
                model.test()
                visuals = model.get_current_visuals(need_GT=need_GT)
                sr_img = util.tensor2img(visuals['rlt'])  # uint8

                # TO-DO fix overwriteu, trebuie facuta o medie sau ceva
                mask_range = final_image[start_i * 16:stop_i * 16,start_j*16:stop_j*16,:]
                # print(mask_range.shape)
                mask = mask_range == 0
                # print(mask.shape)
                # print(mask)
                final_image[start_i * 16:stop_i * 16,start_j*16:stop_j*16,:] += sr_img
                final_image[start_i * 16:stop_i * 16,start_j*16:stop_j*16,:] += mask * final_image[start_i * 16:stop_i * 16,start_j*16:stop_j*16,:]
                final_image[start_i * 16:stop_i * 16,start_j*16:stop_j*16,:] /= 2
                #print(sr_img.shape)
                #print(type(sr_img))

                # save images
                save_img_path = osp.join(dataset_dir, img_name + '.png')
                # util.save_img(sr_img, save_img_path)
                #logger.info(img_name)

        # save whole image
        suffix = opt['suffix']
        if suffix:
            save_img_path = osp.join(dataset_dir, final_img_name + suffix + '.png')
        else:
            save_img_path = osp.join(dataset_dir, final_img_name + '.png')
        util.save_img(final_image, save_img_path)

        logger.info(final_img_name)

        # take central crop and save
        save_img_path = osp.join(dataset_dir, final_img_name + '.png')
        height, width, channels = final_image.shape

        top = int(height/2) - 500
        left = int(width / 2) - 500
        image = final_image[top:top+1000, left:left+1000, :]
        util.save_img(image, save_img_path)


        #exit(0)

    test_stop_time = time.time()

    total_test_time = test_stop_time - test_start_time

    logger.info("total time for running inference on images: {:f}".format(total_test_time))