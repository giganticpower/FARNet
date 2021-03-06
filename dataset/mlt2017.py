import os
import numpy as np
from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset1, TextInstance
from util.io import read_lines
from util.misc import norm2
from util import strs
import cv2
import copy


class Mlt2017Text1(TextDataset1):

    def __init__(self, data_root, is_training=True, transform=None, target_size=640, viz=False, debug=False):
        super().__init__(transform, is_training, target_size, viz, debug)

        self.data_root = os.path.join(data_root, "train_images" if is_training else "test_images")
        self.is_training = is_training
        self.wave_gt_root = os.path.join(data_root, 'wave_mlt_gt')

        if is_training:
            with open(os.path.join(self.data_root, 'train_list')) as f:
                self.img_train_list = [line.strip() for line in f.readlines()]

            self.img_list = self.img_train_list
        else:
            with open(os.path.join(self.data_root, 'test_list')) as f:
                self.img_list = [line.strip() for line in f.readlines()]


    @staticmethod
    def parse_txt(gt_path):
        """
        .mat file parser
        :param gt_path: (str), mat file path
        :return: (list), TextInstance
        """
        lines = read_lines(gt_path + ".txt")
        polygons = []
        for line in lines:
            line = strs.remove_all(line.strip('\ufeff'), '\xef\xbb\xbf')
            gt = line.split(',')
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(int, gt[:8]))
            xx = [x1, x2, x3, x4]
            yy = [y1, y2, y3, y4]

            # if gt[-1].strip() == "###":
            #     label = gt[-1].strip().replace("###", "#")
            # else:
            #     label = "GG"
            pts = np.stack([xx, yy]).T.astype(np.int32)

            d1 = norm2(pts[0] - pts[1])
            d2 = norm2(pts[1] - pts[2])
            d3 = norm2(pts[2] - pts[3])
            d4 = norm2(pts[3] - pts[0])
            if min([d1, d2, d3, d4]) < 2:
               continue
            polygons.append(TextInstance(pts, 'c', len(gt[-1].strip())))

        return polygons

    def __getitem__(self, item):

        image_id = self.img_list[item]

        if self.is_training:
            # Read annotation
            # annotation_id = "{}/gt_{}".format("/".join(image_id.split("/")[0:-1]),
            #                                   image_id.split("/")[-1].replace(".jpg", ''))
            annotation_id = image_id.split('.')[0]
            annotation_path = os.path.join(self.data_root, annotation_id)
            polygons = self.parse_txt(annotation_path)
        else:
            annotation_id = image_id.split('.')[0]
            polygons = None

        # Read image data
        image_path = os.path.join(self.data_root, image_id)
        image = pil_load_img(image_path)
        try:
            h, w, c = image.shape
            assert (c == 3)
        except:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image)
            print("MMMMMMMMMMMMMMMMMM")

        if self.transform:
            image, polygons = self.transform(image, copy.copy(polygons))

        wave_region_scores, confidence_mask = self.get_wave_gt(image, polygons)
        np.save('/data/chh/DRRG-master/data/MLT2017/wave_mlt_gt/'+ annotation_id + '_wave.npy', wave_region_scores)
        # image, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, gt_roi = self.get_training_data(image, polygons, image_id=image_id, image_path=image_path)

        # return image, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, gt_roi, wave_region_scores, confidence_mask
        return wave_region_scores

    def __len__(self):
        return len(self.img_list)


if __name__ == '__main__':
    import os
    from util.augmentation import BaseTransform, Augmentation1, Augmentation
    from util.misc import fill_hole, regularize_sin_cos
    from nmslib import lanms
    from util import bbox_transfor_inv, minConnectPath
    from util import canvas as cav
    import time
    from network.textnet import TextNet
    import torch
    import torch.backends.cudnn as cudnn
    # import cv2

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(
        size=640, mean=means, std=stds
    )

    trainset = Mlt2017Text1(
        data_root='../data/MLT2017',
        is_training=True,
        transform=transform
    )
    # img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta = trainset[944]
    for idx in range(0, len(trainset)):
        t0 = time.time()
        wave = trainset[idx]
        print(idx, wave.shape)
        # img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, gt_roi, wave, mask = trainset[idx]
        # img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, gt_roi, wave, mask\
        #     = map(lambda x: x.cpu().numpy(), (img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, gt_roi, wave, mask))
        #
        # img = img.transpose(1, 2, 0)
        # img = ((img * stds + means) * 255).astype(np.uint8)
        # print(idx, img.shape)
        # top_map = radius_map[:, :, 0]
        # bot_map = radius_map[:, :, 1]
        #
        # print(radius_map.shape)
        #
        # sin_map, cos_map = regularize_sin_cos(sin_map, cos_map)
        # ret, labels = cv2.connectedComponents(tcl_mask[:, :, 0].astype(np.uint8), connectivity=8)
        # cv2.imshow("labels0", cav.heatmap(np.array(labels * 255 / np.max(labels), dtype=np.uint8)))
        # print(np.sum(tcl_mask[:, :, 1]))
        #
        # t0 = time.time()
        # for bbox_idx in range(1, ret):
        #     bbox_mask = labels == bbox_idx
        #     text_map = tcl_mask[:, :, 0] * bbox_mask
        #
        #     boxes = bbox_transfor_inv(radius_map, sin_map, cos_map, text_map, wclip=(2, 8))
        #     # nms
        #     boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), 0.25)
        #     boxes = boxes[:, :8].reshape((-1, 4, 2)).astype(np.int32)
        #     if boxes.shape[0] > 1:
        #         center = np.mean(boxes, axis=1).astype(np.int32).tolist()
        #         paths, routes_path = minConnectPath(center)
        #         boxes = boxes[routes_path]
        #         top = np.mean(boxes[:, 0:2, :], axis=1).astype(np.int32).tolist()
        #         bot = np.mean(boxes[:, 2:4, :], axis=1).astype(np.int32).tolist()
        #
        #         boundary_point = top + bot[::-1]
        #         # for index in routes:
        #
        #         for ip, pp in enumerate(top):
        #             if ip == 0:
        #                 color = (0, 255, 255)
        #             elif ip == len(top) - 1:
        #                 color = (255, 255, 0)
        #             else:
        #                 color = (0, 0, 255)
        #             cv2.circle(img, (int(pp[0]), int(pp[1])), 2, color, -1)
        #         for ip, pp in enumerate(bot):
        #             if ip == 0:
        #                 color = (0, 255, 255)
        #             elif ip == len(top) - 1:
        #                 color = (255, 255, 0)
        #             else:
        #                 color = (0, 255, 0)
        #             cv2.circle(img, (int(pp[0]), int(pp[1])), 2, color, -1)
        #         cv2.drawContours(img, [np.array(boundary_point)], -1, (0, 255, 255), 1)
        # # print("nms time: {}".format(time.time() - t0))
        # # # cv2.imshow("", img)
        # # # cv2.waitKey(0)

        # # print(meta["image_id"])
        # cv2.imshow('imgs', img)
        # cv2.imshow("", cav.heatmap(np.array(labels * 255 / np.max(labels), dtype=np.uint8)))
        # cv2.imshow("tr_mask", cav.heatmap(np.array(tr_mask * 255 / np.max(tr_mask), dtype=np.uint8)))
        # cv2.imshow("tcl_mask",
        #            cav.heatmap(np.array(tcl_mask[:, :, 1] * 255 / np.max(tcl_mask[:, :, 1]), dtype=np.uint8)))
        # # cv2.imshow("top_map", cav.heatmap(np.array(top_map * 255 / np.max(top_map), dtype=np.uint8)))
        # # cv2.imshow("bot_map", cav.heatmap(np.array(bot_map * 255 / np.max(bot_map), dtype=np.uint8)))
        # cv2.waitKey(0)
        # wave = cv2.cvtColor(wave * 255, cv2.COLOR_GRAY2BGR)
        # tr = cv2.cvtColor(tr_mask * 255, cv2.COLOR_GRAY2BGR)
        # tcl = cv2.cvtColor(tcl_mask[:, :, 0] * 255, cv2.COLOR_GRAY2BGR)
        # im_vis = np.concatenate([wave, img, tr, tcl], axis=1)
        # cv2.imwrite('wave_mlt_{}.jpg'.format(idx), im_vis)

