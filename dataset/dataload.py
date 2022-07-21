import copy
import cv2
import torch
import numpy as np
from PIL import Image
from util.config import config as cfg
from layers.proposal_layer import ProposalTarget
from util.misc import find_bottom, find_long_edges, split_edge_seqence, \
    norm2, vector_cos, vector_sin, split_edge_seqence_by_step, point_dist_to_line
from dataset.wave_build import *
from dataset.watershed import watershed
# from torch.multiprocessing import Pool, Process, set_start_method
# try:
#      set_start_method('spawn')
# except RuntimeError:
#     pass


class TextInstance(object):
    def __init__(self, points, orient, text):
        self.orient = orient
        self.text = text
        self.bottoms = None
        self.e1 = None
        self.e2 = None
        if self.text != "#":
            self.label = 1
        else:
            self.label = -1

        remove_points = []
        if len(points) > 4:
            # remove point if area is almost unchanged after removing it
            ori_area = cv2.contourArea(points)
            for p in range(len(points)):
                # attempt to remove p
                index = list(range(len(points)))
                index.remove(p)
                area = cv2.contourArea(points[index])
                if np.abs(ori_area - area)/ori_area < 0.0017 and len(points) - len(remove_points) > 4:
                    remove_points.append(p)
            self.points = np.array([point for i, point in enumerate(points) if i not in remove_points])
        else:
            self.points = np.array(points)

    def find_bottom_and_sideline(self):
        self.bottoms = find_bottom(self.points)  # find two bottoms of this Text
        self.e1, self.e2 = find_long_edges(self.points, self.bottoms)  # find two long edge sequence

    def disk_cover(self, n_disk=15):
        """
        cover text region with several disks
        :param n_disk: number of disks
        :return:
        """
        inner_points1 = split_edge_seqence(self.points, self.e1, n_disk)
        inner_points2 = split_edge_seqence(self.points, self.e2, n_disk)
        inner_points2 = inner_points2[::-1]  # innverse one of long edge

        center_points = (inner_points1 + inner_points2) / 2  # disk center
        radii = norm2(inner_points1 - center_points, axis=1)  # disk radius

        return inner_points1, inner_points2, center_points, radii

    def Equal_width_bbox_cover(self, step=16.0):

        inner_points1, inner_points2 = split_edge_seqence_by_step(self.points, self.e1, self.e2, step=step)
        inner_points2 = inner_points2[::-1]  # innverse one of long edge

        center_points = (inner_points1 + inner_points2) / 2  # disk center

        return inner_points1, inner_points2, center_points

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)


class TextDataset(object):

    def __init__(self, transform, is_training=False):
        super().__init__()
        self.transform = transform
        self.is_training = is_training
        self.proposal = ProposalTarget(k_at_hop1=cfg.k_at_hop1)

    @staticmethod
    def make_text_region(img, polygons):

        tr_mask = np.zeros(img.shape[:2], np.uint8)
        train_mask = np.ones(img.shape[:2], np.uint8)
		# tr_weight = np.ones(img.shape[:2], np.float)
        if polygons is None:
            return tr_mask, train_mask
		
		# region_masks = list()
        # num_positive_bboxes = 0
        for polygon in polygons:
            cv2.fillPoly(tr_mask, [polygon.points.astype(np.int32)], color=(1,))
            if polygon.text == '#':
                cv2.fillPoly(train_mask, [polygon.points.astype(np.int32)], color=(0,))
                   # else:
            #     num_positive_bboxes +=1
            #     deal_mask = np.zeros_like(tr_mask)
            #     cv2.fillPoly(deal_mask, [polygon.points.astype(np.int32)], color=(1,))
            #     region_masks.append(deal_mask)

        # if cfg.weight_method == "BBOX_BALANCED":
        #     pos_region_mask = tr_mask*train_mask
        #     num_region_pixels = np.sum(pos_region_mask)
        #     for idx in range(len(region_masks)):
        #         bbox_region_mask = region_masks[idx] * pos_region_mask
        #         num_bbox_region_pixels = np.sum(bbox_region_mask)
        #         if num_bbox_region_pixels > 0:
        #             per_bbox_region_weight = num_region_pixels * 1.0 / num_positive_bboxes
        #             per_region_pixel_weight = per_bbox_region_weight / num_bbox_region_pixels
        #             tr_weight += bbox_region_mask * per_region_pixel_weight

        return tr_mask, train_mask

    @staticmethod
    def fill_polygon(mask, pts, value):
        """
        fill polygon in the mask with value
        :param mask: input mask
        :param pts: polygon to draw
        :param value: fill value
        """
        # cv2.drawContours(mask, [polygon.astype(np.int32)], -1, value, -1)
        cv2.fillPoly(mask, [pts.astype(np.int32)], color=(value,))
        # rr, cc = drawpoly(polygon[:, 1], polygon[:, 0], shape=(mask.shape[0],mask.shape[1]))
        # mask[rr, cc] = value

    def make_text_center_line(self, sideline1, sideline2,
                              center_line, tcl_msk1, tcl_msk2,
                              radius_map, sin_map, cos_map,
                              expand=0.3, shrink=1, width=1):

        mask = np.zeros_like(tcl_msk1)
        # TODO: shrink 1/2 * radius at two line end
        p1 = np.mean(sideline1, axis=0)
        p2 = np.mean(sideline2, axis=0)
        vpp = vector_sin(p1 - p2)
        if vpp >= 0:
            top_line = sideline2
            bot_line = sideline1
        else:
            top_line = sideline1
            bot_line = sideline2

        if len(center_line) < 5:
            shrink = 0

        for i in range(shrink, len(center_line) - 1 - shrink):

            c1 = center_line[i]
            c2 = center_line[i + 1]
            top1 = top_line[i]
            top2 = top_line[i + 1]
            bottom1 = bot_line[i]
            bottom2 = bot_line[i + 1]
            top = (top1 + top2) / 2
            bottom = (bottom1 + bottom1) / 2

            sin_theta = vector_sin(top - bottom)
            cos_theta = vector_cos(top - bottom)

            p1 = c1 + (top1 - c1) * expand
            p2 = c1 + (bottom1 - c1) * expand
            p3 = c2 + (bottom2 - c2) * expand
            p4 = c2 + (top2 - c2) * expand
            ploy1 = np.stack([p1, p2, p3, p4])

            self.fill_polygon(tcl_msk1, ploy1, value=1)
            self.fill_polygon(sin_map, ploy1, value=sin_theta)
            self.fill_polygon(cos_map, ploy1, value=cos_theta)

            deal_mask = mask.copy()
            self.fill_polygon(deal_mask, ploy1, value=1)
            bbox_point_cords = np.argwhere(deal_mask == 1)
            for y, x in bbox_point_cords:
                point = np.array([x, y], dtype=np.float32)
                # top   h1
                radius_map[y, x, 0] = point_dist_to_line((top1, top2), point)  # 计算point到直线的距离
                # down  h2
                radius_map[y, x, 1] = point_dist_to_line((bottom1, bottom2), point)

            pp1 = c1 + (top1 - c1) * width/norm2(top1 - c1)
            pp2 = c1 + (bottom1 - c1) * width/norm2(bottom1 - c1)
            pp3 = c2 + (bottom2 - c2) * width/norm2(top1 - c1)
            pp4 = c2 + (top2 - c2) * width/norm2(bottom2 - c2)
            poly2 = np.stack([pp1, pp2, pp3, pp4])
            self.fill_polygon(tcl_msk2, poly2, value=1)

    def get_training_data(self, image, polygons, image_id, image_path):

        H, W, _ = image.shape
        if self.transform:
            image, polygons = self.transform(image, copy.copy(polygons))

        tcl_mask = np.zeros((image.shape[0], image.shape[1], 2), np.uint8)
        radius_map = np.zeros((image.shape[0], image.shape[1], 2), np.float32)
        sin_map = np.zeros(image.shape[:2], np.float32)
        cos_map = np.zeros(image.shape[:2], np.float32)

        tcl_msk1 = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        tcl_msk2 = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        if polygons is not None:
            for i, polygon in enumerate(polygons):
                if polygon.text == '#':
                    continue
                polygon.find_bottom_and_sideline()
                sideline1, sideline2, center_points = polygon.Equal_width_bbox_cover(step=4.0)
                self.make_text_center_line(sideline1, sideline2, center_points,
                                           tcl_msk1, tcl_msk2, radius_map, sin_map, cos_map)

        tcl_mask[:, :, 0] = tcl_msk1
        tcl_mask[:, :, 1] = tcl_msk2
        tr_mask, train_mask = self.make_text_region(image, polygons)
        # clip value (0, 1)
        tcl_mask = np.clip(tcl_mask, 0, 1)
        tr_mask = np.clip(tr_mask, 0, 1)
        train_mask = np.clip(train_mask, 0, 1)

        # # to pytorch channel sequence
        image = image.transpose(2, 0, 1)

        if not self.is_training:
            points = np.zeros((cfg.max_annotation, cfg.max_points, 2))
            length = np.zeros(cfg.max_annotation, dtype=int)
            if polygons is not None:
                for i, polygon in enumerate(polygons):
                    pts = polygon.points
                    points[i, :pts.shape[0]] = polygon.points
                    length[i] = pts.shape[0]

            meta = {
                'image_id': image_id,
                'image_path': image_path,
                'annotation': points,
                'n_annotation': length,
                'Height': H,
                'Width': W
            }

            return image, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta

        rpn_roi = self.proposal(tcl_mask, radius_map, sin_map, cos_map)

        gt_roi = np.zeros((cfg.max_roi, rpn_roi.shape[1]))
        gt_roi[:rpn_roi.shape[0], :] = rpn_roi[:cfg.max_roi]

        # gt_roi = np.zeros((cfg.max_roi, 9))

        image = torch.from_numpy(image).float()
        train_mask = torch.from_numpy(train_mask).byte()
        tr_mask = torch.from_numpy(tr_mask).byte()
        tcl_mask = torch.from_numpy(tcl_mask).byte()
        radius_map = torch.from_numpy(radius_map).float()
        sin_map = torch.from_numpy(sin_map).float()
        cos_map = torch.from_numpy(cos_map).float()
        gt_roi = torch.from_numpy(gt_roi).float()

        return image, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, gt_roi

    def get_test_data(self, image, image_id, image_path):
        H, W, _ = image.shape

        if self.transform:
            image, polygons = self.transform(image)

        # to pytorch channel sequence
        image = image.transpose(2, 0, 1)

        meta = {
            'image_id': image_id,
            'image_path': image_path,
            'Height': H,
            'Width': W
        }
        return image, meta

    def __len__(self):
        raise NotImplementedError()


class TextDataset1(object):

    def __init__(self, transform, is_training=False, target_size=640, viz=False, debug=False):
        super().__init__()
        self.transform = transform
        self.is_training = is_training
        self.proposal = ProposalTarget(k_at_hop1=cfg.k_at_hop1)
        self.target_size = target_size
        self.viz = viz
        self.debug = debug

    def pil_load_img(self, path):
        image = Image.open(path)
        image = np.array(image)
        return image

    def load_image_gt_and_confidencemask(self, index):
        '''
        根据索引值返回图像、字符框、文字行内容、confidence mask
        :param index:
        :return:
        '''
        return None, None, None, None, None

    def crop_image_by_bbox(self, image, box):
        w = (int)(np.linalg.norm(box[0] - box[1]))
        h = (int)(np.linalg.norm(box[0] - box[3]))
        width = w
        height = h
        if h > w * 1.5:
            width = h
            height = w
            M = cv2.getPerspectiveTransform(np.float32(box),
                                            np.float32(np.array([[width, 0], [width, height], [0, height], [0, 0]])))
        else:
            M = cv2.getPerspectiveTransform(np.float32(box),
                                            np.float32(np.array([[0, 0], [width, 0], [width, height], [0, height]])))

        warped = cv2.warpPerspective(image, M, (width, height))
        return warped, M

    def get_confidence(self, real_len, pursedo_len):
        if pursedo_len == 0:
            return 0.
        return (real_len - min(real_len, abs(real_len - pursedo_len))) / real_len

    def normalizeMeanVariance(self, in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
        # should be RGB order
        img = in_img.copy().astype(np.float32)

        img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
        img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
        return img

    def resizeGt(self, gtmask):
        return cv2.resize(gtmask, (self.target_size, self.target_size))

    def get_imagename(self, index):
        return None

    def parse_syn_txt(self, word_bboxes, image_id):
        polygons = []
        for i in range(len(word_bboxes)):
            polygon = TextInstance(word_bboxes[i], 'c', 'abc')
            polygons.append(polygon)

        return image_id, polygons

    def parse_txt(self, annotation_path):

        with open(annotation_path) as f:
            lines = [line.strip() for line in f.readlines()]
            image_id = lines[0]
            polygons = []
            for line in lines[1:]:
                points = [float(coordinate) for coordinate in line.split(',')]
                points = np.array(points, dtype=int).reshape(4, 2)
                polygon = TextInstance(points, 'c', 'abc')
                polygons.append(polygon)
        return image_id, polygons

    def pull_item_syn(self, index):
        # if self.get_imagename(index) == 'img_59.jpg':
        #     pass
        # else:
        #     return [], [], [], [], np.array([0])
        image, character_bboxes, word_annotation_path, words, confidence_mask, confidences, img_path = self.load_image_gt_and_confidencemask(index)
        if self.transform:
            image, character_bboxes = self.transform(image, copy.copy(character_bboxes))
        if len(confidences) == 0:
            confidences = 1.0
        else:
            confidences = np.array(confidences).mean()
        wave_region = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        wave_mask = np.ones((image.shape[0], image.shape[1]), np.float32)

        if len(character_bboxes) > 0:
            try:
                wave_mask = wave_building(wave_region, character_bboxes)
            except:
                print(img_path)

        # wave1 = cv2.cvtColor(wave_mask * 255, cv2.COLOR_GRAY2BGR)
        # cv2.imwrite('wave_test1.jpg', wave1)
        wave_scores = self.resizeGt(wave_mask)
        # wave = cv2.cvtColor(wave_scores * 255, cv2.COLOR_GRAY2BGR)
        # cv2.imwrite('wave_test.jpg', wave)
        confidence_mask = self.resizeGt(confidence_mask)
        image1 = self.pil_load_img(img_path)
        wave_scores = torch.from_numpy(wave_scores).float()
        confidence_mask = torch.from_numpy(confidence_mask).float()
        confidences = np.array(confidences, dtype='float32')
        confidences = torch.from_numpy(confidences).float()

        # img_id, polygons = self.parse_syn_txt(word_bboxex, img_id)
        img_id, polygons = self.parse_txt(word_annotation_path)
        image2, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, gt_roi = self.get_training_data(image1, polygons, image_id=img_id, image_path=img_path)

        return image2, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, gt_roi, wave_scores, confidence_mask, confidences

    def pull_item_total_text(self, index):
        image1, wave_region_scores, polygons1, confidence_mask, img_id1, img_path1 = self.load_image_gt_and_confidencemask(index)
        image2, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, gt_roi = self.get_training_data(image1, polygons1, image_id=img_id1, image_path=img_path1)

        return image2, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, gt_roi, wave_region_scores, confidence_mask

    def inference_pursedo_bboxes(self, net, image, word_bbox, word, viz=False):

        word_image, MM = self.crop_image_by_bbox(image, word_bbox)

        real_word_without_space = word.replace('\s', '')
        real_char_nums = len(real_word_without_space)
        input = word_image.copy()
        scale = 64.0 / input.shape[0]
        input = cv2.resize(input, None, fx=scale, fy=scale)
        input1 = cv2.resize(input,(64, 128))
        img_torch = torch.from_numpy(self.normalizeMeanVariance(input1, mean=(0.485, 0.456, 0.406),
                                                                   variance=(0.229, 0.224, 0.225)))
        img_torch = img_torch.permute(2, 0, 1).unsqueeze(0)
        img_torch = img_torch.type(torch.FloatTensor).cuda()
        scores = net.forward_test(img_torch)
        region_scores = scores[0, 8, :, :].cpu().data.numpy()
        region_scores = np.uint8(np.clip(region_scores, 0, 1) * 255)
        bgr_region_scores = cv2.resize(region_scores, (input.shape[1], input.shape[0]))
        bgr_region_scores = cv2.cvtColor(bgr_region_scores, cv2.COLOR_GRAY2BGR)
        pursedo_bboxes = watershed(input, bgr_region_scores, True)

        _tmp = []
        for i in range(pursedo_bboxes.shape[0]):
            if np.mean(pursedo_bboxes[i].ravel()) > 2:
                _tmp.append(pursedo_bboxes[i])
            else:
                print("filter bboxes", pursedo_bboxes[i])
        pursedo_bboxes = np.array(_tmp, np.float32)
        if pursedo_bboxes.shape[0] > 1:
            index = np.argsort(pursedo_bboxes[:, 0, 0])
            pursedo_bboxes = pursedo_bboxes[index]

        confidence = self.get_confidence(real_char_nums, len(pursedo_bboxes))

        bboxes = []
        if confidence <= 0.5:
            width = input.shape[1]
            height = input.shape[0]

            width_per_char = width / len(word)
            for i, char in enumerate(word):
                if char == ' ':
                    continue
                left = i * width_per_char
                right = (i + 1) * width_per_char
                bbox = np.array([[left, 0], [right, 0], [right, height],
                                 [left, height]])
                bboxes.append(bbox)

            bboxes = np.array(bboxes, np.float32)
            confidence = 0.5

        else:
            bboxes = pursedo_bboxes
        if False:
            _tmp_bboxes = np.int32(bboxes.copy())
            _tmp_bboxes[:, :, 0] = np.clip(_tmp_bboxes[:, :, 0], 0, input.shape[1])
            _tmp_bboxes[:, :, 1] = np.clip(_tmp_bboxes[:, :, 1], 0, input.shape[0])
            for bbox in _tmp_bboxes:
                cv2.polylines(np.uint8(input), [np.reshape(bbox, (-1, 1, 2))], True, (255, 0, 0))
            region_scores_color = cv2.applyColorMap(np.uint8(region_scores), cv2.COLORMAP_JET)
            region_scores_color = cv2.resize(region_scores_color, (input.shape[1], input.shape[0]))
            target = self.gaussianTransformer.generate_region(region_scores_color.shape, [_tmp_bboxes])
            target_color = cv2.applyColorMap(target, cv2.COLORMAP_JET)
            viz_image = np.hstack([input[:, :, ::-1], region_scores_color, target_color])
            cv2.imshow("crop_image", viz_image)
            cv2.waitKey()
        bboxes /= scale
        try:
            for j in range(len(bboxes)):
                ones = np.ones((4, 1))
                tmp = np.concatenate([bboxes[j], ones], axis=-1)
                I = np.matrix(MM).I
                ori = np.matmul(I, tmp.transpose(1, 0)).transpose(1, 0)
                bboxes[j] = ori[:, :2]
        except Exception as e:
            print(e, gt_path)

        #         for j in range(len(bboxes)):
        #             ones = np.ones((4, 1))
        #             tmp = np.concatenate([bboxes[j], ones], axis=-1)
        #             I = np.matrix(MM).I
        #             ori = np.matmul(I, tmp.transpose(1, 0)).transpose(1, 0)
        #             bboxes[j] = ori[:, :2]

        bboxes[:, :, 1] = np.clip(bboxes[:, :, 1], 0., image.shape[0] - 1)
        bboxes[:, :, 0] = np.clip(bboxes[:, :, 0], 0., image.shape[1] - 1)

        return bboxes, region_scores, confidence

    @staticmethod
    def make_pursedo_wave(region_scores, polygons, size):
        region_mask = np.zeros((region_scores.shape[0], region_scores.shape[1]), np.float32)
        for polygon in polygons:
            word_bbox = polygon.points
            cv2.fillPoly(region_mask, [np.int32(word_bbox)], (1.0))
            scores_mask = region_mask * region_scores
            if scores_mask.max() >= 0.5:
                region_mask = np.zeros((region_scores.shape[0], region_scores.shape[1]), np.float32)
                continue
            # cv2.fillPoly(region_scores, [np.int32(word_bbox)], (1.0))
            # region_mask = np.zeros((region_scores.shape[0], region_scores.shape[1]), np.float32)
            if word_bbox.shape[0] != 4 and word_bbox.shape[0] != 6:
                cv2.fillPoly(region_scores, [np.int32(word_bbox)], (0.8))
                region_mask = np.zeros((region_scores.shape[0], region_scores.shape[1]), np.float32)
            else:
                word = polygon.text
                if word_bbox.shape[0] == 4:
                    if word == '#':
                        word_len = 1
                        left_up = word_bbox[0]
                        right_up = word_bbox[1]
                        left_down = word_bbox[3]
                        right_down = word_bbox[2]
                        x, y, w, h = cv2.boundingRect((word_bbox.astype(int)))
                        if w >= h:
                            for i in range(x, x + w - 1):
                                for j in range(y, y + h - 1):
                                    point = np.array([i, j])
                                    for k in range(word_len):
                                        distance_left = point_distance_line(point, left_up, left_down)
                                        distance_right = point_distance_line(point, right_up, right_down)
                                        min_distance = min(distance_left, distance_right) + 0.001
                                        distance_all = distance_right+distance_left
                                    scores = float(1.0 - (min_distance / distance_all))  # 计算距离概率分布
                                    if scores < 0.5:
                                        scores = 0.5
                                    if is_in_poly(point, word_bbox):
                                        try:
                                            region_scores[j, i] = np.array(scores)  # 赋值
                                        except:
                                            print(j, i)
                            region_mask = np.zeros((region_scores.shape[0], region_scores.shape[1]), np.float32)
                        else:
                            for i in range(x, x + w - 1):
                                for j in range(y, y + h - 1):
                                    point = np.array([i, j])
                                    for k in range(word_len):
                                        distance_left = point_distance_line(point, left_up, right_up)
                                        distance_right = point_distance_line(point, left_down, right_down)
                                        min_distance = min(distance_left, distance_right) + 0.001
                                        distance_all = distance_right+distance_left
                                    scores = float(1.0 - (min_distance / distance_all))  # 计算距离概率分布
                                    if scores < 0.5:
                                        scores = 0.5
                                    if is_in_poly(point, word_bbox):
                                        try:
                                            region_scores[j, i] = np.array(scores)  # 赋值
                                        except:
                                            print(j, i)
                            region_mask = np.zeros((region_scores.shape[0], region_scores.shape[1]), np.float32)
                    else:
                        word_len = len(word)
                        up_point = []
                        down_point = []
                        left_up = word_bbox[0]
                        right_up = word_bbox[1]
                        left_down = word_bbox[3]
                        right_down = word_bbox[2]
                        wide_up = abs(right_up[0] - left_up[0])
                        high_up = abs(right_up[1] - left_up[1])
                        wide_down = abs(right_down[0] - left_down[0])
                        high_down = abs(right_down[1] - left_down[1])
                        gap_x_up = float(wide_up / word_len)
                        gap_y_up = float(high_up / word_len)
                        gap_x_down = float(wide_down / word_len)
                        gap_y_down = float(high_down / word_len)

                        left_point = []
                        right_point = []
                        wide_left = abs(left_down[0] - left_up[0])
                        high_left = abs(left_up[1] - left_down[1])
                        wide_right = abs(right_down[0] - right_up[0])
                        high_right = abs(right_down[1] - right_up[1])
                        gap_x_left = float(wide_left / word_len)
                        gap_y_left = float(high_left / word_len)
                        gap_x_right = float(wide_right / word_len)
                        gap_y_right = float(high_right / word_len)

                        left_point.append(left_up)
                        for i in range(1, word_len):
                            if left_up[0] > left_down[0]:
                                point_i = np.array([int(left_up[0] - i * gap_x_left), int(left_up[1] + i * gap_y_left)])
                                left_point.append(point_i)
                            else:
                                point_i = np.array([int(left_up[0] + i * gap_x_left), int(left_up[1] + i * gap_y_left)])
                                left_point.append(point_i)
                        left_point.append(left_down)

                        right_point.append(right_up)
                        for i in range(1, word_len):
                            if right_up[0] > right_down[0]:
                                point_i = np.array([int(right_up[0] - i * gap_x_right), int(right_up[1] + i * gap_y_right)])
                                right_point.append(point_i)
                            else:
                                point_i = np.array([int(right_up[0] + i * gap_x_right), int(right_up[1] + i * gap_y_right)])
                                right_point.append(point_i)
                        right_point.append(right_down)

                        up_point.append(left_up)
                        for i in range(1, word_len):
                            if left_up[1] > right_up[1]:
                                point_i = np.array([int(left_up[0] + i * gap_x_up), int(left_up[1] - i * gap_y_up)])
                                up_point.append(point_i)
                            else:
                                point_i = np.array([int(left_up[0] + i * gap_x_up), int(left_up[1] + i * gap_y_up)])
                                up_point.append(point_i)
                        up_point.append(right_up)

                        down_point.append(left_down)
                        for j in range(1, word_len):
                            if left_down[1] > right_down[1]:
                                point_j = np.array([int(left_down[0] + j * gap_x_down), int(left_down[1] - j * gap_y_down)])
                                down_point.append(point_j)
                            else:
                                point_j = np.array(
                                    [int(left_down[0] + j * gap_x_down), int(left_down[1] + j * gap_y_down)])
                                down_point.append(point_j)
                        down_point.append(right_down)

                        x, y, w, h = cv2.boundingRect((word_bbox.astype(int)))
                        if w >= h:
                            for i in range(x, x + w - 1):
                                for j in range(y, y + h - 1):
                                    point = np.array([i, j])
                                    distance_list = []
                                    for k in range(word_len):
                                        distance = point_distance_line(point, up_point[k], down_point[k])
                                        distance_list.append(distance)
                                    distance_min = min(distance_list)
                                    distance_max = max(distance_list) / (word_len - 1)
                                    scores = float(
                                        1.0 - (distance_min + 0.001) / (distance_max))  # 计算距离概率分布
                                    if scores < 0.5:
                                        scores = 0.5
                                    if is_in_poly(point, word_bbox):
                                        try:
                                            region_scores[j, i] = np.array(scores)  # 赋值
                                        except:
                                            print(j, i)
                            region_mask = np.zeros((region_scores.shape[0], region_scores.shape[1]), np.float32)
                        else:
                            for i in range(x, x + w - 1):
                                for j in range(y, y + h - 1):
                                    point = np.array([i, j])
                                    distance_list = []
                                    for k in range(word_len):
                                        distance = point_distance_line(point, left_point[k], right_point[k])
                                        distance_list.append(distance)
                                    distance_min = min(distance_list)
                                    distance_max = max(distance_list) / (word_len - 1)
                                    scores = float(
                                        1.0 - (distance_min + 0.001) / (distance_max))  # 计算距离概率分布
                                    if scores < 0.5:
                                        scores = 0.5
                                    if is_in_poly(point, word_bbox):
                                        try:
                                            region_scores[j, i] = np.array(scores)  # 赋值
                                        except:
                                            print(j, i)
                            region_mask = np.zeros((region_scores.shape[0], region_scores.shape[1]), np.float32)

                elif word_bbox.shape[0] == 6:
                    if word == '#':
                        cv2.fillPoly(region_scores, [np.int32(word_bbox)], (0.6))
                        region_mask = np.zeros((region_scores.shape[0], region_scores.shape[1]), np.float32)
                    else:
                        # 左右段字符长度
                        word_len = len(word)
                        word_len_left = int(word_len/2)
                        word_len_right = int(word_len - word_len_left)
                        # 建立两个区域间隔点
                        up_point_left = []
                        down_point_left = []
                        up_point_right = []
                        down_point_right = []
                        # 定义端点
                        left_up = word_bbox[0]
                        mid_up = word_bbox[1]
                        right_up = word_bbox[2]
                        left_down = word_bbox[5]
                        right_down = word_bbox[3]
                        mid_down = word_bbox[4]
                        # 计算左右区域x,y总间距
                        wide_up_left = abs(mid_up[0] - left_up[0])
                        wide_up_right = abs(mid_up[0] - right_up[0])
                        high_up_left = abs(mid_up[1] - left_up[1])
                        high_up_right = abs(mid_up[1] - right_up[1])
                        wide_down_left = abs(mid_down[0] - left_down[0])
                        wide_down_right = abs(mid_down[0] - right_down[0])
                        high_down_left = abs(mid_down[1] - left_down[1])
                        high_down_right = abs(mid_down[1] - right_down[1])
                        # 计算左右区域x,y间隔大小
                        gap_x_up_left = float(wide_up_left / word_len_left)
                        gap_x_up_right = float(wide_up_right / word_len_right)
                        gap_y_up_left = float(high_up_left / word_len_left)
                        gap_y_up_right = float(high_up_right / word_len_right)
                        gap_x_down_left = float(wide_down_left / word_len_left)
                        gap_x_down_right = float(wide_down_right / word_len_right)
                        gap_y_down_left = float(high_down_left / word_len_left)
                        gap_y_down_right = float(high_down_right / word_len_right)

                        up_point_left.append(left_up)
                        # 添加上间隔点
                        for i in range(1, word_len_left):
                            if left_up[1] > mid_up[1]:
                                point_i = np.array(
                                    [int(left_up[0] + i * gap_x_up_left), int(left_up[1] - i * gap_y_up_left)])
                                up_point_left.append(point_i)
                            else:
                                point_i = np.array(
                                    [int(left_up[0] + i * gap_x_up_left), int(left_up[1] + i * gap_y_up_left)])
                                up_point_left.append(point_i)
                        up_point_left.append(mid_up)

                        for i in range(1, word_len_right):
                            if mid_up[1] > right_up[1]:
                                point_i = np.array(
                                    [int(mid_up[0] + i * gap_x_up_right), int(mid_up[1] - i * gap_y_up_right)])
                                up_point_right.append(point_i)
                            else:
                                point_i = np.array(
                                    [int(mid_up[0] + i * gap_x_up_right), int(mid_up[1] + i * gap_y_up_right)])
                                up_point_right.append(point_i)
                        up_point_right.append(right_up)

                        up_point_left.extend(up_point_right)

                        # 添加下间断点
                        down_point_left.append(left_down)
                        for j in range(1, word_len_left):
                            if left_down[1] > mid_down[1]:
                                point_j = np.array([int(left_down[0] + j * gap_x_down_left),
                                                    int(left_down[1] - j * gap_y_down_left)])
                                down_point_left.append(point_j)
                            else:
                                point_j = np.array([int(left_down[0] + j * gap_x_down_left),
                                                    int(left_down[1] + j * gap_y_down_left)])
                                down_point_left.append(point_j)
                        down_point_left.append(mid_down)

                        for j in range(1, word_len_right):
                            if mid_down[1] > right_down[1]:
                                point_j = np.array([int(mid_down[0] + j * gap_x_down_right),
                                                    int(mid_down[1] - j * gap_y_down_right)])
                                down_point_right.append(point_j)
                            else:
                                point_j = np.array([int(mid_down[0] + j * gap_x_down_right),
                                                    int(mid_down[1] + j * gap_y_down_right)])
                                down_point_right.append(point_j)
                        down_point_right.append(right_down)

                        down_point_left.extend(down_point_right)

                        x, y, w, h = cv2.boundingRect((word_bbox.astype(int)))
                        if w >= h:
                            for i in range(x, x + w - 1):
                                for j in range(y, y + h - 1):
                                    point = np.array([i, j])
                                    distance_list = []
                                    for k in range(word_len):
                                        distance = point_distance_line(point, up_point_left[k], down_point_left[k])
                                        distance_list.append(distance)
                                    distance_min = min(distance_list)
                                    distance_max = max(distance_list) / (int(word_len/2))
                                    scores = float(
                                        1.0 - (distance_min + 0.001) / (distance_max))  # 计算距离概率分布
                                    if scores < 0.5:
                                        scores = 0.5
                                    if is_in_poly(point, word_bbox):
                                        try:
                                            region_scores[j, i] = np.array(scores)  # 赋值
                                        except:
                                            print(j, i)
                            region_mask = np.zeros((region_scores.shape[0], region_scores.shape[1]), np.float32)
                        else:
                            for i in range(x, x + w - 1):
                                for j in range(y, y + h - 1):
                                    point = np.array([i, j])
                                    distance_list = []
                                    for k in range(word_len):
                                        distance = point_distance_line(point, up_point_left[k], down_point_left[k])
                                        distance_list.append(distance)
                                    distance_min = min(distance_list)
                                    distance_max = max(distance_list) / (int(word_len/2))
                                    scores = float(
                                        1.0 - (distance_min + 0.001) / (distance_max))  # 计算距离概率分布
                                    if scores < 0.5:
                                        scores = 0.5
                                    if is_in_poly(point, word_bbox):
                                        try:
                                            region_scores[j, i] = np.array(scores)  # 赋值
                                        except:
                                            print(j, i)
                            region_mask = np.zeros((region_scores.shape[0], region_scores.shape[1]), np.float32)


                # if word == '#':
                #     cv2.fillPoly(region_scores, [np.int32(word_bbox)], (1.0))
                #     region_mask = np.zeros((region_scores.shape[0], region_scores.shape[1]), np.float32)
                # elif word != '':
                #     word_len = len(word)
                #     up_point = []
                #     down_point = []
                #     left_up = word_bbox[0]
                #     right_up = word_bbox[1]
                #     left_down = word_bbox[3]
                #     right_down = word_bbox[2]
                #     wide_up = abs(right_up[0] - left_up[0])
                #     high_up = abs(right_up[1] - left_up[1])
                #     wide_down = abs(right_down[0] - left_down[0])
                #     high_down = abs(right_down[1] - left_down[1])
                #     gap_x_up = float(wide_up / word_len)
                #     gap_y_up = float(high_up / word_len)
                #     gap_x_down = float(wide_down / word_len)
                #     gap_y_down = float(high_down / word_len)
                #     up_point.append(left_up)
                #     for i in range(1, word_len):
                #         point_i = np.array([int(left_up[0] + i * gap_x_up), int(min(left_up[1], right_up[1]) + i * gap_y_up)])
                #         up_point.append(point_i)
                #     up_point.append(right_up)
                #     down_point.append(left_down)
                #     for j in range(1, word_len):
                #         point_j = np.array([int(left_down[0] + j * gap_x_down), int(min(left_down[1], right_down[1]) + j * gap_y_down)])
                #         down_point.append(point_j)
                #     down_point.append(right_down)
                #
                #     x, y, w, h = cv2.boundingRect((word_bbox.astype(int)))
                #     for i in range(x, x + w - 1):
                #         for j in range(y, y + h - 1):
                #             point = np.array([i, j])
                #             distance_list = []
                #             for k in range(word_len):
                #                 distance = point_distance_line(point, up_point[k], down_point[k])
                #                 distance_list.append(distance)
                #             distance_min = min(distance_list)
                #             distance_max = max(distance_list) / (word_len - 1)
                #             scores = float(
                #                 1.0 - (distance_min + 0.001) / (distance_max))  # 计算距离概率分布
                #             if scores < 0.5:
                #                 scores = 0.5
                #             if is_in_poly(point, word_bbox):
                #                 try:
                #                     region_scores[j, i] = np.array(scores)  # 赋值
                #                 except:
                #                     print(j, i)
                #     region_mask = np.zeros((region_scores.shape[0], region_scores.shape[1]), np.float32)
                # else:
                #     cv2.fillPoly(region_scores, [np.int32(word_bbox)], (1.0))
                #     region_mask = np.zeros((region_scores.shape[0], region_scores.shape[1]), np.float32)

        # region_scores = cv2.resize(region_scores, size)
        return region_scores

    @staticmethod
    def make_wave(region, polygons):
        region_scores = np.zeros((region.shape[0], region.shape[1]), np.float32)
        for polygon in polygons:
            word_bbox = polygon.points
            word_len = polygon.text
            up_point = []
            down_point = []
            left_up = word_bbox[0]
            right_up = word_bbox[1]
            left_down = word_bbox[3]
            right_down = word_bbox[2]
            wide_up = abs(right_up[0] - left_up[0])
            high_up = abs(right_up[1] - left_up[1])
            wide_down = abs(right_down[0] - left_down[0])
            high_down = abs(right_down[1] - left_down[1])
            gap_x_up = float(wide_up / word_len)
            gap_y_up = float(high_up / word_len)
            gap_x_down = float(wide_down / word_len)
            gap_y_down = float(high_down / word_len)

            left_point = []
            right_point = []
            wide_left = abs(left_down[0] - left_up[0])
            high_left = abs(left_up[1] - left_down[1])
            wide_right = abs(right_down[0] - right_up[0])
            high_right = abs(right_down[1] - right_up[1])
            gap_x_left = float(wide_left / word_len)
            gap_y_left = float(high_left / word_len)
            gap_x_right = float(wide_right / word_len)
            gap_y_right = float(high_right / word_len)

            # left_point.append(left_up)
            # for i in range(1, word_len):
            #     if left_up[0] > left_down[0]:
            #         point_i = np.array([int(left_up[0] - i * gap_x_left), int(left_up[1] + i * gap_y_left)])
            #         left_point.append(point_i)
            #     else:
            #         point_i = np.array([int(left_up[0] + i * gap_x_left), int(left_up[1] + i * gap_y_left)])
            #         left_point.append(point_i)
            # left_point.append(left_down)
            #
            # right_point.append(right_up)
            # for i in range(1, word_len):
            #     if right_up[0] > right_down[0]:
            #         point_i = np.array([int(right_up[0] - i * gap_x_right), int(right_up[1] + i * gap_y_right)])
            #         right_point.append(point_i)
            #     else:
            #         point_i = np.array([int(right_up[0] + i * gap_x_right), int(right_up[1] + i * gap_y_right)])
            #         right_point.append(point_i)
            # right_point.append(right_down)

            up_point.append(left_up)
            for i in range(1, word_len):
                if left_up[1] > right_up[1]:
                    point_i = np.array([int(left_up[0] + i * gap_x_up), int(left_up[1] - i * gap_y_up)])
                    up_point.append(point_i)
                else:
                    point_i = np.array([int(left_up[0] + i * gap_x_up), int(left_up[1] + i * gap_y_up)])
                    up_point.append(point_i)
            up_point.append(right_up)

            down_point.append(left_down)
            for j in range(1, word_len):
                if left_down[1] > right_down[1]:
                    point_j = np.array([int(left_down[0] + j * gap_x_down), int(left_down[1] - j * gap_y_down)])
                    down_point.append(point_j)
                else:
                    point_j = np.array(
                        [int(left_down[0] + j * gap_x_down), int(left_down[1] + j * gap_y_down)])
                    down_point.append(point_j)
            down_point.append(right_down)

            x, y, w, h = cv2.boundingRect((word_bbox.astype(int)))
            for i in range(x, x + w - 1):
                for j in range(y, y + h - 1):
                    point = np.array([i, j])
                    distance_list = []
                    if is_in_poly(point, word_bbox):
                        for k in range(word_len):
                            distance = point_distance_line(point, up_point[k], down_point[k])
                            distance_list.append(distance)
                            if distance_list:
                                distance_min = min(distance_list)
                                distance_max = max(distance_list) / (word_len - 1)
                                scores = float(
                                    1.0 - (distance_min + 0.001) / (distance_max))  # 计算距离概率分布
                            else:
                                scores = 0.5
                        if scores < 0.5:
                            scores = 0.5
                            try:
                                region_scores[j, i] = np.array(scores)  # 赋值
                            except:
                                print(j, i)
            region_mask = np.zeros((region_scores.shape[0], region_scores.shape[1]), np.float32)

        return region_scores

    def get_wave_gt(self, image, polygons):

        # if self.transform:
        #     image, polygons = self.transform(image, copy.copy(polygons))
        #     image = self.normalizeMeanVariance(image, mean=cfg.means, variance=cfg.stds)
        # else:
        #     image = self.normalizeMeanVariance(image, mean=cfg.means, variance=cfg.stds)

        wave_region_scores = self.make_wave(image, polygons)
        confidence_mask = np.ones((image.shape[0], image.shape[1]), np.float32)
        # confidence_mask = self.resizeGt(confidence_mask)
        # wave_region_scores5 = self.resizeGt(wave_region_scores)
        # wave_region_scores6 = torch.from_numpy(wave_region_scores5).float()
        # confidence_mask = torch.from_numpy(confidence_mask).float()

        return wave_region_scores, confidence_mask


    def inference_pursedo_label(self, net, image, polygons, size=(640, 640)):

        input1 = cv2.resize(image, size)
        img_torch = torch.from_numpy(self.normalizeMeanVariance(input1, mean=cfg.means,
                                                                   variance=cfg.stds))
        img_torch = img_torch.permute(2, 0, 1).unsqueeze(0)
        img_torch = img_torch.type(torch.FloatTensor).cuda()
        scores = net.forward_test(img_torch)
        region_scores = scores[0, 8, :, :].cpu().data.numpy()
        region_scores = np.clip(region_scores, 0, 1)
        region_scores = cv2.resize(region_scores, (image.shape[1], image.shape[0]))
        # wave = cv2.cvtColor(region_scores * 255, cv2.COLOR_GRAY2BGR)
        # cv2.imwrite('region_scores.jpg', wave)
        region_scores = self.make_pursedo_wave(region_scores, polygons, size)

        return region_scores

    @staticmethod
    def make_text_region(img, polygons):

        tr_mask = np.zeros(img.shape[:2], np.uint8)
        train_mask = np.ones(img.shape[:2], np.uint8)
        # tr_weight = np.ones(img.shape[:2], np.float)
        if polygons is None:
            return tr_mask, train_mask

        # region_masks = list()
        # num_positive_bboxes = 0
        for polygon in polygons:
            cv2.fillPoly(tr_mask, [polygon.points.astype(np.int32)], color=(1,))
            if polygon.text == '#':
                cv2.fillPoly(train_mask, [polygon.points.astype(np.int32)], color=(0,))
                # else:
            #     num_positive_bboxes +=1
            #     deal_mask = np.zeros_like(tr_mask)
            #     cv2.fillPoly(deal_mask, [polygon.points.astype(np.int32)], color=(1,))
            #     region_masks.append(deal_mask)

        # if cfg.weight_method == "BBOX_BALANCED":
        #     pos_region_mask = tr_mask*train_mask
        #     num_region_pixels = np.sum(pos_region_mask)
        #     for idx in range(len(region_masks)):
        #         bbox_region_mask = region_masks[idx] * pos_region_mask
        #         num_bbox_region_pixels = np.sum(bbox_region_mask)
        #         if num_bbox_region_pixels > 0:
        #             per_bbox_region_weight = num_region_pixels * 1.0 / num_positive_bboxes
        #             per_region_pixel_weight = per_bbox_region_weight / num_bbox_region_pixels
        #             tr_weight += bbox_region_mask * per_region_pixel_weight

        return tr_mask, train_mask

    @staticmethod
    def fill_polygon(mask, pts, value):
        """
        fill polygon in the mask with value
        :param mask: input mask
        :param pts: polygon to draw
        :param value: fill value
        """
        # cv2.drawContours(mask, [polygon.astype(np.int32)], -1, value, -1)
        cv2.fillPoly(mask, [pts.astype(np.int32)], color=(value,))
        # rr, cc = drawpoly(polygon[:, 1], polygon[:, 0], shape=(mask.shape[0],mask.shape[1]))
        # mask[rr, cc] = value

    def make_text_center_line(self, sideline1, sideline2,
                              center_line, tcl_msk1, tcl_msk2,
                              radius_map, sin_map, cos_map,
                              expand=0.3, shrink=1, width=1):

        mask = np.zeros_like(tcl_msk1)
        # TODO: shrink 1/2 * radius at two line end
        p1 = np.mean(sideline1, axis=0)
        p2 = np.mean(sideline2, axis=0)
        vpp = vector_sin(p1 - p2)
        if vpp >= 0:
            top_line = sideline2
            bot_line = sideline1
        else:
            top_line = sideline1
            bot_line = sideline2

        if len(center_line) < 5:
            shrink = 0

        for i in range(shrink, len(center_line) - 1 - shrink):

            c1 = center_line[i]
            c2 = center_line[i + 1]
            top1 = top_line[i]
            top2 = top_line[i + 1]
            bottom1 = bot_line[i]
            bottom2 = bot_line[i + 1]
            top = (top1 + top2) / 2
            bottom = (bottom1 + bottom1) / 2

            sin_theta = vector_sin(top - bottom)
            cos_theta = vector_cos(top - bottom)

            p1 = c1 + (top1 - c1) * expand
            p2 = c1 + (bottom1 - c1) * expand
            p3 = c2 + (bottom2 - c2) * expand
            p4 = c2 + (top2 - c2) * expand
            ploy1 = np.stack([p1, p2, p3, p4])

            self.fill_polygon(tcl_msk1, ploy1, value=1)
            self.fill_polygon(sin_map, ploy1, value=sin_theta)
            self.fill_polygon(cos_map, ploy1, value=cos_theta)

            deal_mask = mask.copy()
            self.fill_polygon(deal_mask, ploy1, value=1)
            bbox_point_cords = np.argwhere(deal_mask == 1)
            for y, x in bbox_point_cords:
                point = np.array([x, y], dtype=np.float32)
                # top   h1
                radius_map[y, x, 0] = point_dist_to_line((top1, top2), point)  # 计算point到直线的距离
                # down  h2
                radius_map[y, x, 1] = point_dist_to_line((bottom1, bottom2), point)

            pp1 = c1 + (top1 - c1) * width / norm2(top1 - c1)
            pp2 = c1 + (bottom1 - c1) * width / norm2(bottom1 - c1)
            pp3 = c2 + (bottom2 - c2) * width / norm2(top1 - c1)
            pp4 = c2 + (top2 - c2) * width / norm2(bottom2 - c2)
            poly2 = np.stack([pp1, pp2, pp3, pp4])
            self.fill_polygon(tcl_msk2, poly2, value=1)

    def get_training_data(self, image, polygons, image_id, image_path):

        H, W, _ = image.shape
        if self.transform:
            image, polygons = self.transform(image, copy.copy(polygons))
            image = self.normalizeMeanVariance(image, mean=cfg.means, variance=cfg.stds)
        else:
            image = self.normalizeMeanVariance(image, mean=cfg.means, variance=cfg.stds)

        tcl_mask = np.zeros((image.shape[0], image.shape[1], 2), np.uint8)
        radius_map = np.zeros((image.shape[0], image.shape[1], 2), np.float32)
        sin_map = np.zeros(image.shape[:2], np.float32)
        cos_map = np.zeros(image.shape[:2], np.float32)

        tcl_msk1 = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        tcl_msk2 = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        if polygons is not None:
            for i, polygon in enumerate(polygons):
                if polygon.text == '#':
                    continue
                polygon.find_bottom_and_sideline()
                sideline1, sideline2, center_points = polygon.Equal_width_bbox_cover(step=4.0)
                self.make_text_center_line(sideline1, sideline2, center_points,
                                           tcl_msk1, tcl_msk2, radius_map, sin_map, cos_map)

        tcl_mask[:, :, 0] = tcl_msk1
        tcl_mask[:, :, 1] = tcl_msk2
        tr_mask, train_mask = self.make_text_region(image, polygons)

        # tr_test = cv2.cvtColor(tr_mask * 255, cv2.COLOR_GRAY2BGR)
        # cv2.imwrite('tr_test.jpg', tr_test)
        # clip value (0, 1)
        tcl_mask = np.clip(tcl_mask, 0, 1)
        tr_mask = np.clip(tr_mask, 0, 1)
        train_mask = np.clip(train_mask, 0, 1)

        # # to pytorch channel sequence
        image = self.resizeGt(image)
        train_mask = self.resizeGt(train_mask)
        tr_mask = self.resizeGt(tr_mask)
        tcl_mask = self.resizeGt(tcl_mask)
        radius_map = self.resizeGt(radius_map)
        sin_map = self.resizeGt(sin_map)
        cos_map = self.resizeGt(cos_map)

        image = image.transpose(2, 0, 1)

        if not self.is_training:
            points = np.zeros((cfg.max_annotation, cfg.max_points, 2))
            length = np.zeros(cfg.max_annotation, dtype=int)
            if polygons is not None:
                for i, polygon in enumerate(polygons):
                    pts = polygon.points
                    points[i, :pts.shape[0]] = polygon.points
                    length[i] = pts.shape[0]

            meta = {
                'image_id': image_id,
                'image_path': image_path,
                'annotation': points,
                'n_annotation': length,
                'Height': H,
                'Width': W
            }

            return image, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta

        rpn_roi = self.proposal(tcl_mask, radius_map, sin_map, cos_map)

        gt_roi = np.zeros((cfg.max_roi, rpn_roi.shape[1]))
        gt_roi[:rpn_roi.shape[0], :] = rpn_roi[:cfg.max_roi]

        # gt_roi = np.zeros((cfg.max_roi, 9))


        image = torch.from_numpy(image).float()
        train_mask = torch.from_numpy(train_mask).byte()
        tr_mask = torch.from_numpy(tr_mask).byte()
        tcl_mask = torch.from_numpy(tcl_mask).byte()
        radius_map = torch.from_numpy(radius_map).float()
        sin_map = torch.from_numpy(sin_map).float()
        cos_map = torch.from_numpy(cos_map).float()
        gt_roi = torch.from_numpy(gt_roi).float()

        return image, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, gt_roi

    def get_test_data(self, image, image_id, image_path):
        H, W, _ = image.shape

        if self.transform:
            image, polygons = self.transform(image)

        # to pytorch channel sequence
        image = image.transpose(2, 0, 1)

        meta = {
            'image_id': image_id,
            'image_path': image_path,
            'Height': H,
            'Width': W
        }
        return image, meta

    def __len__(self):
        raise NotImplementedError()
