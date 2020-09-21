import json
import numpy as np 
import os.path as osp
import cv2
from tqdm import tqdm

'''
用来将 COCO 的关键点标注转化成 BoundingBox 的标注
最初用的是双肩、双臀四个关键点的标注来确定一个 BoundingBox
后面改需求的时候用双肩、双膝四个关键点的标注来确定一个 BoundingBox
（因为需要包括头和肩膀）
'''


root = '/NAS_REMOTE/PUBLIC/data/coco2017/annotations'
json_file = osp.join(root, 'person_keypoints_train2017.json')
img_root = '/NAS_REMOTE/PUBLIC/data/coco2017/train2017'
scale_w = 1.39
# scale_h_up = 1.33
# scale_h_down = 0.77
scale_h = 1.33
results = []
image_name = []
json_list = []
annto_cnt = 0
viz_mode = False

def convert_use_keypoint():
    global annto_cnt
    with open(json_file, 'r') as f:
        f = json.load(f)
        annotations = f['annotations']
        annotations = sorted(annotations, key=lambda x: x['image_id'])
        # 可视化测试的时候可以用少量的标注，例如 annotations[:1000]
        for item in tqdm(annotations):
            image_id = item['image_id']
            keypoints = item['keypoints']
            num_keypoints = item['num_keypoints']
            fn = '# ' + str(image_id).zfill(12) + '.jpg'

            left_shoulder = 5*3
            right_shoulder = 6*3
            left_elbow = 7*3
            right_elbow = 8*3
            left_wrist = 9*3
            right_wrist = 10*3
            left_hip = 11*3
            right_hip = 12*3
            left_knee = 13*3
            right_knee = 14*3
            
            if num_keypoints < 4:
                continue

            flag1 = keypoints[left_shoulder+2]
            flag2 = keypoints[right_shoulder+2]
            flag3 = keypoints[left_hip+2]
            flag4 = keypoints[right_hip+2]

            if flag1 == 0 or flag2 == 0 or flag3 == 0 or flag4 == 0:
                continue

            # 这四个点一定是有的，只不过可能被遮挡
            x1 = keypoints[left_shoulder]
            y1 = keypoints[left_shoulder+1]
            x2 = keypoints[right_shoulder]
            y2 = keypoints[right_shoulder+1]
            x3 = keypoints[left_hip]
            y3 = keypoints[left_hip+1]
            x4 = keypoints[right_hip]
            y4 = keypoints[right_hip+1]
            x5 = (x1 + x2 + x3 + x4) / 4
            y5 = (y1 + y2 + y3 + y4) / 4

            x6 = keypoints[left_wrist]
            y6 = keypoints[left_wrist+1]
            x7 = keypoints[right_wrist]
            y7 = keypoints[right_wrist+1]
            x8 = keypoints[left_elbow]
            y8 = keypoints[left_elbow+1]
            x9 = keypoints[right_elbow]
            y9 = keypoints[right_elbow+1]


            img = cv2.imread(osp.join(img_root, fn[2:]))
            img_h, img_w = img.shape[0], img.shape[1]


            # BoundingBox xywh，左上角坐标以及长宽
            # 这一步要处理好，不然会有bug
            bbox_x = min(x1, x2, x3, x4)
            if x6 > 0:
                bbox_x = min(bbox_x, x6)
            if x7 > 0:
                bbox_x = min(bbox_x, x7)
            if x8 > 0:
                bbox_x = min(bbox_x, x8) 
            if x9 > 0:
                bbox_x = min(bbox_x, x9)

            bbox_y = min(y1, y2, y3, y4)
            if y6 > 0:
                bbox_y = min(bbox_y, y6)
            if y7 > 0:
                bbox_y = min(bbox_y, y7)
            if y8 > 0:
                bbox_y = min(bbox_y, y8) 
            if y9 > 0:
                bbox_y = min(bbox_y, y9)

            bbox_w = max(x1, x2, x3, x4, x6, x7, x8, x9) - bbox_x
            bbox_h = max(y1, y2, y3, y4, y6, y7, y8, y9) - bbox_y
            bbox_ctl_x = bbox_x + bbox_w / 2
            bbox_ctl_y = bbox_y + bbox_h / 2

            bbox_w = scale_w * bbox_w
            # bbox_h_up = scale_h_up * bbox_h
            # bbox_h_down = scale_h_down * bbox_h
            # bbox_h = bbox_h_up + bbox_h_down
            bbox_h = scale_h * bbox_h
            
            bbox_x = bbox_ctl_x - bbox_w / 2
            bbox_y = bbox_ctl_y - bbox_h / 2
            bbox_x = max(0, bbox_x)
            bbox_y = max(0, bbox_y)

            # 防止下边越界
            if (bbox_x + bbox_w) > img_w:
                bbox_w = img_w - bbox_x
            if (bbox_y + bbox_h) > img_h:
                bbox_h = img_h - bbox_y

            if viz_mode:
                data = {}
                data['img_id'] = str(image_id)
                data['bbox_x'] = bbox_x
                data['bbox_y'] = bbox_y
                data['bbox_w'] = bbox_w
                data['bbox_h'] = bbox_h
                data['left_shoulder'] = (x1, y1)
                data['right_shoulder'] = (x2, y2)
                data['left_hip'] = (x3, y3)
                data['right_hip'] = (x4, y4)
                data['center_point'] = (x5, y5)
                data['flag'] = flag1 and flag2 and flag3 and flag4 
                json_list.append(data)

            else:
                # 关键点被遮挡的情况
                if flag1 == 1 or flag2 == 1 or flag3 == 1 or flag4 == 1:
                    data = bbox_x, bbox_y, bbox_w, bbox_h, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
                else:
                    data = bbox_x, bbox_y, bbox_w, bbox_h,\
                            x1, y1, 2, x2, y2, 2, x5, y5, 2, x3, y3, 2, x4, y4, 2
                results = np.array([data])
                with open('results/result_0916_keypoint.txt', 'a') as f:
                    if image_id in image_name:
                        pass
                    else:
                        image_name.append(image_id)
                        f.write(fn+'\n')
                    np.savetxt(f, results, delimiter=' ')
                annto_cnt += 1
                # print('anno_cnt', annto_cnt)
                
        if viz_mode:
            with open('results/viz_data.json', 'w') as f:
                json.dump(json_list, f)


def convert_use_bbox():
    global annto_cnt
    with open(json_file, 'r') as f:
        f = json.load(f)
        annotations = f['annotations']
        images = f['images']
        annotations = sorted(annotations, key=lambda x: x['image_id'])
        # 可视化测试的时候可以用少量的标注，例如 annotations[:1000]
        for item in annotations:
            image_id = item['image_id']
            keypoints = item['keypoints']
            bbox = item['bbox']
            num_keypoints = item['num_keypoints']
            fn = '# ' + str(image_id).zfill(12) + '.jpg'

            left_shoulder = 5*3
            right_shoulder = 6*3
            left_elbow = 7*3
            right_elbow = 8*3
            left_wrist = 9*3
            right_wrist = 10*3
            left_hip = 11*3
            right_hip = 12*3
            left_knee = 13*3
            right_knee = 14*3
            
            if num_keypoints < 4:
                continue

            flag1 = keypoints[left_shoulder+2]
            flag2 = keypoints[right_shoulder+2]
            flag3 = keypoints[left_hip+2]
            flag4 = keypoints[right_hip+2]

            # if flag1 == 0 or flag2 == 0 or flag3 == 0 or flag4 == 0:
            #     continue

            x1 = keypoints[left_shoulder]
            y1 = keypoints[left_shoulder+1]
            x2 = keypoints[right_shoulder]
            y2 = keypoints[right_shoulder+1]
            x3 = keypoints[left_hip]
            y3 = keypoints[left_hip+1]
            x4 = keypoints[right_hip]
            y4 = keypoints[right_hip+1]
            x5 = (x1 + x2 + x3 + x4) / 4
            y5 = (y1 + y2 + y3 + y4) / 4

            x6 = keypoints[left_wrist]
            y6 = keypoints[left_wrist+1]
            x7 = keypoints[right_wrist]
            y7 = keypoints[right_wrist+1]
            x8 = keypoints[left_elbow]
            y8 = keypoints[left_elbow+1]
            x9 = keypoints[right_elbow]
            y9 = keypoints[right_elbow+1]


            bbox_x = bbox[0]
            bbox_y = bbox[1]
            bbox_w = bbox[2]
            bbox_h = bbox[3]

            max_y = max(y3, y4, y6, y7)
            bbox_h = (max_y - bbox_y) * 1.1 if max_y > 0 else bbox_h

            min_point_x = min(x1, x2, x3, x4)
            max_point_x = max(x1, x2, x3, x4)
            if min_point_x > 0 and min_point_x < bbox_x:
                bbox_w = bbox_w + bbox_x - min_point_x
                bbox_x = min_point_x
            if max_point_x > bbox_x + bbox_w:
                bbox_w = max_point_x - bbox_x

            img = cv2.imread(osp.join(img_root, fn[2:]))
            img_h, img_w = img.shape[0], img.shape[1]

            if (bbox_y + bbox_h) > img_h:
                bbox_h = img_h - bbox_y
            if (bbox_x + bbox_w) > img_w:
                bbox_w = img_w - bbox_x

            # # BoundingBox xywh，左上角坐标以及长宽
            # bbox_x = min(x1, x2, x3, x4, x6, x7, x8, x9)
            # if x6 == 0 or x7 == 0 or x8 == 0 or x9 == 0:
            #     bbox_x = min(x1, x2, x3, x4)
            # bbox_y = min(y1, y2, y3, y4)

            # bbox_w = max(x1, x2, x3, x4, x6, x7, x8, x9) - bbox_x
            # bbox_h = max(y1, y2, y3, y4) - bbox_y
            # bbox_ctl_x = bbox_x + bbox_w / 2
            # bbox_ctl_y = bbox_y + bbox_h / 2

            # bbox_w = scale_w * bbox_w
            # bbox_h_up = scale_h_up * bbox_h
            # bbox_h_down = scale_h_down * bbox_h
            # bbox_h = bbox_h_up + bbox_h_down
            
            # bbox_x = bbox_ctl_x - bbox_w / 2
            # bbox_y = bbox_ctl_y - bbox_h_up
            # bbox_x = max(0, bbox_x)
            # bbox_y = max(0, bbox_y)

            if viz_mode:
                data = {}
                data['img_id'] = str(image_id)
                data['bbox_x'] = bbox_x
                data['bbox_y'] = bbox_y
                data['bbox_w'] = bbox_w
                data['bbox_h'] = bbox_h
                data['left_shoulder'] = (x1, y1)
                data['right_shoulder'] = (x2, y2)
                data['left_hip'] = (x3, y3)
                data['right_hip'] = (x4, y4)
                data['center_point'] = (x5, y5)
                data['flag'] = flag1 and flag2 and flag3 and flag4 
                json_list.append(data)

            else:
                # 关键点没有标注
                if flag1 == 0 or flag2 == 0 or flag3 == 0 or flag4 == 0:
                    data = bbox_x, bbox_y, bbox_w, bbox_h, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
                else:
                    data = bbox_x, bbox_y, bbox_w, bbox_h,\
                        x1, y1, 2, x2, y2, 2, x5, y5, 2, x3, y3, 2, x4, y4, 2
                results = np.array([data])
                with open('results/result_0912.txt', 'a') as f:
                    if image_id in image_name:
                        pass
                    else:
                        image_name.append(image_id)
                        f.write(fn+'\n')
                    np.savetxt(f, results, delimiter=' ')
                annto_cnt += 1
                print('anno_cnt', annto_cnt)
                
        if viz_mode:
            with open('results/viz_data.json', 'w') as f:
                json.dump(json_list, f)


def convert_use_all_bbox():
    global annto_cnt
    with open(json_file, 'r') as f:
        f = json.load(f)
        annotations = f['annotations']
        images = f['images']
        annotations = sorted(annotations, key=lambda x: x['image_id'])
        # 可视化测试的时候可以用少量的标注，例如 annotations[:1000]
        for item in annotations[:100]:
            image_id = item['image_id']
            keypoints = item['keypoints']
            bbox = item['bbox']
            num_keypoints = item['num_keypoints']
            fn = '# ' + str(image_id).zfill(12) + '.jpg'

            left_shoulder = 5*3
            right_shoulder = 6*3
            left_elbow = 7*3
            right_elbow = 8*3
            left_wrist = 9*3
            right_wrist = 10*3
            left_hip = 11*3
            right_hip = 12*3
            left_knee = 13*3
            right_knee = 14*3
            
            if num_keypoints < 4:
                continue

            flag1 = keypoints[left_shoulder+2]
            flag2 = keypoints[right_shoulder+2]
            flag3 = keypoints[left_knee+2]
            flag4 = keypoints[right_knee+2]

            # if flag1 == 0 or flag2 == 0 or flag3 == 0 or flag4 == 0:
            #     continue

            x1 = keypoints[left_shoulder]
            y1 = keypoints[left_shoulder+1]
            x2 = keypoints[right_shoulder]
            y2 = keypoints[right_shoulder+1]
            x3 = keypoints[left_knee]
            y3 = keypoints[left_knee+1]
            x4 = keypoints[right_knee]
            y4 = keypoints[right_knee+1]

            x6 = keypoints[left_hip]
            y6 = keypoints[left_hip+1]
            x7 = keypoints[right_hip]
            y7 = keypoints[right_hip+1]

            if x6 > 0 and x7 > 0:
                x5 = (x6 + x7) / 2
                y5 = (y6 + y7) / 2
            else:
                x5 = (x1 + x2 + x3 + x4) / 4
                y5 = (y1 + y2 + y3 + y4) / 4


            bbox_x = bbox[0]
            bbox_y = bbox[1]
            bbox_w = bbox[2]
            bbox_h = bbox[3]

            if image_id == 2415:
                print(bbox, bbox_y, bbox_w, bbox_h)
            # max_y = max(y3, y4, y6, y7)
            # bbox_h = (max_y - bbox_y) * 1.1 if max_y > 0 else bbox_h

            # 如果关键点超出了 bbox 就将 bbox 扩张
            min_point_x = min(x1, x2, x3, x4)
            max_point_x = max(x1, x2, x3, x4)
            if min_point_x > 0 and min_point_x < bbox_x:
                bbox_w = bbox_w + bbox_x - min_point_x
                bbox_x = min_point_x
            if max_point_x > bbox_x + bbox_w:
                bbox_w = max_point_x - bbox_x

            min_point_y = min(y1, y2, y3, y4)
            max_point_y = max(y1, y2, y3, y4)
            if min_point_y > 0 and min_point_y < bbox_y:
                bbox_h = bbox_h + bbox_y - min_point_y
                bbox_y = min_point_y
            if max_point_y > bbox_y+ bbox_h:
                bbox_h = max_point_y - bbox_y

            # 用双肩的点将头部给划出 bbox
            bbox_rb_x = bbox_x + bbox_w
            bbox_rb_y = bbox_y + bbox_h
            
            min_y = 0
            if y1 > 0 and y2 > 0:
                min_y = min(y1, y2)
            elif y1 > 0:
                min_y = y1
            elif y2 > 0:
                min_y = y2

            length = max(min_y - bbox_y, 0)
            bbox_y += 0.8 * length
            bbox_h = bbox_rb_y - bbox_y
            
            # img = cv2.imread(osp.join(img_root, fn[2:]))
            # img_h, img_w = img.shape[0], img.shape[1]

            # if (bbox_y + bbox_h) > img_h:
            #     bbox_h = img_h - bbox_y
            # if (bbox_x + bbox_w) > img_w:
            #     bbox_w = img_w - bbox_x

            # # BoundingBox xywh，左上角坐标以及长宽
            # bbox_x = min(x1, x2, x3, x4, x6, x7, x8, x9)
            # if x6 == 0 or x7 == 0 or x8 == 0 or x9 == 0:
            #     bbox_x = min(x1, x2, x3, x4)
            # bbox_y = min(y1, y2, y3, y4)

            # bbox_w = max(x1, x2, x3, x4, x6, x7, x8, x9) - bbox_x
            # bbox_h = max(y1, y2, y3, y4) - bbox_y
            # bbox_ctl_x = bbox_x + bbox_w / 2
            # bbox_ctl_y = bbox_y + bbox_h / 2

            # bbox_w = scale_w * bbox_w
            # bbox_h_up = scale_h_up * bbox_h
            # bbox_h_down = scale_h_down * bbox_h
            # bbox_h = bbox_h_up + bbox_h_down
            
            # bbox_x = bbox_ctl_x - bbox_w / 2
            # bbox_y = bbox_ctl_y - bbox_h_up
            # bbox_x = max(0, bbox_x)
            # bbox_y = max(0, bbox_y)

            if viz_mode:
                data = {}
                data['img_id'] = str(image_id)
                data['bbox_x'] = bbox_x
                data['bbox_y'] = bbox_y
                data['bbox_w'] = bbox_w
                data['bbox_h'] = bbox_h
                data['left_shoulder'] = (x1, y1)
                data['right_shoulder'] = (x2, y2)
                data['left_hip'] = (x3, y3)
                data['right_hip'] = (x4, y4)
                data['center_point'] = (x5, y5)
                data['flag'] = flag1 and flag2 and flag3 and flag4 
                json_list.append(data)

            else:
                # 关键点没有标注
                if flag1 == 0 or flag2 == 0 or flag3 == 0 or flag4 == 0:
                    data = bbox_x, bbox_y, bbox_w, bbox_h, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
                else:
                    data = bbox_x, bbox_y, bbox_w, bbox_h,\
                        x1, y1, 2, x2, y2, 2, x5, y5, 2, x3, y3, 2, x4, y4, 2
                results = np.array([data])
                with open('results/result_0915_all_box.txt', 'a') as f:
                    if image_id in image_name:
                        pass
                    else:
                        image_name.append(image_id)
                        f.write(fn+'\n')
                    np.savetxt(f, results, delimiter=' ')
                annto_cnt += 1
                print('anno_cnt', annto_cnt)
                
        if viz_mode:
            with open('results/viz_data.json', 'w') as f:
                json.dump(json_list, f)


convert_use_keypoint()