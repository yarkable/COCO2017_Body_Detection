import json
import numpy as np 
import os.path as osp

root = '/NAS_REMOTE/PUBLIC/data/coco2017/annotations'
json_file = osp.join(root, 'person_keypoints_train2017.json')
scale_w = 1.55
scale_h = 1.33
results = []
image_name = []
json_list = []
annto_cnt = 0
viz_mode = True

with open(json_file, 'r') as f:
    f = json.load(f)
    annotations = f['annotations']
    annotations = sorted(annotations, key=lambda x: x['image_id'])
    for item in annotations[:1000]:
        image_id = item['image_id']
        keypoints = item['keypoints']
        num_keypoints = item['num_keypoints']
        fn = '# ' + str(image_id).zfill(12) + '.jpg'

        left_shoulder = 5*3
        right_shoulder = 6*3
        left_hip = 11*3
        right_hip = 12*3
        
        if num_keypoints < 4:
            continue

        flag1 = keypoints[left_shoulder+2]
        flag2 = keypoints[right_shoulder+2]
        flag3 = keypoints[left_hip+2]
        flag4 = keypoints[right_hip+2]
        if flag1 == 0 or flag2 == 0 or flag3 == 0 or flag4 == 0:
            continue

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

        bbox_x = min(x1, x2, x3, x4)
        bbox_y = min(y1, y2, y3, y4)

        bbox_w = max(x1, x2, x3, x4) - bbox_x
        bbox_h = max(y1, y2, y3, y4) - bbox_y
        bbox_ctl_x = bbox_x + bbox_w / 2
        bbox_ctl_y = bbox_y + bbox_h / 2
        bbox_w = scale_w * bbox_w
        bbox_h = scale_h * bbox_h
        bbox_x = bbox_ctl_x - bbox_w / 2
        bbox_y = bbox_ctl_y - bbox_h / 2
        bbox_x = max(0, bbox_x)
        bbox_y = max(0, bbox_y)

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
            json_list.append(data)

        else:
            if flag1 == 1 or flag2 == 1 or flag3 == 1 or flag4 == 1:
                data = bbox_x, bbox_y, bbox_w, bbox_h, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
            else:
                data = bbox_x, bbox_y, bbox_w, bbox_h,\
                        x1, y1, 2, x2, y2, 2, x5, y5, 2, x3, y3, 2, x4, y4, 2
            results = np.array([data])
            with open('result.txt', 'a') as f:
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

        