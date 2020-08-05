import cv2
import json

img_last = -1
fn_last = ''
with open('results/viz_data.json', 'r') as f:
    data = json.load(f)
    for item in data:
        img_path = '/NAS_REMOTE/PUBLIC/data/coco2017/train2017/{}.jpg'.format(item['img_id'].zfill(12))
        if item['img_id'] == img_last:
            img_path = fn_last
        img = cv2.imread(img_path)
        cv2.rectangle(img, (int(item['bbox_x']), int(item['bbox_y'])), (int(item['bbox_x']+item['bbox_w']), int(item['bbox_y']+item['bbox_h'])), (0,255,0), 1)
        cv2.circle(img, (int(item['left_shoulder'][0]), int(item['left_shoulder'][1])), 2, (0,0,255),3)
        cv2.circle(img, (int(item['right_shoulder'][0]), int(item['right_shoulder'][1])), 2, (0,0,255),3)
        cv2.circle(img, (int(item['left_hip'][0]), int(item['left_hip'][1])), 2, (0,0,255),3)
        cv2.circle(img, (int(item['right_hip'][0]), int(item['right_hip'][1])), 2, (0,0,255),3)
        cv2.circle(img, (int(item['center_point'][0]), int(item['center_point'][1])), 2, (0,0,255),3)

        fn = 'examples/coco_viz_{}.jpg'.format(item['img_id'])
        cv2.imwrite(fn, img)
        img_last = item['img_id']
        fn_last = fn