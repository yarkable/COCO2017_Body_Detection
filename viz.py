import cv2
import json

def viz_single():
    '''
    可视化单张图片
    依靠双肩、双臀四个关键点的标注来确定一个 BoundingBox 
    然后将 BoundingBox 给可视化
    '''
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
            if item['flag']:
                cv2.circle(img, (int(item['left_shoulder'][0]), int(item['left_shoulder'][1])), 2, (0,0,255),3)
                cv2.circle(img, (int(item['right_shoulder'][0]), int(item['right_shoulder'][1])), 2, (0,0,255),3)
                cv2.circle(img, (int(item['left_hip'][0]), int(item['left_hip'][1])), 2, (0,0,255),3)
                cv2.circle(img, (int(item['right_hip'][0]), int(item['right_hip'][1])), 2, (0,0,255),3)
                cv2.circle(img, (int(item['center_point'][0]), int(item['center_point'][1])), 2, (0,0,255),3)

            fn = 'examples/coco_viz_{}.jpg'.format(item['img_id'])
            cv2.imwrite(fn, img)
            img_last = item['img_id']
            fn_last = fn


def viz_four():
    '''
    四张图片拼成一张之后的可视化
    依靠双肩、双臀四个关键点的标注来确定一个 BoundingBox 
    然后将 BoundingBox 给可视化
    '''
    this_file = []
    next_file = []
    prefix = '/NAS_REMOTE/rpcv/jhq/'
    with open('/NAS_REMOTE/rpcv/jhq/label_0809.txt', 'r') as f:

        while 1:
            this_file.append(f.readline())
            if len(this_file) != 1 and this_file[-1][0] == '#':
                next_file.append(this_file.pop(-1))

                fn = this_file.pop(0)[2:].strip('\n')
                img = cv2.imread(prefix+fn)
                
                for label in this_file:
                    label = label.strip('\n')
                    label = list(map(int, list(map(float, label.split(' ')))))
                    bbox_x = label[0]
                    bbox_y = label[1]
                    bbox_w = label[2]
                    bbox_h = label[3]
                    left_shoulder = (label[4], label[5])
                    right_shoulder = (label[7], label[8])
                    left_hip = (label[10], label[11])
                    right_hip = (label[13], label[14])
                    center_point = (label[16], label[17])

                    # print(left_hip, left_shoulder, center_point)

                    cv2.rectangle(img, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (0,255,0), 1)
                    if label[18] != -1:
                        cv2.circle(img, left_shoulder, 2, (0,0,255),3)
                        cv2.circle(img, right_shoulder, 2, (0,0,255),3)
                        cv2.circle(img, left_hip, 2, (0,0,255),3)
                        cv2.circle(img, right_hip, 2, (0,0,255),3)
                        cv2.circle(img, center_point, 2, (0,0,255),3)

                # 注意 fn 可能会有 / ，要进行切片切掉
                filename = 'examples/coco_viz_{}'.format(fn[10:])
                cv2.imwrite(filename, img)

                this_file = next_file
                next_file = []
            else:
                continue



def viz_four_by_result():
    '''
    四张图片拼成一张之后的可视化
    用本地的 result 来可视化，用于查 bug
    '''
    this_file = []
    next_file = []
    prefix = '/NAS_REMOTE/rpcv/jhq/'
    with open('/NAS_REMOTE/rpcv/jhq/label_0916_keypoint.txt', 'r') as f:

        while 1:
            this_file.append(f.readline())
            if len(this_file) != 1 and this_file[-1][0] == '#':
                next_file.append(this_file.pop(-1))

                fn = this_file.pop(0)[2:].strip('\n')
                img = cv2.imread(prefix+fn)
                
                for label in this_file:
                    label = label.strip('\n')
                    label = list(map(int, list(map(float, label.split(' ')))))
                    bbox_x = label[0]
                    bbox_y = label[1]
                    bbox_w = label[2]
                    bbox_h = label[3]
                    left_shoulder = (label[4], label[5])
                    right_shoulder = (label[7], label[8])
                    left_hip = (label[10], label[11])
                    right_hip = (label[13], label[14])
                    center_point = (label[16], label[17])

                    # print(left_hip, left_shoulder, center_point)

                    cv2.rectangle(img, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (0,255,0), 1)
                    if label[18] != -1:
                        cv2.circle(img, left_shoulder, 2, (0,0,255),3)
                        cv2.circle(img, right_shoulder, 2, (0,0,255),3)
                        cv2.circle(img, left_hip, 2, (0,0,255),3)
                        cv2.circle(img, right_hip, 2, (0,0,255),3)
                        cv2.circle(img, center_point, 2, (0,0,255),3)

                # 注意 fn 可能会有 / ，要进行切片切掉
                filename = 'examples/coco_viz_{}'.format(fn)
                cv2.imwrite(filename, img)

                this_file = next_file
                next_file = []
            else:
                continue

viz_four()