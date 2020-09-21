import cv2
import os.path as osp
import numpy as np
from tqdm import tqdm

prefix = '/NAS_REMOTE/PUBLIC/data/coco2017/train2017'

def resize_padding(image, w, h):    
    max_wh = max(image.shape[0], image.shape[1])
    if len(image.shape) == 3:
        newImage = np.zeros((max_wh, max_wh, 3), np.uint8)
        newImage[:image.shape[0], :image.shape[1], :] = image
    else:  # for grayscale
        newImage = np.zeros((max_wh, max_wh), np.uint8)
        newImage[:image.shape[0], :image.shape[1]] = image
    newImage = cv2.resize(newImage, (w, h))
    return newImage

def rescale(img, dst=(416, 416)):
    newimage = np.zeros((416, 416, 3), np.uint8)
    # BGR
    newimage[:,:,0] = 104
    newimage[:,:,1] = 117
    newimage[:,:,2] = 123
    
    ih, iw = img.shape[0: 2]
    eh, ew = dst
    scale = min(eh / ih, ew / iw)
    nh, nw = int(ih * scale), int(iw * scale)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    newimage[:img.shape[0], :img.shape[1]] = img
    return newimage, scale

def concat4img(img1, img2, img3, img4):
    t1 = np.hstack((img1, img2))
    t2 = np.hstack((img3, img4))
    t3 = np.vstack((t1, t2))
    return t3

def viz():
    img1 = cv2.imread('/NAS_REMOTE/PUBLIC/data/coco2017/train2017/000000400456.jpg')
    img2 = cv2.imread('/NAS_REMOTE/PUBLIC/data/coco2017/train2017/000000400456.jpg')
    img3 = cv2.imread('/NAS_REMOTE/PUBLIC/data/coco2017/train2017/000000400456.jpg')
    img4 = cv2.imread('/NAS_REMOTE/PUBLIC/data/coco2017/train2017/000000400456.jpg')
    img1, scale1 = rescale(img1)
    img2, scale2 = rescale(img2)
    img3, scale3 = rescale(img3)
    img4, scale4 = rescale(img4)
    newimg = concat4img(img1, img2, img3, img4)
    cv2.imshow('s', newimg)
    cv2.waitKey(0)

def save_img_and_label(newimg, scale1, scale2, scale3, scale4, label, filename):
    prefix = '/NAS_REMOTE/rpcv/jhq/'
    fn = 'imgs_0916/' + str(filename).zfill(12) + '.jpg'
    cv2.imwrite(prefix + fn, newimg)

    for i in range(4):
            
        label[i] = np.array(label[i])
        label[i][0][0] = max(0, label[i][0][0])
        label[i][0][1] = max(0, label[i][0][1])
        
        if i == 0:
            label[i] = label[i] * ([scale1] * 4 + [scale1, scale1, 1] * 5)
            label[i][0][2] = 416 - label[i][0][0] if (label[i][0][0] + label[i][0][2]) > 416 else label[i][0][2]
            label[i][0][3] = 416 - label[i][0][1] if (label[i][0][1] + label[i][0][3]) > 416 else label[i][0][3]
        if i == 1:
            label[i] = label[i] * ([scale2] * 4 + [scale2, scale2, 1] * 5) + ([416,0,0,0] + [416, 0, 0] * 5)
            label[i][0][2] = 416 - label[i][0][0] if (label[i][0][0] + label[i][0][2]) > 416 * 2 else label[i][0][2]
            label[i][0][3] = 416 - label[i][0][1] if (label[i][0][1] + label[i][0][3]) > 416 else label[i][0][3]
        if i == 2:            
            label[i] = label[i] * ([scale3] * 4 + [scale3, scale3, 1] * 5) + ([0,416,0,0] + [0, 416, 0] * 5)
            label[i][0][2] = 416 - label[i][0][0] if (label[i][0][0] + label[i][0][2]) > 416 else label[i][0][2]
            label[i][0][3] = 416 - label[i][0][1] if (label[i][0][1] + label[i][0][3]) > 416 * 2 else label[i][0][3]
        if i == 3:
            label[i] = label[i] * ([scale4] * 4 + [scale4, scale4, 1] * 5) + ([416,416,0,0] + [416, 416, 0] * 5)
            label[i][0][2] = 416 - label[i][0][0] if (label[i][0][0] + label[i][0][2]) > 416 * 2 else label[i][0][2]
            label[i][0][3] = 416 - label[i][0][1] if (label[i][0][1] + label[i][0][3]) > 416 * 2 else label[i][0][3]
    
    with open('/NAS_REMOTE/rpcv/jhq/label_0916_keypoint.txt', 'a') as f:

        f.write('# ' + fn + '\n')
        np.savetxt(f, label[0])
        np.savetxt(f, label[1])
        np.savetxt(f, label[2])
        np.savetxt(f, label[3])
        print('[INFO] ' + fn + ' saved!')

def main():

    name = [0,0,0,0]
    label = [[], [], [], []]
    index = -1
    count = 0
    newimg = np.zeros((416, 416, 3))
    scale1, scale2, scale3, scale4 = 0, 0, 0, 0

    with open('results/result_0916_keypoint.txt', 'r') as f:
        for line in tqdm(f):
            line = line.strip(' ').strip('\n')

            if line.startswith('#'):
                count += 1
                index += 1
                index = index % 4
                if index % 4 == 0 and count != 1:
                    save_img_and_label(newimg, scale1, scale2, scale3, scale4, label, count)

                fn = line.strip('#').strip(' ')

                label[index] = []   # clear
                name[index] = fn


                    
            else:
                
                line = list(map(float, line.split(' ')))
                label[index].append(line)

                if index % 4  == 3:# 末尾了
                    img1 = cv2.imread(osp.join(prefix, name[0]))
                    img2 = cv2.imread(osp.join(prefix, name[1]))
                    img3 = cv2.imread(osp.join(prefix, name[2]))
                    img4 = cv2.imread(osp.join(prefix, name[3]))
                    img1, scale1 = rescale(img1)
                    img2, scale2 = rescale(img2)
                    img3, scale3 = rescale(img3)
                    img4, scale4 = rescale(img4)
                    newimg = concat4img(img1, img2, img3, img4)

if __name__ == "__main__":
    main()

