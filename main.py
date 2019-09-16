import math
from MaskRCNN import MaskRCNN
#from Reconstructor import Reconstructor
from InpaintModel import InpaintModel
import cv2
import numpy as np

def filter_bboxes_and_mask(size, boxes, masks):
    masks = masks * 255.
    w, h = size
    mask = np.zeros((h, w, 3), dtype=np.int)

    bboxes = []
    log_areas = []
    for box in boxes:
        l, t, r, b = box
        area = (b - t)*(r - l)
        log_area = math.floor(math.log(area) / math.log(10.))
        log_areas.append(log_area)

    max_log_area = max(log_areas)
    for log_area, box, m in zip(log_areas, boxes, masks):
        if(log_area != max_log_area):
            l, t, r, b = [int(x) for x in box]
            mask[t:b, l:r, 0] = 255. #m[0, t:b, l:r]
            mask[t:b, l:r, 1] = 255. #m[0, t:b, l:r]
            mask[t:b, l:r, 2] = 255. #m[0, t:b, l:r]
            bboxes.append([t, b, l, r])

    return bboxes, mask

if __name__ == '__main__':
    mrcn = MaskRCNN('./models/config.yaml')
    #recon = Reconstructor('./models/frozen_graph_256.pb', (512,256),  refine=True)
    recon = InpaintModel('./models/inpaint-place2.ckpt', max_size=1200)

    image = cv2.imread('test2.jpg')
    h, w = image.shape[:2]

    _, _, boxes, masks = mrcn.detect(image, ['person', 'dog', 'car'])
    detect_image = image.copy()
    for box in boxes:
        print(box)
        l, t, r, b = [int(x) for x in box]
        cv2.rectangle(detect_image, (l,t), (r,b), (255,0,255), 2)
    cv2.imwrite('detection.jpg', detect_image)

    bboxes, mask = filter_bboxes_and_mask((w, h), boxes, masks)
    cv2.imwrite('mask.jpg', mask)

    image1 = recon.inpaint(image, mask)
    cv2.imwrite('result_mask.jpg', image1)

