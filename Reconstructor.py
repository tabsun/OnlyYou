import argparse
import random
import string
import cv2
import numpy as np
import tensorflow as tf

def normalize_image(image):
    """
    normalize_image: Give a normalized image matrix which can be used with implot, etc.
    Maps to [0, 1]
    """
    im = image.astype(float)
    if np.max(im) != np.min(im):
        return (im - np.min(im))/(np.max(im) - np.min(im))
    else:
        return np.zeros(im.shape)

def threshold_image(image, threshold=0.5):
    '''
    Threshold the image to make all its elements greater than threshold*MAX = 1
    '''
    m, M = np.min(image), np.max(image)
    im = normalize_image(image)
    im[im >= threshold] = 1
    im[im < 1] = 0
    return im

class Reconstructor:
    # member variables
    graph = None
    persistent_sess = None
    input_node = None
    output_node = None
    input_size = (1024, 512)

    def __init__(self, graph_path, input_size, refine=False):
        if(refine):
            # model_path is the model dir which contains *.checkpoint *.meta ...
            self.input_size = input_size
            tag_str = ''.join(random.choice(string.ascii_uppercase+string.digits) for _ in range(8))
            with tf.gfile.GFile(graph_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            self.graph = tf.get_default_graph()
            tf.import_graph_def(graph_def, name='CA_%s'%tag_str)
            self.persistent_sess = tf.Session(graph=self.graph)
            for op in self.graph.get_operations():
                print(op.name)

            self.input_node  = self.graph.get_tensor_by_name('CA_%s/input_image:0'%tag_str)
            self.output_node = self.graph.get_tensor_by_name('CA_%s/ContextAttention/reconstruction:0'%tag_str)
            
            # warm up
            self.persistent_sess.run(
                self.output_node, 
                feed_dict={
                    self.input_node:[np.ndarray(shape=(input_size[1], input_size[0], 3), dtype=np.float32)]
                }
            )
            print("Loading model CA_%s done" % tag_str)

    def __del__(self):
        if(self.persistent_sess):
            self.persistent_sess.close()

    def preprocess(self, image, mask_image):
        assert image.shape == mask_image.shape
        h, w, _ = image.shape
        std_size = max(h, w)
        
        std_image = np.zeros((std_size, std_size, 3), dtype=np.float32)
        std_mask  = std_image.copy()
        offset_x, offset_y = (std_size-w)//2, (std_size-h)//2
        std_image[offset_y:offset_y+h, offset_x:offset_x+w, :] = image
        std_mask[offset_y:offset_y+h, offset_x:offset_x+w, :] = mask_image
        
        iw, ih = int(self.input_size[0]/2), int(self.input_size[1])
        image = cv2.resize(     std_image, (iw, ih))
        mask_image = cv2.resize(std_mask,  (iw, ih))
        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask_image, 0)
        input_image = np.concatenate([image, mask], axis=2)
        
        return input_image

    def recovery(self, image, rects, refine=False):
        if(refine):
            return self.recovery_refine(image, rects)
        else:
            return self.recovery_coarse(image, rects)

    def recovery_coarse(self, image, rects):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
 
        for rect in rects:
            t, b, l, r = rect
            local_mask = threshold_image(normalize_image(np.max(image[t:b,l:r,:], axis=2)), 0.5).astype(np.uint8)
            # merge the mask
            mask[t:b, l:r] = np.maximum(mask[t:b,l:r], local_mask)

        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        recon = image.copy()
        # **An Image Inpainting Technique Based on the Fast Marching Method** - TELEA
        # **Navier-Stokes, Fluid Dynamics, and Image and Video Inpainting**   -NS
        recon = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
        return recon

    def recovery_refine(self, image, rects):
        if len(rects) == 0:
            return image
        h, w, _ = image.shape
        rects = np.array(rects)
        assert(rects.shape[1] == 4)
        rects[:,0][rects[:,0] < 0] = 0
        rects[:,1][rects[:,1] > h-1] = h-1
        rects[:,2][rects[:,2] < 0] = 0
        rects[:,3][rects[:,3] > w-1] = w-1
        
        top, _, left, _ = np.min(rects, axis=0)
        _, bottom, _, right = np.max(rects, axis=0)
        std_size = max(h,w)
        box_size = max(right-left, bottom-top)
        block_size = self.input_size[1]

        # make square-size blank image
        offset_x, offset_y = (std_size-w)//2, (std_size-h)//2
        blank_image = np.zeros((std_size,std_size,3), dtype=np.uint8)
        blank_image[offset_y:offset_y+h, offset_x:offset_x+w,:] = image
        if std_size < block_size:
            scale = block_size*1.0 / std_size
        else:
            if box_size > block_size:
                scale = block_size*1.0 / min(std_size, (1.5*box_size))
            else:
                scale = 1.0 
        input_image = cv2.resize(blank_image, (int(std_size*scale), int(std_size*scale)))
        input_rects = []
        for rect in rects:
            t, b, l, r = rect
            t = int((t+offset_y)*scale+0.5)
            b = int((b+offset_y)*scale+0.5)
            l = int((l+offset_x)*scale+0.5)
            r = int((r+offset_x)*scale+0.5)
            input_rects.append([t,b,l,r])

        recon_input_image = self._recovery_refine(input_image, input_rects)
        recon_blank_image = cv2.resize(recon_input_image, (std_size,std_size))
        recon_image = recon_blank_image[offset_y:offset_y+h, offset_x:offset_x+w,:]
        for rect in rects:
            t, b, l, r = rect
            image[t:b, l:r, :] = recon_image[t:b, l:r, :]
        return image

    def _recovery_refine(self, image, rects):
        rects = np.array(rects)
        h, w, _ = image.shape
        top, _, left, _ = np.min(rects, axis=0)
        _, bottom, _, right = np.max(rects, axis=0)
        block_size = self.input_size[1]
        assert(block_size >= right-left and block_size >= bottom-top)
        assert(h >= block_size and w >= block_size)

        range_l = max(0, right - block_size)
        range_t = max(0, bottom - block_size)
        range_r = min(w-block_size, left)
        range_b = min(h-block_size, top)
        best_x, best_y = left - (block_size-right+left)//2, top - (block_size-bottom+top)//2
        sx = min(max(range_l, best_x), range_r)
        sy = min(max(range_t, best_y), range_b)
        rect = [sx, sy, sx+block_size, sy+block_size]

        roi_image = image[sy:sy+block_size, sx:sx+block_size, :]
        mask = np.zeros(roi_image.shape, dtype=np.uint8)
        for rec_rect in rects:
            t, b, l, r = rec_rect
            mask[t-sy:b-sy, l-sx:r-sx, :] = 255
        roi_rec_image = self.inpaint(roi_image, mask)
        
        for rec_rect in rects:
            t, b, l, r = rec_rect
            image[t:b, l:r, :] = roi_rec_image[t-sy:b-sy, l-sx:r-sx,:]
        return image
        
        
    def inpaint(self, image, mask):
        input_image = self.preprocess(image, mask)
        output = self.persistent_sess.run(self.output_node, feed_dict={self.input_node:input_image})
        
        # postprocess
        h, w, _ = image.shape
        std_size = max(w, h)
        recon_image = cv2.resize(output[0], (std_size,std_size))
        offset_x, offset_y = (std_size-w)//2, (std_size-h)//2
        recon_image = recon_image[offset_y:offset_y+h, offset_x:offset_x+w, :]
        recon_image = cv2.cvtColor(recon_image, cv2.COLOR_BGR2RGB)
        return recon_image

if __name__ == '__main__':
    recon = Reconstructor('./models/frozen_graph_256.pb', (512,256),  refine=True)

    image = cv2.imread('test.png')
    h, w = image.shape[:2]
    mask = np.zeros((h, w, 3), dtype=np.int)
    mask[10:30, 10:30, :] = 255
    image = recon.inpaint(image, mask)
    cv2.imwrite('inpaint.png', image)
