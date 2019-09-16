import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng
from inpaint_model import InpaintCAModel

class InpaintModel:
    def __init__(self, model_path, max_size=768):
        self.max_size = max_size
        # build the graph
        self.model = InpaintCAModel()
        self.input_image = tf.placeholder(name='input_image', shape=[1, max_size, max_size*2, 3], dtype=tf.float32)
        net_output       = self.model.build_server_graph(self.input_image)
        net_output = (net_output + 1.) * 127.5
        self.output_image = tf.saturate_cast(net_output, tf.uint8)
        # restore pretrained model
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=sess_config)
        saver = tf.train.Saver([v for v in tf.global_variables()])
        saver.restore(self.sess, model_path)

    def inpaint(self, image, mask):
        assert(image.shape == mask.shape)
        h, w, _ = image.shape
        assert(h <= self.max_size and w <= self.max_size)
        grid = 8
        ih, iw = h//grid*grid, w//grid*grid

        input_im = np.zeros((self.max_size,self.max_size,3), dtype=np.uint8)
        input_mask = np.zeros((self.max_size,self.max_size,3), dtype=np.uint8)
        input_im[:ih, :iw, :] = image[:ih, :iw, :]
        input_mask[:ih, :iw, :] = mask[:ih, :iw, :]

        input_im = np.expand_dims(input_im, 0)
        input_mask = np.expand_dims(input_mask, 0)
        input_image = np.concatenate([input_im, input_mask], axis=2)

        output, = self.sess.run([self.output_image], feed_dict={self.input_image: input_image})
        image[:ih, :iw, :] = output[0][:ih, :iw, :]
        return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='', type=str,
                        help='The filename of image to be completed.')
    parser.add_argument('--mask', default='', type=str,
                        help='The filename of mask, value 255 indicates mask.')
    parser.add_argument('--output', default='output.png', type=str,
                        help='Where to write output.')
    parser.add_argument('--checkpoint_dir', default='', type=str,
                        help='The directory of tensorflow checkpoint.')


    args = parser.parse_args()

    model = InpaintModel(args.checkpoint_dir)
    print("Model initialization done")

    image = cv2.imread(args.image)
    mask = cv2.imread(args.mask)
    
    image = model.inpaint(image, mask)
    cv2.imwrite('output.png', image)
