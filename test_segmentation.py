# -*- coding: utf-8 -*-
"""test_segmentation.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1sstJhtXqi2x2fHiBP978OjokVpaISSIa
"""

import unittest
from PIL import Image
from models.segmentation_model import segment_image

class TestSegmentation(unittest.TestCase):

    def setUp(self):
        self.test_image = Image.open('image1.jpg')  image for testing

    def test_segment_image(self):
        segmented_image = segment_image(self.test_image)
        self.assertIsInstance(segmented_image, Image.Image, "The output should be a PIL Image object.")

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()