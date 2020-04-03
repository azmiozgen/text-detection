import argparse

import matplotlib.pyplot as plt

from lib.text_detection import TextDetection
from lib.utils import plt_show
from lib.config import Config

## Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input image")
parser.add_argument("-o", "--output", type=str, help="Path to the output image")
parser.add_argument("-d", "--direction", default='both+', type=str, choices=set(("light", "dark", "both", "both+")), help="Text searching")
parser.add_argument("-t", "--tesseract", action='store_true', help="Tesseract assistance")
parser.add_argument("--details", action='store_true', help="Detailed run with intermediate steps")
parser.add_argument("-f", "--fulltesseract", action='store_true', help="Full Tesseract")
args = vars(parser.parse_args())
IMAGE_FILE = args["input"]
OUTPUT_FILE = args["output"]
DIRECTION = args["direction"]
TESS = args["tesseract"]
DETAILS = args["details"]
FULL_OCR = args["fulltesseract"]

if __name__ == "__main__":
    config = Config()
    td = TextDetection(IMAGE_FILE, config, direction=DIRECTION, use_tesseract=TESS, details=DETAILS)
    if FULL_OCR:
        bounded, res = td.full_OCR()
        plt_show((td.img, "Original"), (bounded, "Final"), (res, "Mask"))
    else:
        res = td.detect()
        plt_show((td.img, "Original"), (td.final, "Final"), (res, "Mask"))
        if OUTPUT_FILE:
            plt.imsave(OUTPUT_FILE, td.final)
            print("{} saved".format(OUTPUT_FILE))
