from argparse import ArgumentParser, Namespace

import cv2
import numpy as np

def main():
    parser = ArgumentParser(
        prog="feature-detector",
        description="Generates a feature histogram for the provided images."
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="The input file to generate features for."
    )

    parser.add_argument(
        "-s",
        "--down-sample",
        default="1",
        help="Fractions for downsampling.  Use a value of '1' for no downsampling.  Takes a comma separated list."
    )

    parser.add_argument(
        "-g",
        "--convert-grayscale",
        action="store_true",
        default=False,
        help="Converts the image to grayscale."
    )

    args = parser.parse_args()
    fast: cv2.FastFeatureDetector = cv2.FastFeatureDetector_create()

    for img in generate_images(args):     
        keypoints = get_features(img, fast)
        
        if img.shape == 2:
            height, width = img.shape
            feature_img = np.zeros((height, width, 3), dtype=np.uint8)

            feature_img[:, :, 0] = img
            feature_img[:, :, 1] = img
            feature_img[:, :, 2] = img
        else:
            feature_img = img.copy()

        cv2.drawKeypoints(feature_img, keypoints, feature_img)
        cv2.imshow("Feature Image", feature_img)

        cv2.waitKey(0)

    cv2.destroyAllWindows()
            

def generate_images(args: Namespace):
    samples = [float(s.strip()) for s in args.down_sample.split(',')]

    img = cv2.imread(args.input)
    height, width, _ = img.shape

    if args.convert_grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for sample in samples:
        target_height = int(height * sample)
        target_width = int(width * sample)

        yield cv2.resize(img, (target_width, target_height))

def get_features(img: np.ndarray, fast: cv2.FastFeatureDetector):
    return fast.detect(img, None)

if __name__ == "__main__":
    main()