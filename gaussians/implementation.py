"""Example implementation of the BinaryClassifier class
"""

from BinaryClassifier import BinaryClassifier

# Images within `./images` 1-7.jpg have ground truths, 8 and 9.jpg do not

TRAIN_IMAGES = [
    './images/1.jpg',
    './images/2.jpg',
    './images/3.jpg'
]

TRAIN_TRUTHS = [
    './images/1_mask.png',
    './images/2_mask.png',
    './images/3_mask.png'
]

TUNE_IMAGES = [
    './images/4.jpg',
    './images/7.jpg'
]

TUNE_TRUTHS = [
    './images/4_mask.png',
    './images/7_mask.jpg'
]

TEST_IMAGES = [
    './images/5.jpg',
    './images/6.jpg'
]

TEST_TRUTHS = [
    './images/5_mask.jpg',
    './images/6_mask.jpg'
]

# Change this to run a different pipeline
RUN_PIPELINE = 2

if RUN_PIPELINE == 1:

    # Pipeline 1
    ## This trains on the train images with 3 gaussians, a threshold of 0.5 and then
    ## tests the model on the test images.

    print("Running pipeline 1")

    APPLE_CLASSIFIER = BinaryClassifier(gabor=False, log=True)
    APPLE_CLASSIFIER.load_train_images(TRAIN_IMAGES, TRAIN_TRUTHS, display=False)
    APPLE_CLASSIFIER.train(k=3)
    APPLE_CLASSIFIER.set_threshold(0.5)
    APPLE_CLASSIFIER.test(TEST_IMAGES, truths=TEST_TRUTHS, display=True)

elif RUN_PIPELINE == 2:

    # Pipeline 2
    ## This tunes the model over a variable number of gaussians (max 5) and threshold
    ## (between 0.01 and 0.99 in steps of 0.01), finds the optimum parameters and
    ## trains the model on these. Finally, the model is tested against the test images.

    print("Running pipeline 2")

    APPLE_CLASSIFIER = BinaryClassifier(gabor=False, log=True)
    APPLE_CLASSIFIER.load_train_images(TRAIN_IMAGES, TRAIN_TRUTHS, display=False)
    APPLE_CLASSIFIER.tune(TUNE_IMAGES, TUNE_TRUTHS, max_k=5, display=False)
    APPLE_CLASSIFIER.test(TEST_IMAGES, truths=TEST_TRUTHS, display=True)
