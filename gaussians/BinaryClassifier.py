"""
Classes:
    BinaryClassifier - Classifies images pixel-by-pixel using a mixture of gaussians model.
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
from Gabor import Gabor

class BinaryClassifier:

    """
    Adds the ability to train a binary classifier using a mixture of gaussians
    model over a given set of images, tune hyperparameters on more images, and finally test
    it on an unseen data set.

    Public methods:
        load_train_images - loads in images to train the classifier on
        set_threshold - change the discrimination threshold (> threshold -> classified as class)
        train - trains the classifier on the images given by `load_train_images`
        tune - tunes the model hyperparameters on a given development set of data
        test - tests the model on unseen data, giving quantitative results and graphs
        if truth masks exist

    Public instance variables:
        None

    """

    def __init__(self, gabor=False, log=True):

        # For EM algorithm, once likelihood update changes by less than this amount, iteration halts
        self._eps = 10

        # Whether or not to print logging data to stdout
        self._logging = log

        # threshold for discriminating between positive and negative classification (used by ROC)
        # can be updated via the `set_threshold` method
        # 0.5 says that if the probability of an individual pixel being 'class' is > 0.5, then
        # classify it as 'class', else, 'not class'
        self._threshold = 0.5

        # Whether or not to apply a gabor filter to the images to improve model by accounting
        # for the textures in the image rather than colour alone
        self._gabor = gabor
        if self._gabor:
            self._gabor = Gabor()

        # used to avoid division by zero by adding a small epsilon
        self._flt_min = sys.float_info.min

        # These will hold all images and their ground truth masks
        self._images = {}
        self._truths = {}

        # will contain RGB data for each pixel which corresponds to the class and those which don't
        self._data_true = []
        self._data_false = []

        # Priors on pixel data [True, False]
        self._prior = [0.5, 0.5]

        # Data related to the mixture of gaussians fitted model (see `train`)
        self._k = 1
        self._model_true = None
        self._model_false = None

    def _log(self, *args):
        """Print logging information to stdout
        """
        if self._logging:
            print(*args)

    def load_train_images(self, images, truths, display=True):
        """
        Reads in training images and their ground truths and calculates a prior on the
        classification of pixels.

        Parameters:
            images - list of images to train on (file path)
            truths - list of ground truth images corresponding to  `images` list
            (black=False, white=True)
            display - whether or not (`True`/`False`) to display images along with their
            ground truth masks (for debugging)

        Side-effects:
            Populates `_data_true`, `_data_false`, `_prior`
            Displays given images and truth masks (if `display` is `True`)

        Returns:
            None
        """

        self._log("Loading in training images")

        # Loop through each image and its corresponding truth mask, populating two
        # lists of pixel data which differentiates True and False matches with the class
        for image, truth in zip(images, truths):
            image = plt.imread(image) # read in pixels of image

            # Either apply the gabor filter or squash pixels between [0,1] (not necessary for gabor)
            if self._gabor:
                image = self._gabor.process(image)
            else:
                image = image / 255

            # Ground truth is a 1-channel representation of pixels which match and don't match class
            truth = (plt.imread(truth)[:, :, 2] != 0).astype(int)

            # Populate list of pixels corresponding to the class (True) and others (False)
            for row in range(truth.shape[0]):
                for col in range(truth.shape[1]):
                    if truth[row, col] == 1.0:
                        self._data_true.append(image[row, col])
                    else:
                        self._data_false.append(image[row, col])

            # Display each image along with the corresponding ground truth
            if display:
                _, (ax1, ax2) = plt.subplots(1, 2, figsize=(50, 50))
                ax1.imshow(image)
                ax1.set_title('Image')
                ax2.imshow(truth)
                ax2.set_title('Ground Truth')

        self._log("All training images successfully loaded")

        plt.show()

        # Transpose data for succeeding computations
        self._data_true = np.asarray(self._data_true).T
        self._data_false = np.asarray(self._data_false).T

        # Calculate priors on True and False matches based on proportion of True pixels
        # in the training set
        pixels = self._data_true.shape[1] + self._data_false.shape[1] # Total no. of pixels
        self._prior = [self._data_true.shape[1] / pixels, self._data_false.shape[1] / pixels]

    def set_threshold(self, threshold):
        """Allows public adjustment of the threshold value by which to discriminate classification
        """
        self._threshold = threshold

    def train(self, k=1):
        """
        Trains the images loaded via `load_train_images` on a mixture of gaussians model over the
        colour channels. Fits these gaussian mixtures over pixels which match the class, and pixels
        which do not (giving two models in the end).

        Parameters:
            k - Number of gaussians to use

        Side effects:
            `_model_true` and `_model_false` are populated with the fitted models.
            `_k` is set to the number of gaussians used in the mixture

        Returns:
            None
        """

        self._k = k # number of gaussians to use in the model

        # Fit the model to pixels corresponding to and not to the class
        self._model_true = self._fit_model(self._data_true)
        self._model_false = self._fit_model(self._data_false)

    def tune(self, images, truths, max_k=5, display=True):
        """
        Tunes the hyperparameters (threshold and k) over a list of images and given ground truth
        labels (the development/validation set).

        Parameters:
            images - list of images to tune on (file paths)
            truths - list of corresponding ground truth images
            max_k - The maximum number of gaussians to use in a single model (higher can overfit
            and take a long time to tune over)
            display - whether or not to show images (ROC curves)

        Side effects:
            _k and _threshold will be changed to their optimum values

        Returns:
            None
        """

        self._log("Starting model tuning.")

        # For ease of computation, the pixels for each image are arranged in a single 'line', and
        # appended to each other. These pixels are stored in the following lists:
        full_image = None
        full_truth = None

        # Loop over each image and truth file, appending the pixels to the above lists
        for image, truth in zip(images, truths):

            # Data pre-processing
            image = plt.imread(image)
            if self._gabor:
                image = self._gabor.process(image)
            else:
                image = image / 255
            truth = (plt.imread(truth)[:, :, 2] != 0).astype(int)

            dim_x, dim_y, dim_z = image.shape

            # Perform the concatenations for both image and truth pixels
            image = image.reshape(dim_x * dim_y, dim_z)
            full_image = image if full_image is None else \
            np.concatenate((full_image, image), axis=0)

            truth = truth.reshape(dim_x * dim_y)
            full_truth = truth if full_truth is None else \
            np.concatenate((full_truth, truth), axis=0)

        # Add an extra axis (array -> multidimensional array, expected by _posterior_mask)
        full_image = full_image[:, None]
        full_truth = full_truth[:, None]

        # Keep track of best F1 and ROC metrics
        best_f1_score = [-np.inf, 0, 0] # F1 score, threshold, k
        best_roc = [-np.inf, 0, 0] # ROC, threshold, k

        # Loop over each 'k' (number of gaussians)
        for k in range(1, max_k+1):
            self.train(k) # Train a model on k gaussians
            posterior_true, _ = self._posterior_mask(full_image) # Classify each pixel

            sensitivities = []
            specificities = []

            # Sweep over the threshold from 0.01 to 0.99 in steps of 0.01 to find the optimum
            for threshold in range(1, 100):
                self.set_threshold(threshold/100)
                roc, f1_score, sensitivity, specificity = \
                self._calculate_merit(full_truth, posterior_true)
                sensitivities.append(sensitivity)
                specificities.append(specificity)
                if roc > best_roc[0]:
                    best_roc = [roc, self._threshold, k]
                if f1_score > best_f1_score[0]:
                    best_f1_score = [f1_score, self._threshold, k]

            # Plot an ROC curve
            if display:
                plt.plot(1 - np.array(specificities), sensitivities, label='k = {}'.format(k))

        # Add random guess line to ROC curve
        if display:
            plt.plot([0, 1], [0, 1], '--', label='Random guess')
            plt.xlabel('1 - specificity')
            plt.ylabel('sensitivity')
            plt.legend()
            plt.show()

        # Logging (for debugging and results)
        self._log("Tuning complete.")
        self._log("Best F1 score: {:.3f} for threshold of {} and k = {}".format(*best_f1_score))
        self._log("Best ROC score: {:.3f} for threshold of {} and k = {}".format(*best_roc))

        # If the best score uses a k different to that last trained on (max_k), the model needs
        # to be retrained using the optimum k
        if best_f1_score[2] != max_k:
            self._log("Retraining for k = {} corresponding to best F1 score"\
            .format(best_f1_score[2]))
            self.train(best_f1_score[2])

        # Set threshold to optimum value
        self.set_threshold(best_f1_score[1])

    def test(self, images, truths=None, display=True):
        """
        Allows testing of the model on images which were not seen during the training (or tuning)
        periods. If you have truth labels for these (ground truth image masks) these can also be
        passed through to allow for calculation of the F1 score and an ROC merit value.

        Parameters:
            images - list of images to train on (file paths)
            truths - list of ground truth images corresponding to  `images` list
            (black=False, white=True)
            display - whether or not to show the calculated masks alongside the original images

        Side effects:
            Displays images, actual truth masks (if provided) and calculated truth masks

        Returns:
            None
        """

        for image, truth in zip(images, truths):
            pixels = plt.imread(image)
            if self._gabor:
                pixels = self._gabor.process(pixels)
            else:
                pixels = pixels / 255

            # Calculate the classification of each pixel (and threshold for the binary data)
            posterior_true, posterior_true_binary = self._posterior_mask(pixels)

            # If the truth mask is given, quantitatively evaluate model
            if truths:
                truth = (plt.imread(truth)[:, :, 2] != 0).astype(int)
                roc, f1_score, _, _ = self._calculate_merit(truth, posterior_true)
                self._log("Image [{}]: F1 = {:.4f}, ROC = {:.4f}".format(image, f1_score, roc))

            # Display calculated masks alongside truth (if given) and original image
            if display:
                if truths:
                    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(50, 50))
                    ax1.imshow(pixels)
                    ax1.set_title('Image')
                    ax2.imshow(truth)
                    ax2.set_title('Ground Truth')
                    ax3.imshow(posterior_true)
                    ax3.set_title('Posterior')
                    ax4.imshow(posterior_true_binary)
                    ax4.set_title('Binary Posterior')
                else:
                    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(50, 50))
                    ax1.imshow(pixels)
                    ax1.set_title('Image')
                    ax2.imshow(posterior_true)
                    ax2.set_title('Posterior')
                    ax3.imshow(posterior_true_binary)
                    ax3.set_title('Binary Posterior')

        plt.show()

    def _posterior_mask(self, image):
        """
        Calculate the mask over the pixels corresponding to whether or not the pixel
        'conforms' to the class being modeled.

        Parameters:
            image - list of pixel data

        Side effects:
            None

        Returns:
            posterior_true - Probabilities of each pixel conforming to the modeled class
            posterior_true_binary - Probabilities as above, thresholded to 0, 1 by `self._threshold`
        """
        dim_y, dim_x, dim_z = image.shape

        # Calculate likelihood of each pixel matching the class (using the two separate models
        # generated over the 'True' pixels and 'False' pixels).
        likelihood_true = BinaryClassifier._likelihood(image.reshape(dim_y * dim_x, dim_z).T, \
        self._model_true).reshape(dim_y, dim_x)
        likelihood_false = BinaryClassifier._likelihood(image.reshape(dim_y * dim_x, dim_z).T, \
        self._model_false).reshape(dim_y, dim_x)

        posterior_true = likelihood_true * self._prior[0] / (likelihood_true * self._prior[0] \
         + likelihood_false * self._prior[1])
        posterior_true_binary = (posterior_true > self._threshold).astype(int)

        return posterior_true, posterior_true_binary

    def _calculate_merit(self, truth, calculated):
        """
        Calcultes merit scores for the classification process. Uses F1 scores and other metrics.

        Parameters:
            truth - list of [0, 1] pixels corresponding to correct classification (mask)
            calculated - list of pixels corresponding to calculated classification (posterior mask),
            which can be the original probabilities or thresholded to [0, 1]

        Side effects:
            None

        Returns: (all defined in the code below)
            roc
            f1_score
            sensitivity
            specificity
        """

        # First calculate the true/false positives/negatives. Each of these 4 possible labels can be
        # distinguished easily as below:
        differences = 2 * truth - (calculated > self._threshold).astype(int)
        true_pos = np.sum(differences == 1) # 2*1 - 1 = 1
        false_neg = np.sum(differences == 2) # 2*1 - 0 = 2
        false_pos = np.sum(differences == -1) # 2*0 - 1 = -1
        true_neg = np.sum(differences == 0) # 2*0 - 0 = 0

        # Calculate required metrics for F1 and ROC
        sensitivity = true_pos / (true_pos + false_neg + self._flt_min)
        specificity = true_neg / (true_neg + false_pos + self._flt_min)
        precision = true_pos / (true_pos + false_pos + self._flt_min)
        recall = sensitivity # same as sensitivity

        # F1 Score
        f1_score = 2 * (precision * recall) / (precision + recall + self._flt_min)

        # This is a derived metric, noting that both a high sensitivity and high specificity will
        # result in few false negatives and false positives, and hence a high 'ROC' is good. The
        # naming has been chosen based on its derivation from the ROC curve
        roc = sensitivity * specificity

        return roc, f1_score, sensitivity, specificity

    def _fit_model(self, data):
        """
        Fit a mixture of gaussians model to the given data

        Parameters:
            data - A list of pixels to which the model should be fit to (either corresponding to
            pixels which match the class, or those which do not)

        Side effects:
            None

        Returns:
            None
        """

        # Provide information for debugging
        self._log("Fitting {} gaussians to data".format(self._k))

        # Get shape information of the given data (covers number of pixels and number of channels)
        n_dims, n_data = data.shape

        # Calculate the mean over the channels (3-vector if 3 channels, such as RGB)
        mean = np.mean(data, axis=1)

        # And the covariance matrix (3x3 if 3 channels, such as RGB)
        cov = 1 / n_data * (data - mean[:, None]) @ (data - mean[:, None]).T

        post_hidden = np.zeros(shape=(self._k, n_data))
        model_estimate = dict()
        model_estimate['d'] = n_dims
        model_estimate['k'] = self._k

        # Initialise parameters (arbitrary values) - weights, means and covariances of each gaussian
        model_estimate['weight'] = (1 / self._k) * np.ones(shape=(self._k))
        model_estimate['mean'] = mean[:, None] * (0.5 + np.random.uniform(size=(self._k)))
        model_estimate['cov'] = \
        (2 + 0.4 * np.random.normal(size=(self._k)))[None, None] * cov[:, :, None]

        log_likelihood = BinaryClassifier._log_likelihood(data, model_estimate)

        # Keep track of number of iterations
        iteration_count = 1

        # Keep iterating until change in likelihood is less than self._eps
        # This performs the EM (expectation-maximisation) algorithm with some clever
        # tricks for vectorising the calculations over the data set to improve speed exponentially.
        # See the given PDF file (credit to UCL) for implementation details
        while True:

            # Expectation step
            for current_gauss in range(self._k):
                current_weight = model_estimate['weight'][current_gauss]
                current_mean = model_estimate['mean'][:, current_gauss]
                current_cov = model_estimate['cov'][:, :, current_gauss]
                post_hidden[current_gauss, :] = current_weight * \
                BinaryClassifier._gauss_probability(data, current_mean, current_cov)
            post_hidden /= np.sum(post_hidden, axis=0) + self._flt_min

            # Maximisation step
            responsibility_k = np.sum(post_hidden, axis=1)
            for current_gauss in range(self._k):
                current_responsibility = post_hidden[current_gauss, :]
                model_estimate['weight'][current_gauss] = \
                np.sum(current_responsibility) / np.sum(post_hidden)
                model_estimate['mean'][:, current_gauss] = \
                data @ current_responsibility / np.sum(current_responsibility)

                denom = responsibility_k[current_gauss]
                shifted = data - model_estimate['mean'][:, current_gauss][:, None]
                model_estimate['cov'][:, :, current_gauss] = \
                ((shifted * current_responsibility[None, :]) @ (shifted.T) / denom)

            new_log_likelihood = BinaryClassifier._log_likelihood(data, model_estimate)
            log_likelihood_delta = abs(new_log_likelihood - log_likelihood)

            self._log('Log Likelihood After Iter {} : {:4.3f}'\
            .format(iteration_count, new_log_likelihood))

            if log_likelihood_delta > self._eps:
                log_likelihood = new_log_likelihood
                iteration_count += 1
            else:
                break

        return model_estimate

    @staticmethod
    def _likelihood(data, model_estimate):
        """Calculate likelihood of seeing each pixel individually with our model estimate
        """
        n_data = data.shape[1]
        logs = np.zeros((model_estimate['k'], n_data))

        for current_c in range(model_estimate['k']):
            current_weight = model_estimate['weight'][current_c]
            current_mean = model_estimate['mean'][:, current_c]
            current_cov = model_estimate['cov'][:, :, current_c]
            logs[current_c, :] = current_weight * \
            BinaryClassifier._gauss_probability(data, current_mean, current_cov)

        # We only sum over the k's here as we want to keep the likelihood for each individual
        # pixel to calculate the posterior If we wanted the likelihood of all data, we would
        # multiply each value in likelihood together (not add, as not log!).
        likelihood = np.sum(logs, axis=0)

        return likelihood

    @staticmethod
    def _log_likelihood(data, model_estimate):
        """Calculate log likelihood of seeing the full image (all pixels) with our model estimate
        """

        n_data = data.shape[1]
        logs = np.zeros((model_estimate['k'], n_data))

        for current_c in range(model_estimate['k']):
            current_weight = model_estimate['weight'][current_c]
            current_mean = model_estimate['mean'][:, current_c]
            current_cov = model_estimate['cov'][:, :, current_c]
            logs[current_c, :] = current_weight * \
            BinaryClassifier._gauss_probability(data, current_mean, current_cov)

        # We want the likelihood of seeing all these pixels, and hence we add over k and all pixels,
        # not just k. Add not multiply due to log likelihood
        log_likelihood = np.sum(np.log(np.sum(logs, axis=0)))

        return np.asscalar(log_likelihood)

    @staticmethod
    def _gauss_probability(data, mean, cov):
        """Calculate probability of observing data points under a gauss. distribution ~ [mean, cov]
        """
        data = data - mean[:, None]
        sig = np.linalg.inv(cov)
        power = (data.T.dot(sig) * data.T).sum(1)
        prob = np.power(np.linalg.det(2 * np.pi * cov), -0.5) * np.exp(-0.5 * power)
        return prob
