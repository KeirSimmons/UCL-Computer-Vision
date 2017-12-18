"""
Classes:
    Gabor - Adds functionality for applying Gabor filters to an image
"""

import numpy as np
import cv2

class Gabor:

    """
    Takes in an image and applies a Gabor filter to it with customisable parameters.

    Public methods:
        process - runs an image through the gabor filter
        filters - Returns a list of filters (should be used for debugging purposes only)

    Public instance variables:
        None

    Modified from: https://gist.github.com/odebeir/5237529
    """

    def __init__(self, params=None):
        self._filters = None
        self._params = {} if not isinstance(params, dict) else params

        # Default values of params:
        self._set_defaults([
            ('ksize', 31),
            ('sigma', 1.0),
            ('lambda', 12.0),
            ('gamma', 0.02),
            ('psi', 0)
        ])

        # Build the filters
        self._build_filters()

    def _set_defaults(self, defaults):
        """
        Set default values in a dictionary (which can be overridden via the `params`
        parameter in `__init__`)
        """
        for key, value in defaults:
            # If key doesn't exist, set it to its default value
            if self._get_param(key) is None:
                self._set_param(key, value)

    def _get_param(self, key):
        # Returns `None` if key does not exist
        return self._params.get(key)

    def _set_param(self, key, value):
        self._params[key] = value

    def _build_filters(self):
        """ returns a list of kernels in several orientations
        """
        filters = []
        ksize = self._params['ksize']
        for theta in np.arange(0, np.pi, np.pi / 32):
            params = {
                'ksize':(ksize, ksize),
                'sigma':self._params['sigma'],
                'theta':theta,
                'lambd':self._params['lambda'],
                'gamma':self._params['gamma'],
                'psi':self._params['psi'],
                'ktype':cv2.CV_32F
            }
            kern = cv2.getGaborKernel(**params)
            kern /= 1.5*kern.sum()
            filters.append((kern, params))
        self._filters = filters

    def process(self, img):
        """ returns the img filtered by the filter list
        """
        accum = np.zeros_like(img)
        for kern, _ in self._filters:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            np.maximum(accum, fimg, accum)
        return accum

    def filters(self):
        """ Tool to examine the calculated filters for debugging purposes
        """
        return self._filters
