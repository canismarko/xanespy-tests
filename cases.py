import os

import unittest
import h5py

from xanespy.utilities import prog


class XanespyTestCase(unittest.TestCase):

    def run(self, *args, **kwargs):
        prog.quiet = True
        return super().run(*args, **kwargs)

    def assertApproximatelyEqual(self, actual, expected,
                                 tolerance=0.01, msg=None):
        """Assert that 'actual' is within relative 'tolerance' of
        'expected'."""
        # Check for lists
        if isinstance(actual, list) or isinstance(actual, tuple):
            for a, b in zip(actual, expected):
                self.assertApproximatelyEqual(a, b, tolerance=tolerance)
        else:
            diff = abs(actual - expected)
            acceptable_diff = abs(expected * tolerance)
            if diff > acceptable_diff:
                msg = "{actual} is not close to {expected}"
                self.fail(msg=msg.format(actual=actual, expected=expected))


# class HDFTestCase(XanespyTestCase):
#     """A test case that sets up and tears down an HDF file."""
#     def setUp(self):
#         curdir = os.path.dirname(os.path.realpath(__file__))
#         self.hdf_filename = os.path.join(curdir, 'txm-frame-test.hdf')
#         if os.path.exists(self.hdf_filename):
#             os.remove(self.hdf_filename)
#         self.hdf_file = h5py.File(self.hdf_filename, 'w-')

#     def tearDown(self):
#         self.hdf_file.close()
#         if os.path.exists(self.hdf_filename):
#             os.remove(self.hdf_filename)
