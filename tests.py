#!/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright © 2016 Mark Wolf
#
# This file is part of scimap.
#
# Scimap is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Scimap is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Scimap.  If not, see <http://www.gnu.org/licenses/>.

# flake8: noqa

import datetime as dt
import unittest
import math
import os
import shutil

import h5py
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
import pytz

from cases import XanespyTestCase
from xanespy.utilities import xycoord, prog, position, Extent
from xanespy.xanes_frameset import XanesFrameset, calculate_direct_whiteline, calculate_gaussian_whiteline
from xanespy.xanes_math import transform_images
from xanespy.frame import ( TXMFrame, xy_to_pixel, pixel_to_xy,
                        Pixel, rebin_image, apply_reference)
from xanespy.edges import KEdge, k_edges
from xanespy.importers import import_ssrl_frameset, _average_frames, magnification_correction
from xanespy.xradia import XRMFile, decode_ssrl_params, decode_aps_params
from xanespy.beamlines import (sector8_xanes_script, ssrl6_xanes_script,
                           Zoneplate, ZoneplatePoint, Detector)
from xanespy.txmstore import TXMStore

testdir = os.path.dirname(__file__)
ssrldir = os.path.join(testdir, 'ssrl-txm-data')
apsdir = os.path.join(testdir, 'aps-txm-data')

# Silence progress bars for testing
# prog.quiet = True


class SSRLScriptTest(unittest.TestCase):
    """Verify that a script is created for running an operando TXM
    experiment at SSRL beamline 6-2c. These tests conform to the
    results of the beamline's in-house script generator. They could be
    changed but the effects on the beamline operation should be
    checked first.
    """

    def setUp(self):
        self.output_path = os.path.join(testdir, 'ssrl_script.txt')
        self.scaninfo_path = os.path.join(testdir, 'ScanInfo_ssrl_script.txt')
        # Check to make sure the file doesn't already exists
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        assert not os.path.exists(self.output_path)
        # Values taken from SSRL 6-2c beamtime on 2015-02-22
        self.zp = Zoneplate(
            start=ZoneplatePoint(x=-7.40, y=-2.46, z=-1255.46, energy=8250),
            end=ZoneplatePoint(x=4.14, y=1.38, z=703.06, energy=8640),
        )

    def tearDown(self):
        return
        os.remove(self.output_path)
        os.remove(self.scaninfo_path)

    def test_scaninfo_generation(self):
        """Check that the script writes all the filenames to a ScanInfo file
        for TXM Wizard."""
        with open(self.output_path, 'w') as f:
            ssrl6_xanes_script(dest=f,
                               edge=k_edges["Ni_NCA"](),
                               binning=2,
                               zoneplate=self.zp,
                               iterations=["Test0", "Snorlax"],
                               frame_rest=0,
                               repetitions=8,
                               ref_repetitions=15,
                               positions=[position(3, 4, 5)],
                               reference_position=position(0, 1, 2),
                               abba_mode=False)
        scaninfopath = os.path.join(testdir, 'ScanInfo_ssrl_script.txt')
        self.assertTrue(os.path.exists(scaninfopath))
        with open(scaninfopath) as f:
            self.assertEqual(f.readline(), 'VERSION 1\n')
            self.assertEqual(f.readline(), 'ENERGY 1\n')
            self.assertEqual(f.readline(), 'TOMO 0\n')
            self.assertEqual(f.readline(), 'MOSAIC 0\n')
            self.assertEqual(f.readline(), 'MULTIEXPOSURE 4\n')
            self.assertEqual(f.readline(), 'NREPEATSCAN   1\n')
            self.assertEqual(f.readline(), 'WAITNSECS   0\n')
            self.assertEqual(f.readline(), 'NEXPOSURES   8\n')
            self.assertEqual(f.readline(), 'AVERAGEONTHEFLY   0\n')
            self.assertEqual(f.readline(), 'REFNEXPOSURES  15\n')
            self.assertEqual(f.readline(), 'REF4EVERYEXPOSURES   8\n')
            self.assertEqual(f.readline(), 'REFABBA 0\n')
            self.assertEqual(f.readline(), 'REFAVERAGEONTHEFLY 0\n')
            self.assertEqual(f.readline(), 'MOSAICUP   1\n')
            self.assertEqual(f.readline(), 'MOSAICDOWN   1\n')
            self.assertEqual(f.readline(), 'MOSAICLEFT   1\n')
            self.assertEqual(f.readline(), 'MOSAICRIGHT   1\n')
            self.assertEqual(f.readline(), 'MOSAICOVERLAP 0.20\n')
            self.assertEqual(f.readline(), 'MOSAICCENTRALTILE   1\n')
            self.assertEqual(f.readline(), 'FILES\n')
            self.assertEqual(f.readline(), 'ref_Test0_08250.0_eV_000of015.xrm\n')


    def test_script_generation(self):
        """Check that the script first moves to the first energy point and location."""
        ref_repetitions = 10
        with open(self.output_path, 'w') as f:
            ssrl6_xanes_script(dest=f,
                               edge=k_edges["Ni_NCA"](),
                               binning=2,
                               zoneplate=self.zp,
                               iterations=["Test0", "Snorlax"],
                               frame_rest=0,
                               ref_repetitions=ref_repetitions,
                               positions=[position(3, 4, 5), position(6, 7, 8)],
                               reference_position=position(0, 1, 2),
                               abba_mode=True)
        with open(self.output_path, 'r') as f:
            # Check that the first couple of lines set up the correct data
            self.assertEqual(f.readline(), ';; 2D XANES ;;\n')
            # Sets up the first energy correctly
            self.assertEqual(f.readline(), ';;;; set the MONO and the ZP\n')
            self.assertEqual(f.readline(), 'sete 8250.00\n')
            self.assertEqual(f.readline(), 'moveto zpx -7.40\n')
            self.assertEqual(f.readline(), 'moveto zpy -2.46\n')
            self.assertEqual(f.readline(), 'moveto zpz -1255.46\n')
            self.assertEqual(f.readline(), ';;;; Move to reference position\n')
            self.assertEqual(f.readline(), 'moveto x 0.00\n')
            self.assertEqual(f.readline(), 'moveto y 1.00\n')
            self.assertEqual(f.readline(), 'moveto z 2.00\n')
            # Collects the first set of references frames
            self.assertEqual(f.readline(), ';;;; Collect reference frames\n')
            self.assertEqual(f.readline(), 'setexp 0.50\n')
            self.assertEqual(f.readline(), 'setbinning 2\n')
            self.assertEqual(f.readline(), 'collect ref_Test0_08250.0_eV_000of010.xrm\n')
            # Read-out the rest of the "collect ..." commands
            [f.readline() for i in range(1, ref_repetitions)]
            # Moves to and collects first sample frame
            self.assertEqual(f.readline(), ';;;; Move to sample position 0\n')
            self.assertEqual(f.readline(), 'moveto x 3.00\n')
            self.assertEqual(f.readline(), 'moveto y 4.00\n')
            self.assertEqual(f.readline(), 'moveto z 5.00\n')
            self.assertEqual(f.readline(), ';;;; Collect frames sample position 0\n')
            self.assertEqual(f.readline(), 'setexp 0.50\n')
            self.assertEqual(f.readline(), 'setbinning 2\n')
            self.assertEqual(f.readline(), 'collect Test0_fov0_08250.0_eV_000of005.xrm\n')


class SSRLImportTest(XanespyTestCase):
    """Check that the program can import a collection of SSRL frames from
    a directory."""
    def setUp(self):
        prog.quiet = True
        self.hdf = os.path.join(ssrldir, 'testdata.h5')

    def tearDown(self):
        return
        if os.path.exists(self.hdf):
            os.remove(self.hdf)

    def test_imported_hdf(self):
        import_ssrl_frameset(ssrldir, hdf_filename=self.hdf, quiet=True)
        # Check that the file was created
        self.assertTrue(os.path.exists(self.hdf))
        with h5py.File(self.hdf, mode='r') as f:
            group = f['ssrl-test-data_rep1']
            keys = list(group.keys())
            self.assertIn('intensities', keys)
            self.assertEqual(group['intensities'].shape, (2, 1024, 1024))
            self.assertIn('references', keys)
            self.assertIn('absorbances', keys)
            self.assertEqual(group['pixel_sizes'].attrs['unit'], 'um')
            self.assertTrue(np.array_equal(group['energies'].value, np.array([8324., 8354.])))
            self.assertIn('timestamps', keys)
            self.assertIn('filenames', keys)
            self.assertIn('original_positions', keys)
            self.assertIn('relative_positions', keys)

    def test_params_from_ssrl(self):
        # First a reference frame
        ref_filename = "rep01_000001_ref_201511202114_NCA_INSITU_OCV_FOV01_Ni_08250.0_eV_001of010.xrm"
        result = decode_ssrl_params(ref_filename)
        expected = {
            'repetition': 1,
            'date_string': '',
            'sample_name': 'NCA_INSITU_OCV_FOV01_Ni',
            'position_name': '',
            'is_background': True,
            'energy': 8250.0,
        }
        self.assertEqual(result, expected)
        # Now a sample field of view
        sample_filename = "rep01_201511202114_NCA_INSITU_OCV_FOV01_Ni_08250.0_eV_001of010.xrm"
        result = decode_ssrl_params(sample_filename)
        expected = {
            'repetition': 1,
            'date_string': '',
            'sample_name': 'NCA_INSITU_OCV_FOV01_Ni',
            'position_name': '',
            'is_background': False,
            'energy': 8250.0,
        }
        self.assertEqual(result, expected)

    def test_magnification_correction(self):
        # Prepare some fake data
        img1 = [[1,1,1,1,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,0,0,0,1],
                [1,1,1,1,1]]
        img2 = [[0,0,0,0,0],
                [0,1,1,1,0],
                [0,1,0,1,0],
                [0,1,1,1,0],
                [0,0,0,0,0]]
        imgs = np.array([img1, img2], dtype=np.float)
        pixel_sizes = np.array([1, 2])
        scales, translations = magnification_correction(imgs, pixel_sizes)
        # Check that the first result is not corrected
        print(transform_images(imgs, scales=scales, translations=translations))
        print("scales:", scales, "Translations", translations)
        self.assertEqual(scales[1], 0.5)
        self.assertEqual(list(translations[0]), [0, 0])


class TXMStoreTest(XanespyTestCase):
    hdfname = os.path.join(ssrldir, 'txmstore-test.h5')
    @classmethod
    def setUpClass(cls):
        # Prepare an HDF5 file that these tests can use.
        import_ssrl_frameset(ssrldir, hdf_filename=cls.hdfname, quiet=True)

    @classmethod
    def tearDownClass(cls):
        # Delete temporary HDF5 files
        if os.path.exists(cls.hdfname):
            os.remove(cls.hdfname)

    def test_getters(self):
        store = TXMStore(hdf_filename=self.hdfname,
                         groupname='ssrl-test-data_rep1',
                         mode='r')
        self.assertEqual(store.intensities.shape, (2, 1024, 1024))
        self.assertEqual(store.references.shape, (2, 1024, 1024))
        self.assertEqual(store.absorbances.shape, (2, 1024, 1024))
        self.assertEqual(store.pixel_sizes.shape, (2,))
        self.assertEqual(store.energies.shape, (2,))
        self.assertEqual(store.timestamps.shape, (2, 2))
        self.assertEqual(store.positions.shape, (2, 3))

    def test_setters(self):
        pass


class ApsScriptTest(unittest.TestCase):
    """Verify that a script is created for running an operando
    TXM experiment at APS beamline 8-BM-B."""

    def setUp(self):
        self.output_path = os.path.join(testdir, 'aps_script.txt')
        # Check to make sure the file doesn't already exists
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        assert not os.path.exists(self.output_path)
        # Values taken from APS beamtime on 2015-11-11
        self.zp = Zoneplate(
            start=ZoneplatePoint(x=0, y=0, z=3110.7, energy=8313),
            step=9.9329 / 2 # Original script assumed 2eV steps
        )
        self.det = Detector(
            start=ZoneplatePoint(x=0, y=0, z=389.8, energy=8313),
            step=0.387465 / 2 # Original script assumed 2eV steps
        )

    def tear_down(self):
        pass
    # os.remove(self.output_path)

    def test_file_created(self):
        with open(self.output_path, 'w') as f:
            sector8_xanes_script(dest=f, edge=k_edges["Ni"](),
                                 zoneplate=self.zp, detector=self.det,
                                 names=["test_sample"], sample_positions=[])
        # Check that a file was created
        self.assertTrue(
            os.path.exists(self.output_path)
        )

    def test_binning(self):
        with open(self.output_path, 'w') as f:
            sector8_xanes_script(dest=f, edge=k_edges["Ni"](),
                                 binning=2, zoneplate=self.zp,
                                 detector=self.det, names=[],
                                 sample_positions=[])
        with open(self.output_path, 'r') as f:
            firstline = f.readline().strip()
        self.assertEqual(firstline, "setbinning 2")

    def test_exposure(self):
        with open(self.output_path, 'w') as f:
            sector8_xanes_script(dest=f, edge=k_edges["Ni"](),
                                 exposure=44, zoneplate=self.zp,
                                 detector=self.det, names=["test_sample"],
                                 sample_positions=[])
        with open(self.output_path, 'r') as f:
            f.readline()
            secondline = f.readline().strip()
        self.assertEqual(secondline, "setexp 44")

    def test_energy_approach(self):
        """This instrument can behave poorly unless the target energy is
        approached from underneath (apparently)."""
        with open(self.output_path, 'w') as f:
            sector8_xanes_script(dest=f, edge=k_edges['Ni'](),
                                 zoneplate=self.zp, detector=self.det,
                                 names=[], sample_positions=[])
        with open(self.output_path, 'r') as f:
            lines = f.readlines()
        # Check that the first zone plate is properly set
        assert False, "Test output of lines"
        print(lines)

    def test_first_frame(self):
        with open(self.output_path, 'w') as f:
            sector8_xanes_script(
                dest=f,
                edge=k_edges['Ni'](),
                sample_positions=[position(x=1653, y=-1727, z=0)],
                zoneplate=self.zp,
                detector=self.det,
                names=["test_sample"],
            )
        with open(self.output_path, 'r') as f:
            lines = f.readlines()
        # Check that x, y are set
        self.assertEqual(lines[2].strip(), "moveto x 1653.00")
        self.assertEqual(lines[3].strip(), "moveto y -1727.00")
        self.assertEqual(lines[4].strip(), "moveto z 0.00")
        # Check that the energy approach lines are in tact
        self.assertEqual(lines[5].strip(), "moveto energy 8150.00")
        self.assertEqual(lines[54].strip(), "moveto energy 8248.00")
        # Check that energy is set
        self.assertEqual(lines[55].strip(), "moveto energy 8250.00")
        # Check that zone-plate and detector are set
        self.assertEqual(lines[56].strip(), "moveto zpz 2797.81")
        self.assertEqual(lines[57].strip(), "moveto detz 377.59")
        # Check that collect command is sent
        self.assertEqual(
            lines[58].strip(),
            "collect test_sample_xanes0_8250_0eV.xrm"
        )

    def test_second_location(self):
        with open(self.output_path, 'w') as f:
            sector8_xanes_script(
                dest=f,
                edge=k_edges['Ni'](),
                sample_positions=[position(x=1653, y=-1727, z=0),
                                  position(x=1706.20, y=-1927.20, z=0)],
                zoneplate=self.zp,
                detector=self.det,
                names=["test_sample", "test_reference"],
            )
        with open(self.output_path, 'r') as f:
            lines = f.readlines()
        self.assertEqual(lines[247], "moveto x 1706.20\n")
        self.assertEqual(lines[248], "moveto y -1927.20\n")
        self.assertEqual(lines[250].strip(), "moveto energy 8150.00")

    def test_multiple_iterations(self):
        with open(self.output_path, 'w') as f:
            sector8_xanes_script(
                dest=f,
                edge=k_edges['Ni'](),
                sample_positions=[position(x=1653, y=-1727, z=0),
                                  position(x=1706.20, y=-1927.20, z=0)],
                zoneplate=self.zp,
                detector=self.det,
                iterations=["ocv"] + ["{:02d}".format(soc) for soc in range(1, 10)],
                names=["test_sample", "test_reference"],
            )
        with open(self.output_path, 'r') as f:
            lines = f.readlines()
        self.assertEqual(
            lines[58].strip(),
            "collect test_sample_xanesocv_8250_0eV.xrm"
        )
        self.assertEqual(
            lines[1090].strip(),
            "collect test_sample_xanes02_8342_0eV.xrm"
        )


class ZoneplateTest(XanespyTestCase):
    def setUp(self):
        # Values taken from APS beamtime on 2015-11-11
        self.aps_zp = Zoneplate(
            start=ZoneplatePoint(x=0, y=0, z=3110.7, energy=8313),
            z_step=9.9329 / 2 # Original script assumed 2eV steps
        )
        # Values taken from SSRL 6-2c on 2015-02-22
        self.ssrl_zp = Zoneplate(
            start=ZoneplatePoint(x=-7.40, y=-2.46, z=-1255.46, energy=8250),
            end=ZoneplatePoint(x=4.14, y=1.38, z=703.06, energy=8640),
        )

    def test_constructor(self):
        with self.assertRaises(ValueError):
            # Either `step` or `end` must be passed
            Zoneplate(start=None)
        with self.assertRaises(ValueError):
            # Passing both step and end is confusing
            Zoneplate(start=None, z_step=1, end=1)
        # Check that step is set if not expicitely passed
        zp = Zoneplate(
            start=ZoneplatePoint(x=0, y=0, z=3110.7, energy=8313),
            end=ZoneplatePoint(x=0, y=0, z=3120.6329, energy=8315)
        )
        self.assertApproximatelyEqual(zp.step.z, 9.9329 / 2)

    def test_z_from_energy(self):
        result = self.aps_zp.position(energy=8315).z
        self.assertApproximatelyEqual(result, 3120.6329)

    def test_position(self):
        result = self.aps_zp.position(energy=8315)
        self.assertApproximatelyEqual(result, (0, 0, 3120.6329))
        result = self.ssrl_zp.position(energy=8352)
        self.assertApproximatelyEqual(result, (-4.38, -1.46, -743.23))
        # self.assertApproximatelyEqual(result.x, 0)
        # self.assertApproximatelyEqual(result.y, 0)
        # self.assertApproximatelyEqual(result.z, 3120.6329)


class XrayEdgeTest(unittest.TestCase):
    def setUp(self):
        class DummyEdge(KEdge):
            regions = [
                (8250, 8290, 20),
                (8290, 8295, 1),
            ]
            pre_edge = (8250, 8290)
            post_edge = (8290, 8295)
            map_range = (8291, 8293)

        self.edge = DummyEdge()

    def test_energies(self):
        self.assertEqual(
            self.edge.all_energies(),
            [8250, 8270, 8290, 8291, 8292, 8293, 8294, 8295]
        )

    def test_norm_energies(self):
        self.assertEqual(
            self.edge.energies_in_range(),
            [8291, 8292, 8293]
        )

    def test_post_edge_xs(self):
        x = np.array([1, 2, 3])
        X = self.edge._post_edge_xs(x)
        expected = np.array([[1, 1], [2, 4], [3, 9]])
        self.assertTrue(np.array_equal(X, expected))
        # Test it with a single value
        x = 5
        X = self.edge._post_edge_xs(x)
        self.assertTrue(np.array_equal(X, [[5, 25]]))
        # Test with a single value but first order
        x = 5
        self.edge.post_edge_order = 1
        X = self.edge._post_edge_xs(x)
        self.assertTrue(np.array_equal(X, [[5]]))

# class TXMMapTest(HDFTestCase):
#
#     def setUp(self):
#         ret = super().setUp()
#         # Disable progress bars and notifications
#         prog.quiet = True
#         # Create an HDF Frameset for testing
#         self.fs = XanesFrameset(filename=self.hdf_filename,
#                                 groupname='mapping-test',
#                                 edge=k_edges['Ni'])
#         for i in range(0, 3):
#             frame = TXMFrame()
#             frame.energy = i + 8342
#             print(frame.energy)
#             frame.approximate_energy = frame.energy
#             ds = np.zeros(shape=(3, 3))
#             ds[:] = i + 1
#             frame.image_data = ds
#             self.fs.add_frame(frame)
#         self.fs[1].image_data.write_direct(np.array([
#             [0, 1, 4],
#             [1, 2.5, 1],
#             [4, 6, 0]
#         ]))
#         return ret

#     def test_max_energy(self):
#         expected = [
#             [8344, 8344, 8343],
#             [8344, 8344, 8344],
#             [8343, 8343, 8344]
#         ]
#         result = self.fs.whiteline_map()
#         print(result)
#         self.assertTrue(np.array_equal(result, expected))


class TXMMathTest(XanespyTestCase):
    """Holds tests for functions that perform base-level calculations."""

    def test_calculate_direct_whiteline(self):
        absorbances = [700, 705, 703]
        energies = [50, 55, 60]
        data = pd.Series(absorbances, index=energies)
        out, goodness = calculate_direct_whiteline(data, edge=k_edges['Ni_NCA']())
        self.assertApproximatelyEqual(out, 55)
        # Test using multi-dimensional absorbances (eg. image frames)
        absorbances = [np.array([700, 700]),
                       np.array([705, 703]),
                       np.array([703, 707])]
        data = pd.Series(absorbances, index=energies)
        out, goodness = calculate_direct_whiteline(data, edge=k_edges['Ni_NCA']())
        self.assertApproximatelyEqual(out[0], 55)
        self.assertApproximatelyEqual(out[1], 60)

    def test_calculate_gaussian_whiteline(self):
        """These test patterns do not contain enough data to properly fit,
        they merely test if the routine completes without errors."""
        absorbances = [700, 698, 705, 703, 702]
        energies = [8250, 8252, 8351, 8440, 8450]
        data = pd.Series(absorbances, index=energies)
        out, goodness = calculate_gaussian_whiteline(data, edge=k_edges['Ni_NCA']())
        self.assertApproximatelyEqual(out, 8333)
        # Test using multi-dimensional absorbances (eg. image frames)
        absorbances = [np.array([700, 700]),
                       np.array([698, 703]),
                       np.array([705, 704]),
                       np.array([703, 705]),
                       np.array([702, 707])]
        data = pd.Series(absorbances, index=energies)
        out, goodness = calculate_gaussian_whiteline(data, edge=k_edges['Ni_NCA']())
        self.assertApproximatelyEqual(out[0], 8333)
        self.assertApproximatelyEqual(out[1], 8333)

    def test_2d_whiteline(self):
        # Test using two-dimensional absorbances (ie. image frames)
        absorbances = [
            np.array([[502, 600],
                      [700, 800]]),
            np.array([[501, 601],
                      [702, 802]]),
            np.array([[500, 603],
                      [701, 801]]),
        ]
        energies = [50, 55, 60]
        data = pd.Series(absorbances, index=energies)
        out, goodness = calculate_direct_whiteline(data, edge=k_edges['Ni_NCA']())
        expected = [[50, 60],
                    [55, 55]]
        self.assertTrue(np.equal(out, expected))
        # self.assertApproximatelyEqual(out[0][0], 50)
        # self.assertApproximatelyEqual(out[0][1], 60)
        # self.assertApproximatelyEqual(out[1][0], 55)
        # self.assertApproximatelyEqual(out[1][1], 55)

    def test_fit_whiteline(self):
        filename = 'tests/testdata/NCA-cell2-soc1-fov1-xanesspectrum.tsv'
        data = pd.Series.from_csv(filename, sep="\t")
        # data = data[:8360]
        edge = k_edges['Ni_NCA']()
        peak, goodness = edge.fit(data)
        self.assertTrue(8352 < peak.center() < 8353,
                        "Center not within range {} eV".format(peak.center()))

        # Check that the residual differences are not too high
        # residuals = peak.residuals
        self.assertTrue(
            goodness < 0.01,
            "residuals too high: {}".format(goodness)
        )

class TXMFramesetTest(XanespyTestCase):
    """Set of python tests that work on full framesets and require data
    from multiple frames to make sense."""
    originhdf = os.path.join(ssrldir, 'txmstore-test.h5')
    temphdf = os.path.join(ssrldir, 'txmstore-test-tmp.h5')

    @classmethod
    def setUpClass(cls):
        # Prepare an HDF5 file that these tests can use.
        import_ssrl_frameset(ssrldir, hdf_filename=cls.originhdf, quiet=True)

    def setUp(self):
        # Copy the HDF5 file so we can safely make changes
        shutil.copy(self.originhdf, self.temphdf)
        self.frameset = XanesFrameset(filename=self.temphdf,
                                      edge=k_edges['Ni_NCA'])

    def tearDown(self):
        if os.path.exists(self.temphdf):
            os.remove(self.temphdf)

    @classmethod
    def tearDownClass(cls):
        # Delete temporary HDF5 files
        if os.path.exists(cls.originhdf):
            os.remove(cls.originhdf)
        pass

    def test_correct_magnification(self):
        self.frameset.correct_magnification()

class TXMFrameTest(XanespyTestCase):

    def test_average_frames(self):
        # Define three frames for testing
        frame1 = TXMFrame()
        frame1.image_data = np.array([
            [1, 2, 3],
            [3, 4, 5],
            [2, 5, 7],
        ])
        frame2 = TXMFrame()
        frame2.image_data = np.array([
            [3, 5, 7],
            [7, 9, 11],
            [5, 11, 15],
        ])
        frame3 = TXMFrame()
        frame3.image_data = np.array([
            [7, 11, 15],
            [15, 19, 23],
            [11, 23, 31],
        ])
        avg_frame = average_frames(frame1, frame2, frame3)
        expected_array = np.array([
            [11/3, 18/3, 25/3],
            [25/3, 32/3, 39/3],
            [18/3, 39/3, 53/3],
        ])
        # Check that it returns an array with same shape
        self.assertEqual(
            frame1.image_data.shape,
            avg_frame.image_data.shape
        )
        # Check that the averaging is correct
        self.assertTrue(np.array_equal(avg_frame.image_data, expected_array))

    def test_params_from_aps(self):
        """Check that the new naming scheme is decoded properly."""
        ref_filename = "ref_xanesocv_8250_0eV.xrm"
        result = decode_aps_params(ref_filename)
        expected = {
            'sample_name': 'ocv',
            'position_name': 'ref',
            'is_background': True,
            'energy': 8250.0,
        }
        self.assertEqual(result, expected)

    def test_pixel_size(self):
        sample_filename = "rep01_201502221044_NAT1050_Insitu03_p01_OCV_08250.0_eV_001of002.xrm"
        xrm = XRMFile(os.path.join(ssrldir, sample_filename), flavor="ssrl")
        self.assertApproximatelyEqual(
            xrm.um_per_pixel(),
            (0.0325783, 0.0325783)
        )

    def test_timestamp_from_xrm(self):
        sample_filename = "rep01_201502221044_NAT1050_Insitu03_p01_OCV_08250.0_eV_001of002.xrm"
        xrm = XRMFile(os.path.join(ssrldir, sample_filename), flavor="ssrl")
        # Check start time
        start = dt.datetime(2015, 2, 22,
                            10, 47, 19,
                            tzinfo=pytz.timezone('US/Pacific'))
        self.assertEqual(xrm.starttime(), start)
        # Check end time (offset determined by exposure time)
        end = dt.datetime(2015, 2, 22,
                          10, 47, 19, 500000,
                          tzinfo=pytz.timezone('US/Pacific'))
        self.assertEqual(xrm.endtime(), end)
        xrm.close()

        # Test APS frame
        sample_filename = "20151111_UIC_XANES00_sam01_8313.xrm"
        xrm = XRMFile(os.path.join(apsdir, sample_filename), flavor="aps-old1")
        # Check start time
        start = dt.datetime(2015, 11, 11, 15, 42, 38, tzinfo=pytz.timezone('US/Central'))
        self.assertEqual(xrm.starttime(), start)
        # Check end time (offset determined by exposure time)
        end = dt.datetime(2015, 11, 11, 15, 43, 16, tzinfo=pytz.timezone('US/Central'))
        self.assertEqual(xrm.endtime(), end)
        xrm.close()

    def test_extent(self):
        frame = TXMFrame()
        frame.relative_position = (0, 0, 0)
        frame.um_per_pixel = Pixel(vertical=0.0390625, horizontal=0.0390625)
        expected = Extent(
            left=-20, right=20,
            bottom=-10, top=10
        )
        self.assertEqual(frame.extent(img_shape=(512, 1024)), expected)

    def test_xy_to_pixel(self):
        extent = Extent(
            left=-1000, right=-900,
            top=300, bottom=250
        )
        result = xy_to_pixel(
            xy=xycoord(x=-950, y=250),
            extent=extent,
            shape=(10, 10)
        )
        self.assertEqual(result, Pixel(vertical=0, horizontal=5))

    def test_pixel_to_xy(self):
        extent = Extent(
            left=-1000, right=-900,
            top=300, bottom=250
        )
        result = pixel_to_xy(
            pixel=Pixel(vertical=10, horizontal=5),
            extent=extent,
            shape=(10, 10)
        )
        self.assertEqual(result, xycoord(x=-950, y=300))

    def test_shift_data(self):
        frame = TXMFrame()
        frame.image_data = self.hdf_file.create_dataset(
            name = 'shifting_data',
            data = np.array([
                [1, 2],
                [5, 7]
            ])
        )
        # Shift in x direction
        frame.shift_data(1, 0)
        expected_data = [
            [2, 1],
            [7, 5]
        ]
        self.assertTrue(np.array_equal(frame.image_data, expected_data))
        # Shift in negative y direction
        frame.shift_data(0, -1)
        expected_data = [
            [7, 5],
            [2, 1]
        ]
        self.assertTrue(np.array_equal(frame.image_data, expected_data))

    def test_rebinning(self):
        frame = TXMFrame()
        original_data = np.array([
            [1., 1., 3., 3.],
            [2, 2, 5, 5],
            [5, 6, 7, 9],
            [8, 12, 11, 10],
        ])
        # Check that binning to same shape return original array
        result_data = rebin_image(original_data, new_shape=(4, 4))
        self.assertTrue(
            result_data is original_data
        )
        # Check for rebinning by shape
        result_data = rebin_image(original_data, new_shape=(2, 2))
        expected_data = np.array([
            [6, 16],
            [31, 37]
        ])
        self.assertTrue(np.array_equal(result_data, expected_data))
        # Check for rebinning by factor
        frame.image_data = self.hdf_file.create_dataset(
            name = 'rebinning_data_factor',
            chunks = True,
            data = original_data
        )
        frame.rebin(factor=2)
        self.assertTrue(np.array_equal(frame.image_data, expected_data))
        # Check for error with no arguments
        with self.assertRaises(ValueError):
            frame.rebin()
        # Check for error if trying to rebin to larger shapes
        with self.assertRaisesRegex(ValueError, 'larger than original shape'):
            frame.rebin(factor=0.5)
        with self.assertRaisesRegex(ValueError, 'larger than original shape'):
            frame.rebin(new_shape=(6, 6))

    def test_rebin_odd(self):
        """There is a bug where oddly shaped arrays don't rebin well."""
        original_data = np.array([
            [1., 1., 3., 3., 4],
            [2, 2, 5, 5, 5],
            [5, 6, 7, 9, 2],
            [8, 12, 11, 10, 1],
        ])
        # Check that binning to same shape return original array
        result_data = rebin_image(original_data, new_shape=(2, 2))
        expected_data = np.array([
            [6, 16],
            [31, 37]
        ])
        self.assertTrue(np.array_equal(result_data, expected_data))

    def test_subtract_background(self):
        data = np.array([
            [10, 1],
            [0.1, 50]
        ])
        background = np.array([
            [100, 100],
            [100, 100]
        ])
        expected = np.array([
            [1, 2],
            [3, math.log10(2)]
        ])
        result = apply_reference(data, background)
        self.assertTrue(
            np.array_equal(result, expected)
        )
        # Check that uneven samples are rebinned
        data = np.array([
            [3, 1, 0.32, 0],
            [2, 4, 0, 0.68],
            [0.03, -.1, 22, 21],
            [0.07, 0.1, 0, 7],
        ])
        result = apply_reference(data, background)
        self.assertTrue(
            np.array_equal(result, expected)
        )

frameset_testdata = [
    np.array([
        [12, 8, 2.4, 0],
        [9, 11, 0, 1.6],
        [0.12, 0.08, 48, 50],
        [0.09, 0.11, 52, 50],
    ])
]

class MockDataset():
    def __init__(self, value=None):
        self.value = value

    @property
    def shape(self):
        return self.value.shape

class MockFrame(TXMFrame):
    image_data = MockDataset()
    hdf_filename = None
    def __init__(self, *args, **kwargs):
        pass


class MockFrameset(XanesFrameset):
    hdf_filename = None
    parent_groupname = None
    active_particle_idx = None
    edge = k_edges['Ni_NCA']
    def __init__(self, *args, **kwargs):
        pass

    def normalizer(self):
        return Normalize(0, 1)

    def __len__(self):
        return len(frameset_testdata)

    def __iter__(self):
        for d in frameset_testdata:
            frame = MockFrame()
            frame.image_data.value = d
            yield frame

# class TXMGtkViewerTest(unittest.TestCase):
#     @unittest.expectedFailure
#     def test_background_frame(self):
#         from txm import gtk_viewer
#         fs = MockFrameset()
#         viewer = gtk_viewer.GtkTxmViewer(frameset=fs,
#                                          plotter=plotter.DummyGtkPlotter(frameset=fs))

if __name__ == '__main__':
    unittest.main()
