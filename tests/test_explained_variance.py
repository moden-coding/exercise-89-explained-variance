#!/usr/bin/env python3

import contextlib
import io
import re
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import sklearn

from src.explained_variance import explained_variance, main


def _spy(method):
    """Wrap a method so calls to it are recorded on `.mock` while the
    original implementation still runs."""
    mock = MagicMock(name="fit spy")

    def wrapper(self, *args, **kwargs):
        mock(*args, **kwargs)
        return method(self, *args, **kwargs)

    wrapper.mock = mock
    return wrapper


class TestExplainedVariance(unittest.TestCase):

    def test_values(self):
        v, ev = explained_variance()
        self.assertEqual(
            len(v), 10,
            msg="explained_variance() should return 10 variances (one per "
                "original column). Got %d." % (len(v),))
        self.assertEqual(
            len(ev), 10,
            msg="explained_variance() should return 10 explained variances "
                "(one per PCA component). Got %d." % (len(ev),))
        self.assertAlmostEqual(
            sum(v), 9.41021150221,
            msg="The sum of the raw variances should be about 9.4102.")
        self.assertAlmostEqual(
            sum(ev), 9.41021150221,
            msg="The sum of the explained variances (from PCA) should equal "
                "the sum of the raw variances, about 9.4102.")

    def test_pca(self):
        fit_method = _spy(sklearn.decomposition.PCA.fit)
        with patch.object(sklearn.decomposition.PCA, "fit", new=fit_method), \
             patch("src.explained_variance.PCA",
                   wraps=sklearn.decomposition.PCA) as mypca:
            v, ev = explained_variance()
            mypca.assert_called_once()
            args, kwargs = mypca.call_args
            if len(args) > 0:
                self.assertEqual(
                    args[0], 10,
                    msg="PCA should be constructed with n_components=10, "
                        "one component per original column.")
            fit_method.mock.assert_called()
            args, kwargs = fit_method.mock.call_args
            df = args[0]
            self.assertEqual(
                df.shape[0], 400,
                msg="The DataFrame passed to PCA.fit should have 400 rows "
                    "(one per data point). Got %d." % (df.shape[0],))
            self.assertEqual(
                df.shape[1], 10,
                msg="The DataFrame passed to PCA.fit should have 10 columns. "
                    "Got %d." % (df.shape[1],))

    def test_print(self):
        with patch("src.explained_variance.plt.show") as pshow:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                main()
            pshow.assert_called_once()
            out = buf.getvalue()
            self.assertIn(
                "The variances are:", out,
                msg="main() should print a line starting with "
                    "'The variances are:'.")
            self.assertIn(
                "The explained variances after PCA are:", out,
                msg="main() should print a line starting with 'The "
                    "explained variances after PCA are:'.")

            m = re.search(r"^The variances are: *(.*)$", out, re.MULTILINE)
            variances = m[1].split()
            self.assertEqual(
                len(variances), 10,
                msg="Expected ten variances to be printed after 'The "
                    "variances are:'.")
            for value in variances:
                self.assertRegex(
                    value, r"^\d+\.\d\d\d$",
                    msg="Each variance should be printed with exactly three "
                        "decimal places. Got %r." % (value,))

            m = re.search(
                r"^The explained variances after PCA are: *(.*)$", out,
                re.MULTILINE)
            evariances = m[1].split()
            self.assertEqual(
                len(evariances), 10,
                msg="Expected ten explained variances to be printed after "
                    "'The explained variances after PCA are:'.")
            for value in evariances:
                self.assertRegex(
                    value, r"^\d+\.\d\d\d$",
                    msg="Each explained variance should be printed with "
                        "exactly three decimal places. Got %r." % (value,))

    def test_plot(self):
        with patch("src.explained_variance.plt.plot") as myplot, \
             patch("src.explained_variance.plt.show") as myshow:
            main()
            myplot.assert_called_once()
            myshow.assert_called_once()
            args = myplot.call_args[0]
            self.assertTrue(
                (args[0] == np.arange(1, 11)).all(),
                msg="The x values passed to plt.plot should be 1..10 "
                    "(np.arange(1, 11)).")
            np.testing.assert_allclose(
                args[1],
                [8.075043, 8.890709, 9.410212, 9.410212, 9.410212, 9.410212,
                 9.410212, 9.410212, 9.410212, 9.410212],
                err_msg="The y values passed to plt.plot should be the "
                        "cumulative sum of the explained variances.")


if __name__ == '__main__':
    unittest.main()
