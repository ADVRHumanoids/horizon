"""Opt-in end-to-end regression for profiling and non-profiling native builds."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get("HORIZON_KYON_EXAMPLE_BINARY"),
                     "Set HORIZON_KYON_EXAMPLE_BINARY to enable the Kyon integration test")
class KyonCollisionExampleTest(unittest.TestCase):
    def test_comparison_report_and_independent_kinematics(self):
        import pinocchio as pin

        with tempfile.TemporaryDirectory(prefix="kyon-example-test-") as temporary:
            output = Path(temporary)
            command = [sys.executable, "-m", "horizon.examples.kyon_collision",
                       "--binary", os.environ["HORIZON_KYON_EXAMPLE_BINARY"],
                       "--N", "12", "--collision", "both", "--samples", "1",
                       "--warmup", "0", "--output", temporary]
            if os.environ.get("KYON_URDF"):
                command += ["--urdf", os.environ["KYON_URDF"]]
            result = subprocess.run(command, capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            manifest = json.loads((output / "results.json").read_text())
            self.assertEqual([c["mode"] for c in manifest["cases"]], ["off", "on"])
            model = pin.buildModelFromUrdf(manifest["urdf"])
            data = model.createData()
            target = np.asarray(manifest["cases"][0]["profile"]["target"])
            for case in manifest["cases"]:
                directory = output / case["mode"]
                profile = case["profile"]; sample = profile["samples"][0]
                self.assertTrue(case["valid"])
                self.assertTrue(sample["solver_converged"])
                self.assertGreater(sample["min_self_clearance"], 0)
                if case["mode"] == "on":
                    self.assertGreater(sample["min_sdf_clearance"], .01)
                    self.assertGreater(sample["collision_derivative_calls"], 0)
                else:
                    self.assertLess(sample["min_sdf_clearance"], -.005)
                    self.assertEqual(sample["collision_derivative_calls"], 0)
                with np.load(directory / "replay.npz") as replay:
                    self.assertEqual(replay["q"].shape, (13, model.nq))
                    self.assertAlmostEqual(replay["environment_clearance"].min(),
                                           sample["min_sdf_clearance"], places=8)
                    self.assertAlmostEqual(replay["self_clearance"].min(),
                                           sample["min_self_clearance"], places=8)
                    for q in replay["q"]:
                        self.assertTrue(pin.isNormalized(model, q, 1e-8))
                        pin.forwardKinematics(model, data, q)
                        pin.updateFramePlacements(model, data)
                        feet = np.array([data.oMf[model.getFrameId(f"contact_{i}")].translation
                                         for i in range(1, 5)])
                        np.testing.assert_allclose(feet, replay["feet"], atol=1e-4)
                    tool = data.oMf[model.getFrameId("dagana_1_base")].translation
                    self.assertLess(np.linalg.norm(tool - target), .005)
                phases = ("linearize_quadratize", "backward_pass", "line_search") if profile["profiling_enabled"] else ()
                for phase in phases:
                    stats = sample["phase_statistics"][phase]
                    self.assertGreater(stats["calls"], 0)
                    self.assertGreater(stats["total_us"], 0)
                    self.assertLessEqual(stats["p50_us"], stats["p95_us"])
                if not profile["profiling_enabled"]:
                    self.assertFalse(sample.get("phase_statistics"))
                    self.assertIn("Internal solver timings unavailable", (output / "report.html").read_text())
                with (directory / "trajectory_iterations_0.csv").open() as stream:
                    trace = list(csv.DictReader(stream))
                elapsed = np.array([float(row["elapsed_ms"]) for row in trace])
                self.assertTrue(np.all(np.diff(elapsed) >= 0))
                self.assertTrue(all(float(row["callback_interval_ms"]) >= 0 for row in trace))
                self.assertTrue(any(row["accepted"] == "0" for row in trace)
                                or case["mode"] == "off")
            for filename in ("report.html", "profile.svg", "profile.png"):
                self.assertGreater((output / filename).stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
