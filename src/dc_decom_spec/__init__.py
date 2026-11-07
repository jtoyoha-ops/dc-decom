"""DC-DECOM spec version (the Energies Section 3 formulation, switchable fixes).

A copy of the v1 package ``dc_decom`` as of v1.0.1 in which the
implementation errors E1-E8 (code contradicting an explicit statement of
Energies 19(10) 2485, Section 3 and Tables 6-9) can be fixed one at a time
through ``SpecFixes``, and the points the paper does not state can be varied
through ``SpecReadings``. With all fixes (``SpecFixes()``), the default
readings and the inputs ``data/parameters_spec`` it reproduces the v1
package ``dc_decom`` of v2.0.0 (tests/spec/test_legacy_equivalence.py).
``SpecFixes.none()`` reproduces the v1.0.1 package when given the v1.0.1
inputs (``data/parameters_csv`` at tag v1.0.1).
"""
__version__ = "2.0.0-spec"
__author__ = "Jin Toyohara"
