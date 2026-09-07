# rev384_helper_probe.py
#
# rev384 -- import target for 384Py_multifile_osexec_probe.py's Q1
# probe (does a local multi-file `import` survive `codon build
# -release`?). Deliberately tiny: no logic that matters beyond
# "does this get linked in and actually execute".
#
# Named "rev384_..." rather than "384Py_..." because it is imported
# by name, and identifiers cannot start with a digit -- see the
# ===384-NAMING-NOTE=== block in 384Py_multifile_osexec_probe.py.

PROBE_HELPER_TAG:str = "helper-v1"


def helper_marker_line(value:int) -> str:
  # Same "[tag] key=value" convention as the project's existing log
  # markers (e.g. [gpu-hybrid-run-done], [maxd-check-done]).
  return f"[384-helper-computed] tag={PROBE_HELPER_TAG} value={value}"


def helper_expected_value() -> int:
  return 12345
