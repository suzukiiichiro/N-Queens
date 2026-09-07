# 384Py_multifile_osexec_probe.py
#
# rev384 -- FEASIBILITY PROBE ONLY. Not a kernel revision, no GPU code,
# no constellations. This file exists to answer exactly two open
# questions before 383's real implementation (maxd-gated os.system
# dispatch to CRunner binaries + validation-helper split into its own
# file) is attempted:
#
#   Q1: does `import <local .py file>` survive `codon build -release`
#       when the imported file lives in the same directory as the
#       entry point?
#   Q2: does os.system(cmd) + a log-file round trip (an external shell
#       command writes a log file, Codon's native file I/O reads it
#       back) work as expected on the real cudacodon machine?
#
# Q2 is deliberately built the same way 374/382's own validate.sh
# harnesses already talk to their C binaries: redirect stdout to a
# log file, then grep/parse marker lines out of it. This probe just
# moves that same pattern one level in, from bash into Codon itself,
# since that is exactly the mechanism 383's real maxd->CRunner-binary
# dispatch will need.
#
# ===384-NAMING-NOTE===
# The imported helper file is named "rev384_helper_probe.py", NOT
# "384Py_helper_probe.py" as the project's usual "{rev}Py_..." prefix
# would suggest. Python/Codon import statements require the module
# name to be a valid identifier, and identifiers cannot start with a
# digit -- "import 384Py_helper_probe" is a syntax error. This is a
# one-off naming exception for files that must be import targets, not
# a change to the project's file-naming convention in general.
# ===384-NAMING-NOTE-END===

import os
import rev384_helper_probe

PROBE_LOG_DIR:str = "384_probe_logs"
PROBE_LOG_FILE:str = PROBE_LOG_DIR + "/384_osexec_probe.log"


def run_import_probe() -> bool:
  # Q1: if this call resolves at all, the import already survived
  # codon build -release by the time we get here -- but we still
  # exercise a return value from the helper module to make sure it's
  # not just a name that happened to link, but an actually-callable
  # function returning real data.
  value:int = rev384_helper_probe.helper_expected_value()
  line:str = rev384_helper_probe.helper_marker_line(value)
  print(line)
  ok:bool = (value == 12345)
  print(f"[384-import-probe] {'PASS' if ok else 'FAIL'} imported_value={value}")
  return ok


def run_osexec_probe() -> bool:
  # Q2: os.system() is the one process-launch primitive Codon exposes
  # natively (no subprocess/Popen module exists in Codon's stdlib as
  # of this writing). We use it exactly the way 383's real dispatch
  # will: fire a command, redirect its output to a log file, then
  # read that log file back with Codon's own file I/O and look for
  # marker lines -- the same convention as every "[xxx-run-done]"
  # line already used throughout this project's own log output.
  os.system(f"mkdir -p {PROBE_LOG_DIR}")

  cmd:str = (
    f'echo "[384-osexec-dummy-start] hello from external process" > {PROBE_LOG_FILE}; '
    f'echo "computed_value=12345" >> {PROBE_LOG_FILE}; '
    f'echo "[384-osexec-dummy-done] status=OK" >> {PROBE_LOG_FILE}'
  )
  rc:int = os.system(cmd)
  if rc != 0:
    print(f"[384-osexec-probe] FAIL os.system returned rc={rc}")
    return False

  found_start:bool = False
  found_value:bool = False
  found_done:bool = False

  f = open(PROBE_LOG_FILE, "r")
  for raw_line in f:
    line:str = raw_line.strip()
    if line.startswith("[384-osexec-dummy-start]"):
      found_start = True
    if line == "computed_value=12345":
      found_value = True
    if line.startswith("[384-osexec-dummy-done]") and "status=OK" in line:
      found_done = True
  f.close()

  ok:bool = found_start and found_value and found_done
  print(f"[384-osexec-probe] {'PASS' if ok else 'FAIL'} start={found_start} value={found_value} done={found_done} log={PROBE_LOG_FILE}")
  return ok


def main() -> None:
  print("384 probe: multi-file import + os.system log round-trip")
  import_ok:bool = run_import_probe()
  osexec_ok:bool = run_osexec_probe()
  overall:bool = import_ok and osexec_ok
  print(f"[384-probe-summary] import_probe={'PASS' if import_ok else 'FAIL'} osexec_probe={'PASS' if osexec_ok else 'FAIL'} overall={'PASS' if overall else 'FAIL'}")


if __name__ == "__main__":
  main()
