import argparse
import logging
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

from core.DigitisationPipeline import DigitisationPipeline


def initMain (arg0_argv=None):
  """
  @brief Startup CLI entrypoint for McEvedy & Jones computer vision digitisation into output/mcevedy_data.json.
  @param arg0_argv Optional list of command-line arguments.
  @return Integer process exit code (0 on success, 1 on failure).
  """
  #Convert from parameters
  argv_array = arg0_argv if arg0_argv is not None else sys.argv[1:]

  #Declare local instance variables
  args_obj = None
  exit_code = 1
  is_ok = False
  parser_obj = argparse.ArgumentParser(
    description="Digitise McEvedy & Jones (1978) population curves into output/mcevedy_data.json."
  )
  pipeline_obj = None

  logging.basicConfig(
    format="[%(levelname)s] %(message)s",
    level=logging.INFO
  )

  parser_obj.add_argument(
    "--config",
    default="common/digitisation/pipeline_config.json5",
    help="Path to pipeline JSON5 configuration file."
  )
  args_obj = parser_obj.parse_args(argv_array)

  pipeline_obj = DigitisationPipeline(args_obj.config, {"repo_root": REPO_ROOT})
  is_ok = pipeline_obj.F_runPipeline()
  exit_code = 0 if is_ok else 1

  #Return statement
  return exit_code


if __name__ == "__main__":
  sys.exit(initMain())

