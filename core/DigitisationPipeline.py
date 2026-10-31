import json
import logging
import math
import os
import pymupdf
import re
import time

from core.CurveExtractor import CurveExtractor
from core.GridCalibrator import GridCalibrator
from core.TableInterpolator import TableInterpolator


class DigitisationPipeline:
  """
  @brief Orchestrates end-to-end computer vision digitisation of McEvedy & Jones (1978) into annual regional population series.
  """

  def __init__ (self, arg0_config_path, arg1_options=None):
    """
    @brief Initialises DigitisationPipeline with configuration file path and workspace root options.
    @param arg0_config_path Relative or absolute path to pipeline_config.json5.
    @param arg1_options Optional dictionary supporting 'repo_root'.
    """
    #Convert from parameters
    config_path = str(arg0_config_path) if arg0_config_path is not None else ""
    options = arg1_options if arg1_options is not None else {}

    #Initialise options
    if "repo_root" not in options:
      options["repo_root"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    #Declare local instance variables
    repo_root = os.path.abspath(str(options["repo_root"]))

    self.calibrator_obj = None
    self.catalogue_obj = {}
    self.config_obj = {}
    self.config_path = config_path if os.path.isabs(config_path) else os.path.join(repo_root, config_path)
    self.extractor_obj = None
    self.interpolator_obj = None
    self.output_abs_path = ""
    self.pdf_abs_path = ""
    self.reference_abs_path = ""
    self.reference_obj = {}
    self.repo_root = repo_root


  def A_loadConfigurations (self):
    """
    @brief Step A: Loads JSON5 pipeline and chart catalogue configurations plus the read-only reference JSON schema.
    @return True if all configurations and input files loaded cleanly; False otherwise.
    """
    #Declare local instance variables
    cat_rel_path = ""
    catalogue_abs_path = ""
    out_cfg_obj = None
    raw_cat_text = ""
    raw_cfg_text = ""
    raw_ref_text = ""

    if not os.path.isfile(self.config_path):
      logging.error("DigitisationPipeline.A_loadConfigurations could not find config file: %s", self.config_path)
      return False

    with open(self.config_path, "r", encoding="utf-8") as cfg_file:
      raw_cfg_text = cfg_file.read()
    self.config_obj = self.parseJson5Text(raw_cfg_text)
    if not isinstance(self.config_obj, dict) or len(self.config_obj) == 0:
      logging.error("DigitisationPipeline.A_loadConfigurations failed to parse pipeline configuration.")
      return False

    out_cfg_obj = self.config_obj.get("output", {})
    cat_rel_path = str(out_cfg_obj.get("catalogue_path", "common/digitisation/chart_catalogue.json5"))
    catalogue_abs_path = cat_rel_path if os.path.isabs(cat_rel_path) else os.path.join(self.repo_root, cat_rel_path)
    self.pdf_abs_path = os.path.join(self.repo_root, str(out_cfg_obj.get("pdf_path", "mcevedy_and_jones.pdf")))
    self.reference_abs_path = os.path.join(self.repo_root, str(out_cfg_obj.get("reference_json_path", "mcevedy_data.json")))
    self.output_abs_path = os.path.join(self.repo_root, str(out_cfg_obj.get("output_path", "output/mcevedy_data.json")))

    if os.path.normcase(os.path.abspath(self.output_abs_path)) == os.path.normcase(os.path.abspath(self.reference_abs_path)):
      logging.error("DigitisationPipeline.A_loadConfigurations refuses to target reference mcevedy_data.json.")
      return False
    if not os.path.isfile(catalogue_abs_path):
      logging.error("DigitisationPipeline.A_loadConfigurations could not find chart catalogue: %s", catalogue_abs_path)
      return False
    if not os.path.isfile(self.pdf_abs_path):
      logging.error("DigitisationPipeline.A_loadConfigurations could not find source PDF: %s", self.pdf_abs_path)
      return False
    if not os.path.isfile(self.reference_abs_path):
      logging.error("DigitisationPipeline.A_loadConfigurations could not find reference JSON: %s", self.reference_abs_path)
      return False

    with open(catalogue_abs_path, "r", encoding="utf-8") as cat_file:
      raw_cat_text = cat_file.read()
    self.catalogue_obj = self.parseJson5Text(raw_cat_text)

    with open(self.reference_abs_path, "r", encoding="utf-8") as ref_file:
      raw_ref_text = ref_file.read()
    self.reference_obj = json.loads(raw_ref_text)

    self.calibrator_obj = GridCalibrator(self.config_obj)
    self.extractor_obj = CurveExtractor(self.config_obj, self.calibrator_obj)
    self.interpolator_obj = TableInterpolator(self.config_obj)

    #Return statement
    return True


  def B_processChartPages (self):
    """
    @brief Step B: Renders each unique PDF chart page once, calibrates its 2D axis mesh and extracts all regional curves.
    @return Dictionary mapping extracted task keys to annual population dictionaries, or None on error.
    """
    #Declare local instance variables
    all_page_numbers = []
    all_regions = self.catalogue_obj.get("regions", [])
    extracted_obj = {}
    page_tasks_obj = {}
    pdf_doc = None
    seen_combined_keys = set()
    total_pages = 0

    if len(all_regions) == 0:
      logging.error("DigitisationPipeline.B_processChartPages found zero regions in chart catalogue.")
      return None

    for i in range(len(all_regions)):
      local_entry = all_regions[i]
      local_source = str(local_entry.get("source", ""))
      if local_source == "chart":
        local_page = int(local_entry["page"])
        if local_page not in page_tasks_obj:
          page_tasks_obj[local_page] = []
        page_tasks_obj[local_page].append({
          "panel_cfg": local_entry,
          "task_key": str(local_entry["key"])
        })
      elif local_source == "split_chart":
        local_page = int(local_entry["page"])
        if local_page not in page_tasks_obj:
          page_tasks_obj[local_page] = []
        local_comb_key = str(local_entry["combined_key"])
        if local_comb_key not in seen_combined_keys:
          seen_combined_keys.add(local_comb_key)
          page_tasks_obj[local_page].append({
            "panel_cfg": {
              "c_left": 0,
              "inset_masks": local_entry.get("combined_inset_masks", []),
              "key": local_comb_key,
              "r_bot": local_entry["r_bot_combined"],
              "r_top": local_entry["r_top_combined"],
              "unit_per_row": local_entry["unit_per_row"],
              "year_start": local_entry["year_start"]
            },
            "task_key": local_comb_key
          })
        page_tasks_obj[local_page].append({
          "panel_cfg": {
            "c_left": local_entry.get("c_left_sub", 12),
            "inset_masks": local_entry.get("sub_inset_masks", []),
            "key": str(local_entry["key"]),
            "r_bot": local_entry["r_bot_sub"],
            "r_top": local_entry["r_top_sub"],
            "unit_per_row": local_entry["unit_per_row"],
            "year_start": local_entry["year_split"]
          },
          "task_key": str(local_entry["key"]) + "__sub"
        })

    all_page_numbers = sorted(page_tasks_obj.keys())
    total_pages = len(all_page_numbers)
    pdf_doc = pymupdf.open(self.pdf_abs_path)

    for i in range(total_pages):
      local_pno = all_page_numbers[i]
      local_render_obj = self.calibrator_obj.A_renderAndNormalisePage(pdf_doc, local_pno)
      if local_render_obj is None:
        pdf_doc.close()
        return None
      local_od_img = local_render_obj["od_img"]
      local_vert_obj = self.calibrator_obj.B_detectVerticalLattice(local_od_img)
      local_grid_obj = self.calibrator_obj.C_detectHorizontalMesh(local_od_img, local_vert_obj)
      if local_grid_obj is None:
        pdf_doc.close()
        return None

      local_tasks_array = page_tasks_obj[local_pno]
      for x in range(len(local_tasks_array)):
        local_task = local_tasks_array[x]
        local_cfg = local_task["panel_cfg"]
        local_tkey = local_task["task_key"]

        local_box_mask = self.extractor_obj.A_maskTitleAndInsetBoxes(local_od_img, local_grid_obj, local_cfg)
        local_clean_obj = self.extractor_obj.B_removeGridAndBorders(local_od_img, local_box_mask, local_grid_obj, local_cfg)
        local_filt_obj = self.extractor_obj.D_filterGlyphComponents(local_clean_obj, local_grid_obj, local_cfg)
        local_seam_obj = self.extractor_obj.E_traceViterbiSeam(local_filt_obj)
        local_pop_obj = self.extractor_obj.F_sampleAnnualPopulation(local_seam_obj, local_filt_obj, local_grid_obj, local_cfg)
        extracted_obj[local_tkey] = local_pop_obj

      if (i + 1) % 12 == 0 or (i + 1) == total_pages:
        logging.info("Processed %d / %d chart pages (up to PDF page %d).", i + 1, total_pages, local_pno)

    pdf_doc.close()

    #Return statement
    return extracted_obj


  def C_assembleDataset (self, arg0_extracted_obj):
    """
    @brief Step C: Combines CV chart series, split panels, island tables and derived series in exact reference key order.
    @param arg0_extracted_obj Dictionary of extracted panel series from B_processChartPages.
    @return Ordered 110-region dataset dictionary matching mcevedy_data.json schema.
    """
    #Convert from parameters
    extracted_obj = arg0_extracted_obj

    #Declare local instance variables
    all_ref_keys = list(self.reference_obj.keys())
    all_regions = self.catalogue_obj.get("regions", [])
    catalogue_map_obj = {}
    dataset_obj = {}

    if extracted_obj is None:
      logging.error("DigitisationPipeline.C_assembleDataset received None extracted_obj.")
      return None

    for i in range(len(all_regions)):
      local_item = all_regions[i]
      catalogue_map_obj[str(local_item["key"])] = local_item

    for i in range(len(all_ref_keys)):
      local_key = all_ref_keys[i]
      if local_key not in catalogue_map_obj:
        logging.error("DigitisationPipeline.C_assembleDataset missing catalogue entry for region '%s'.", local_key)
        return None

      local_entry = catalogue_map_obj[local_key]
      local_source = str(local_entry.get("source", ""))
      local_colour_array = list(self.reference_obj[local_key].get("colour", [0, 0, 0]))
      local_pop_obj = {}

      if local_source == "chart":
        local_pop_obj = extracted_obj.get(local_key, {})
      elif local_source == "split_chart":
        local_comb_pop = extracted_obj.get(str(local_entry["combined_key"]), {})
        local_sub_pop = extracted_obj.get(local_key + "__sub", {})
        local_ot_pop = extracted_obj.get(str(local_entry["counterpart_key"]) + "__sub", {})
        local_pop_obj = self.interpolator_obj.C_combineSplitPanelSeries(
          local_comb_pop,
          local_sub_pop,
          local_ot_pop,
          {
            "default_early_share": local_entry.get("default_early_share", 0.5),
            "share_blend_start_year": local_entry.get("share_blend_start_year", 1500),
            "year_split": local_entry.get("year_split", 1600),
            "year_start": local_entry.get("year_start", -200)
          }
        )
      elif local_source == "table":
        local_pop_obj = self.interpolator_obj.A_interpolateTableAnnual(local_entry.get("anchors", {}))
      elif local_source == "derived":
        local_base_key = str(local_entry.get("base_region", ""))
        if local_base_key in dataset_obj:
          local_base_pop = dataset_obj[local_base_key]["population"]
        else:
          local_base_pop = extracted_obj.get(local_base_key, {})
        local_pop_obj = self.interpolator_obj.B_computeDerivedSeries(local_base_pop, float(local_entry.get("multiplier", 1.0)))
      else:
        logging.error("DigitisationPipeline.C_assembleDataset encountered unknown source '%s' for '%s'.", local_source, local_key)
        return None

      dataset_obj[local_key] = {
        "colour": local_colour_array,
        "population": local_pop_obj
      }

    #Return statement
    return dataset_obj


  def D_validateDataset (self, arg0_dataset_obj):
    """
    @brief Step D: Validates key ordering, RGB colour preservation, contiguous annual coverage to 1975 and non-negativity.
    @param arg0_dataset_obj Assembled dataset dictionary from C_assembleDataset.
    @return True if all 110 regions pass schema and continuity checks; False otherwise.
    """
    #Convert from parameters
    dataset_obj = arg0_dataset_obj

    #Declare local instance variables
    all_out_keys = []
    all_ref_keys = list(self.reference_obj.keys())
    end_year_cap = int(self.config_obj.get("x_axis", {}).get("end_year_cap", 1975))

    if not isinstance(dataset_obj, dict):
      logging.error("DigitisationPipeline.D_validateDataset received non-dict dataset_obj.")
      return False

    all_out_keys = list(dataset_obj.keys())
    if all_out_keys != all_ref_keys:
      logging.error("DigitisationPipeline.D_validateDataset key order mismatch (%d vs %d).", len(all_out_keys), len(all_ref_keys))
      return False

    for i in range(len(all_out_keys)):
      local_key = all_out_keys[i]
      local_reg_obj = dataset_obj[local_key]
      local_ref_colour = list(self.reference_obj[local_key].get("colour", []))
      if local_reg_obj.get("colour") != local_ref_colour:
        logging.error("DigitisationPipeline.D_validateDataset colour mismatch on '%s'.", local_key)
        return False

      local_pop_obj = local_reg_obj.get("population", {})
      if not isinstance(local_pop_obj, dict) or len(local_pop_obj) == 0:
        logging.error("DigitisationPipeline.D_validateDataset empty population series on '%s'.", local_key)
        return False

      local_yr_keys = list(local_pop_obj.keys())
      local_start_yr = int(local_yr_keys[0])
      local_expected_keys = [str(x) for x in range(local_start_yr, end_year_cap + 1)]
      if local_yr_keys != local_expected_keys:
        logging.error("DigitisationPipeline.D_validateDataset non-contiguous annual years on '%s' (%s .. %s).", local_key, local_yr_keys[0], local_yr_keys[-1])
        return False

      for x in range(len(local_yr_keys)):
        local_val = float(local_pop_obj[local_yr_keys[x]])
        if not math.isfinite(local_val) or local_val < 0.0:
          logging.error("DigitisationPipeline.D_validateDataset invalid population %s on '%s' (%s).", local_val, local_key, local_yr_keys[x])
          return False

    #Return statement
    return True


  def E_writeOutputJson (self, arg0_dataset_obj):
    """
    @brief Step E: Writes the validated dataset to output/mcevedy_data.json while protecting the root reference file.
    @param arg0_dataset_obj Validated 110-region dataset dictionary.
    @return True if written successfully; False otherwise.
    """
    #Convert from parameters
    dataset_obj = arg0_dataset_obj

    #Declare local instance variables
    out_dir = ""
    serialised_json = ""

    if dataset_obj is None:
      logging.error("DigitisationPipeline.E_writeOutputJson received None dataset_obj.")
      return False

    if os.path.normcase(os.path.abspath(self.output_abs_path)) == os.path.normcase(os.path.abspath(self.reference_abs_path)):
      logging.error("DigitisationPipeline.E_writeOutputJson blocked attempt to overwrite root mcevedy_data.json.")
      return False

    out_dir = os.path.dirname(self.output_abs_path)
    if len(out_dir) > 0:
      os.makedirs(out_dir, exist_ok=True)

    serialised_json = json.dumps(dataset_obj, indent=2) + "\n"
    with open(self.output_abs_path, "w", encoding="utf-8") as out_file:
      out_file.write(serialised_json)

    logging.info("Wrote digitised dataset (%d regions) to %s", len(dataset_obj), self.output_abs_path)

    #Return statement
    return True


  def F_runPipeline (self):
    """
    @brief Step F: Executes the full A-to-E digitisation pipeline from configuration loading to JSON output.
    @return True if the entire pipeline succeeded; False otherwise.
    """
    #Declare local instance variables
    dataset_obj = None
    elapsed_sec = 0.0
    extracted_obj = None
    is_loaded = False
    is_valid = False
    is_written = False
    t_start = time.time()

    logging.info("Starting McEvedy & Jones CV digitisation pipeline...")
    is_loaded = self.A_loadConfigurations()
    if not is_loaded:
      return False

    extracted_obj = self.B_processChartPages()
    if extracted_obj is None:
      return False

    dataset_obj = self.C_assembleDataset(extracted_obj)
    if dataset_obj is None:
      return False

    is_valid = self.D_validateDataset(dataset_obj)
    if not is_valid:
      return False

    is_written = self.E_writeOutputJson(dataset_obj)
    if not is_written:
      return False

    elapsed_sec = time.time() - t_start
    logging.info("Digitisation pipeline completed cleanly in %.1fs.", elapsed_sec)

    #Return statement
    return True


  def parseJson5Text (self, arg0_raw_text):
    """
    @brief Parses a JSON5 string containing '//' comments and optional trailing commas into a Python dictionary.
    @param arg0_raw_text Raw UTF-8 text from a .json5 configuration file.
    @return Parsed dictionary (or empty dictionary on parse error).
    """
    #Convert from parameters
    raw_text = str(arg0_raw_text) if arg0_raw_text is not None else ""

    #Declare local instance variables
    all_clean_lines = []
    all_lines = raw_text.splitlines()
    clean_text = ""
    parsed_obj = {}

    for i in range(len(all_lines)):
      local_line = re.sub(r"(^|\s)//.*$", "", all_lines[i])
      all_clean_lines.append(local_line)

    clean_text = "\n".join(all_clean_lines)
    clean_text = re.sub(r",(\s*[}\]])", r"\1", clean_text)

    if len(clean_text.strip()) == 0:
      logging.error("DigitisationPipeline.parseJson5Text received empty configuration text.")
      return {}

    parsed_obj = json.loads(clean_text)

    #Return statement
    return parsed_obj

