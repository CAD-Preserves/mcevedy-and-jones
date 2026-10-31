import logging
import numpy as np
from scipy.interpolate import PchipInterpolator


class TableInterpolator:
  """
  @brief Interpolates and combines annual population series for tabular micro-islands, derived regions and split chart panels.
  """

  def __init__ (self, arg0_config_obj):
    """
    @brief Initialises the TableInterpolator with pipeline configuration settings.
    @param arg0_config_obj Dictionary containing parsed pipeline_config.json5 settings.
    """
    #Convert from parameters
    config_obj = arg0_config_obj if arg0_config_obj is not None else {}

    #Declare local instance variables
    output_cfg_obj = config_obj.get("output", {})
    x_cfg_obj = config_obj.get("x_axis", {})

    self.config_obj = config_obj
    self.end_year_cap = int(x_cfg_obj.get("end_year_cap", 1975))
    self.rounding_decimals_chart = int(output_cfg_obj.get("rounding_decimals_chart", 4))
    self.rounding_decimals_micro = int(output_cfg_obj.get("rounding_decimals_micro", 5))


  def A_interpolateTableAnnual (self, arg0_anchors_obj, arg1_options=None):
    """
    @brief Generates shape-preserving annual population estimates from printed McEvedy & Jones island tables.
    @param arg0_anchors_obj Mapping of year strings/integers to population figures in millions.
    @param arg1_options Optional dictionary supporting 'end_year' and 'rounding_decimals'.
    @return Dictionary mapping every integer year string up to 1975 to interpolated millions.
    """
    #Convert from parameters
    anchors_obj = arg0_anchors_obj
    options = arg1_options if arg1_options is not None else {}

    #Initialise options
    if "end_year" not in options:
      options["end_year"] = self.end_year_cap
    if "rounding_decimals" not in options:
      options["rounding_decimals"] = self.rounding_decimals_micro

    #Declare local instance variables
    all_year_pairs = []
    annual_pop_obj = {}
    end_year = int(options["end_year"])
    interp_fn = None
    rounding_decimals = int(options["rounding_decimals"])
    start_year = 0
    vals_array = None
    years_array = None

    if not isinstance(anchors_obj, dict) or len(anchors_obj) == 0:
      logging.error("TableInterpolator.A_interpolateTableAnnual received empty anchors_obj.")
      return {}

    for i in anchors_obj:
      local_yr = int(i)
      if local_yr <= end_year:
        all_year_pairs.append((local_yr, float(anchors_obj[i])))
    all_year_pairs.sort(key=lambda pair: pair[0])

    if len(all_year_pairs) == 0:
      logging.error("TableInterpolator.A_interpolateTableAnnual found no anchors <= end_year.")
      return {}

    start_year = all_year_pairs[0][0]
    years_array = np.array([pair[0] for pair in all_year_pairs], dtype=np.float64)
    vals_array = np.array([pair[1] for pair in all_year_pairs], dtype=np.float64)

    if len(all_year_pairs) >= 2:
      interp_fn = PchipInterpolator(years_array, vals_array, extrapolate=True)
      for x in range(start_year, end_year + 1):
        local_val = max(0.0, float(interp_fn(float(x))))
        annual_pop_obj[str(x)] = round(local_val, rounding_decimals)
    else:
      for x in range(start_year, end_year + 1):
        annual_pop_obj[str(x)] = round(max(0.0, float(vals_array[0])), rounding_decimals)

    #Return statement
    return annual_pop_obj


  def B_computeDerivedSeries (self, arg0_base_pop_obj, arg1_multiplier, arg2_options=None):
    """
    @brief Scales a base region annual population series by a constant factor (e.g. Bhutan & Sikkim = 0.1 * Nepal).
    @param arg0_base_pop_obj Annual population dictionary of the reference region.
    @param arg1_multiplier Multiplicative scalar factor.
    @param arg2_options Optional dictionary supporting 'rounding_decimals'.
    @return Scaled annual population dictionary with identical year keys.
    """
    #Convert from parameters
    base_pop_obj = arg0_base_pop_obj
    multiplier = float(arg1_multiplier)
    options = arg2_options if arg2_options is not None else {}

    #Initialise options
    if "rounding_decimals" not in options:
      options["rounding_decimals"] = self.rounding_decimals_micro

    #Declare local instance variables
    derived_pop_obj = {}
    rounding_decimals = int(options["rounding_decimals"])

    if not isinstance(base_pop_obj, dict) or len(base_pop_obj) == 0:
      logging.error("TableInterpolator.B_computeDerivedSeries received invalid base_pop_obj.")
      return {}

    for i in base_pop_obj:
      local_scaled = max(0.0, float(base_pop_obj[i])*multiplier)
      derived_pop_obj[str(i)] = round(local_scaled, rounding_decimals)

    #Return statement
    return derived_pop_obj


  def C_combineSplitPanelSeries (self, arg0_combined_pop_obj, arg1_sub_pop_obj, arg2_ot_sub_pop_obj, arg3_options=None):
    """
    @brief Merges a pre-split combined regional curve (e.g. Palestine & Jordan <= 1500) with post-split sub-panel curves.
    @param arg0_combined_pop_obj Extracted annual series for the combined chart up to year_split.
    @param arg1_sub_pop_obj Extracted annual series for this region's sub-panel from year_split to 1975.
    @param arg2_ot_sub_pop_obj Extracted annual series for the counterpart sub-panel from year_split to 1975.
    @param arg3_options Options dict containing 'year_start', 'year_split', 'share_blend_start_year', 'default_early_share'.
    @return Continuous annual population dictionary from year_start to 1975.
    """
    #Convert from parameters
    combined_pop_obj = arg0_combined_pop_obj
    sub_pop_obj = arg1_sub_pop_obj
    ot_sub_pop_obj = arg2_ot_sub_pop_obj
    options = arg3_options if arg3_options is not None else {}

    #Initialise options
    if "default_early_share" not in options:
      options["default_early_share"] = 0.5
    if "end_year" not in options:
      options["end_year"] = self.end_year_cap
    if "rounding_decimals" not in options:
      options["rounding_decimals"] = self.rounding_decimals_chart
    if "share_blend_start_year" not in options:
      options["share_blend_start_year"] = 1400
    if "year_split" not in options:
      options["year_split"] = 1500
    if "year_start" not in options:
      options["year_start"] = -200

    #Declare local instance variables
    blend_span = 1.0
    blend_start = int(options["share_blend_start_year"])
    early_share = float(options["default_early_share"])
    end_year = int(options["end_year"])
    merged_pop_obj = {}
    ot_split_val = 0.0
    rounding_decimals = int(options["rounding_decimals"])
    split_share = early_share
    split_sum = 0.0
    sub_split_val = 0.0
    year_split = int(options["year_split"])
    year_start = int(options["year_start"])

    if not isinstance(combined_pop_obj, dict) or not isinstance(sub_pop_obj, dict):
      logging.error("TableInterpolator.C_combineSplitPanelSeries received invalid series dictionaries.")
      return {}

    sub_split_val = float(sub_pop_obj.get(str(year_split), 0.0))
    ot_split_val = float(ot_sub_pop_obj.get(str(year_split), 0.0)) if isinstance(ot_sub_pop_obj, dict) else 0.0
    split_sum = sub_split_val + ot_split_val
    if split_sum > 1e-6:
      split_share = max(0.05, min(0.95, sub_split_val/split_sum))

    blend_span = max(1.0, float(year_split - blend_start))
    for i in range(year_start, end_year + 1):
      if i < year_split:
        local_comb = float(combined_pop_obj.get(str(i), 0.0))
        if i <= blend_start:
          local_w = early_share
        else:
          local_t = max(0.0, min(1.0, float(i - blend_start)/blend_span))
          local_w = early_share + local_t*(split_share - early_share)
        merged_pop_obj[str(i)] = round(max(0.0, local_comb*local_w), rounding_decimals)
      else:
        local_sub = float(sub_pop_obj.get(str(i), 0.0))
        merged_pop_obj[str(i)] = round(max(0.0, local_sub), rounding_decimals)

    #Return statement
    return merged_pop_obj

