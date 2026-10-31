import cv2
import logging
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import PchipInterpolator


class CurveExtractor:
  """
  @brief Extracts sub-pixel population curves and annual time series from calibrated McEvedy & Jones chart panels.
  """

  def __init__ (self, arg0_config_obj, arg1_calibrator_obj):
    """
    @brief Initialises CurveExtractor with morphological, glyph-filtering and Viterbi parameters.
    @param arg0_config_obj Parsed pipeline_config.json5 dictionary.
    @param arg1_calibrator_obj Instance of GridCalibrator for coordinate transformations.
    """
    #Convert from parameters
    config_obj = arg0_config_obj if arg0_config_obj is not None else {}
    calibrator_obj = arg1_calibrator_obj

    #Declare local instance variables
    ext_cfg_obj = config_obj.get("extraction", {})
    out_cfg_obj = config_obj.get("output", {})
    x_cfg_obj = config_obj.get("x_axis", {})

    self.air_ink_threshold = float(ext_cfg_obj.get("air_ink_threshold", 75.0))
    self.air_ink_vgrid = float(ext_cfg_obj.get("air_ink_threshold_vgrid", 155.0))
    self.baseline_subtract_scale = float(ext_cfg_obj.get("baseline_subtract_scale", 1.25))
    self.bin_threshold = float(ext_cfg_obj.get("binary_component_threshold", 118.0))
    self.calibrator_obj = calibrator_obj
    self.curve_min_h = int(ext_cfg_obj.get("curve_min_height_px", 35))
    self.curve_min_w = int(ext_cfg_obj.get("curve_min_width_px", 25))
    self.dot_bonus_od = float(ext_cfg_obj.get("dot_bonus_od", 350.0))
    self.dot_flank_min_od = float(ext_cfg_obj.get("dot_flank_min_od", 55.0))
    self.dot_min_area = int(ext_cfg_obj.get("dot_min_area_px", 5))
    self.dot_rescue_r = float(ext_cfg_obj.get("dot_rescue_radius_px", 6.5))
    self.end_year_cap = int(x_cfg_obj.get("end_year_cap", 1975))
    self.endpoint_search_px = int(ext_cfg_obj.get("endpoint_1975_search_px", 26))
    self.glyph_er_max_h = int(ext_cfg_obj.get("glyph_eroded_max_height_px", 27))
    self.glyph_er_max_w = int(ext_cfg_obj.get("glyph_eroded_max_width_px", 52))
    self.glyph_max_h = int(ext_cfg_obj.get("glyph_max_height_px", 46))
    self.glyph_max_w = int(ext_cfg_obj.get("glyph_max_width_px", 62))
    self.grid_percentile = float(ext_cfg_obj.get("grid_percentile", 45.0))
    self.grid_percentile_vert = float(ext_cfg_obj.get("grid_percentile_vert", 65.0))
    self.grid_subtract_scale = float(ext_cfg_obj.get("grid_subtract_scale", 1.15))
    self.grid_subtract_vert = float(ext_cfg_obj.get("grid_subtract_scale_vert", 1.25))
    self.knockout_half_w = float(ext_cfg_obj.get("knockout_half_width_px", 16.0))
    self.map_contrast_min = float(ext_cfg_obj.get("map_grid_contrast_min", 35.0))
    self.map_horiz_min_od = float(ext_cfg_obj.get("map_border_horiz_min_od", 155.0))
    self.map_horiz_p20_od = float(ext_cfg_obj.get("map_border_horiz_p20_min_od", 115.0))
    self.minkowski_max_exc = float(ext_cfg_obj.get("minkowski_max_excursion_px", 180.0))
    self.minkowski_min_exc = float(ext_cfg_obj.get("minkowski_min_excursion_px", 6.0))
    self.minkowski_min_run = int(ext_cfg_obj.get("minkowski_min_run_px", 15))
    self.pen_radius_px = float(ext_cfg_obj.get("pen_radius_px", 2.4))
    self.rounding_decimals = int(out_cfg_obj.get("rounding_decimals_chart", 4))
    self.slope_chord_base = float(ext_cfg_obj.get("slope_chord_base_px", 6.0))
    self.slope_chord_factor = float(ext_cfg_obj.get("slope_chord_factor", 3.0))
    self.solid_margin_od = float(ext_cfg_obj.get("solid_ink_margin_od", 48.0))
    self.solid_min_od = float(ext_cfg_obj.get("solid_ink_min_od", 175.0))
    self.thick_bridge_od = float(ext_cfg_obj.get("thick_bridge_min_od", 80.0))
    self.thick_cols_array = list(x_cfg_obj.get("thick_vertical_cols", [0, 7, 12, 22]))
    self.v_air_beta = float(ext_cfg_obj.get("viterbi_air_beta", 9.0))
    self.v_air_beta_ko = float(ext_cfg_obj.get("viterbi_air_beta_knockout", 3.0))
    self.v_air_gamma = float(ext_cfg_obj.get("viterbi_air_gamma", 1.6))
    self.v_air_gamma_ko = float(ext_cfg_obj.get("viterbi_air_gamma_knockout", 0.25))
    self.v_air_self_pen = float(ext_cfg_obj.get("viterbi_air_self_penalty", -50.0))
    self.v_max_step = int(ext_cfg_obj.get("viterbi_max_step_px", 95))
    self.v_step_alpha = float(ext_cfg_obj.get("viterbi_step_alpha", 0.07))


  def A_maskTitleAndInsetBoxes (self, arg0_od_img, arg1_grid_obj, arg2_panel_cfg_obj):
    """
    @brief Detects and masks the panel's title/locator-map frame and any configured right-hand inset boxes.
    @param arg0_od_img Float32 optical density image.
    @param arg1_grid_obj Calibrated 2D grid dictionary.
    @param arg2_panel_cfg_obj Panel configuration dictionary ('r_top', 'r_bot', optional 'c_left' and 'inset_masks').
    @return Boolean mask of shape od_img.shape marking non-chart exclusion regions.
    """
    #Convert from parameters
    od_img = arg0_od_img
    grid_obj = arg1_grid_obj
    panel_cfg_obj = arg2_panel_cfg_obj if arg2_panel_cfg_obj is not None else {}

    #Declare local instance variables
    all_insets = panel_cfg_obj.get("inset_masks", [])
    band_h = 0
    box_mask = None
    bx1 = 0
    bx2 = 0
    c_left = int(panel_cfg_obj.get("c_left", 0))
    cal = self.calibrator_obj
    h = 0
    has_left_inset = False
    inner_cols_array = []
    mid_cols_array = []
    r_bot = float(panel_cfg_obj.get("r_bot", 35))
    r_top = float(panel_cfg_obj.get("r_top", 0))
    right_wall_array = None
    w = 0
    x0 = 0
    x_search_max = 0
    x_search_min = 0
    xr_found = None
    y_bot_at_xr = None
    y_last_rule = None
    y_scan_max = 0
    yb0 = 0
    yb_right = 0
    yt0 = 0
    yt_right = 0

    if od_img is None or grid_obj is None:
      logging.error("CurveExtractor.A_maskTitleAndInsetBoxes received None input.")
      return None

    h, w = od_img.shape
    box_mask = np.zeros((h, w), dtype=bool)
    for i in range(len(all_insets)):
      if int(all_insets[i].get("c1", 99)) == 0:
        has_left_inset = True

    if c_left == 0 and not has_left_inset:
      x0 = int(round(grid_obj["x_cols"][0]))
      yt0 = int(round(cal.F_rowToPixelY(grid_obj, r_top, x0)))
      yb0 = int(round(cal.F_rowToPixelY(grid_obj, r_bot, x0)))
      band_h = min(44, yb0 - yt0 - 14)

      if band_h >= 18:
        x_search_min = int(round(grid_obj["x_cols"][3] + 15))
        x_search_max = min(w - 6, int(round(grid_obj["x_cols"][20] + 25)))
        for i in range(x_search_min, x_search_max):
          local_yt = int(round(cal.F_rowToPixelY(grid_obj, r_top, i)))
          local_v_str = od_img[local_yt + 8:local_yt + band_h, i - 1:i + 2].max(axis=1).mean()
          local_top_touch = od_img[max(0, local_yt - 4):local_yt + 12, i - 1:i + 2].max()
          if local_v_str > 135.0 and local_top_touch > 115.0:
            local_top_xs = np.linspace(x0 + 12, max(x0 + 18, i - 10), 20).astype(int)
            local_top_ys = np.round([cal.F_rowToPixelY(grid_obj, r_top, x) for x in local_top_xs]).astype(int)
            local_top_vals = [od_img[max(0, local_top_ys[x] - 4):min(h, local_top_ys[x] + 10), local_top_xs[x]].max() for x in range(len(local_top_xs))]
            if float(np.percentile(local_top_vals, 20.0)) > 65.0:
              local_prof = od_img[local_yt + 8:local_yt + band_h, i - 3:i + 5].mean(axis=0)
              xr_found = i - 3 + int(np.argmax(local_prof))
              break

      if xr_found is not None:
        yt_right = int(round(cal.F_rowToPixelY(grid_obj, r_top, xr_found)))
        yb_right = int(round(cal.F_rowToPixelY(grid_obj, r_bot, xr_found)))
        y_scan_max = yb_right - 12
        if y_scan_max > yt_right + 28:
          right_wall_array = cv2.dilate(
            od_img[yt_right:y_scan_max, xr_found - 3:xr_found + 4].max(axis=1, keepdims=True),
            np.ones((5, 1), dtype=np.uint8)
          )[:, 0]
          for i in range(1, 18):
            if grid_obj["x_cols"][i] < xr_found - 18:
              inner_cols_array.append(int(round(grid_obj["x_cols"][i])))
          for i in range(1, 17):
            if grid_obj["x_cols"][i + 1] < xr_found - 12:
              mid_cols_array.append(int(round(0.5*(grid_obj["x_cols"][i] + grid_obj["x_cols"][i + 1]))))

          for i in range(yt_right + 30, y_scan_max):
            if right_wall_array[i - yt_right] < 95.0:
              break
            local_xs = np.linspace(x0 + 14, xr_found - 12, 29).astype(int)
            local_ys = np.round([i + (cal.F_rowToPixelY(grid_obj, r_top, x) - yt_right) for x in local_xs]).astype(int)
            local_h_vals = [od_img[max(0, local_ys[x] - 3):min(h, local_ys[x] + 4), local_xs[x]].max() for x in range(len(local_xs))]
            local_h_med = float(np.median(local_h_vals))
            local_h_p20 = float(np.percentile(local_h_vals, 20.0))
            if local_h_med > self.map_horiz_min_od and local_h_p20 > self.map_horiz_p20_od:
              y_last_rule = i
              if len(inner_cols_array) >= 3 and len(mid_cols_array) >= 2:
                local_p1 = min(h - 2, max(i, int(np.max(local_ys))) + 10)
                local_p2 = min(yb_right - 6, local_p1 + 32)
                if local_p2 > local_p1 + 10:
                  local_col_ons = [od_img[local_p1:local_p2, x - 2:x + 3].max(axis=1).mean() for x in inner_cols_array[:5]]
                  local_col_offs = [float(np.median(od_img[local_p1:local_p2, x - 3:x + 4])) for x in mid_cols_array[:5]]
                  if float(np.min(local_col_ons)) > 62.0 and float(np.median(local_col_offs)) < 34.0 and (float(np.mean(local_col_ons)) - float(np.mean(local_col_offs))) > self.map_contrast_min:
                    y_bot_at_xr = i
                    break

          if y_bot_at_xr is None:
            y_bot_at_xr = y_last_rule

        if y_bot_at_xr is not None:
          bx1 = max(0, x0 - 8)
          bx2 = min(w, xr_found + 8)
          for i in range(bx1, bx2):
            local_tilt = cal.F_rowToPixelY(grid_obj, r_top, i) - yt_right
            local_ym1 = max(0, int(round(yt_right + local_tilt)) - 8)
            local_ym2 = min(h, int(round(y_bot_at_xr + local_tilt)) + 12)
            box_mask[local_ym1:local_ym2, i] = True

    for i in range(len(all_insets)):
      local_inset = all_insets[i]
      local_ix1 = max(0, int(round(grid_obj["x_cols"][int(local_inset["c1"])])) - 6)
      local_ix2 = min(w, int(round(grid_obj["x_cols"][int(local_inset["c2"])])) + 6)
      for x in range(local_ix1, local_ix2):
        local_iy1 = max(0, int(round(cal.F_rowToPixelY(grid_obj, float(local_inset["r1"]), x))) - 6)
        local_iy2 = min(h, int(round(cal.F_rowToPixelY(grid_obj, float(local_inset["r2"]), x))) + 8)
        box_mask[local_iy1:local_iy2, x] = True

    #Return statement
    return box_mask


  def B_removeGridAndBorders (self, arg0_od_img, arg1_box_mask, arg2_grid_obj, arg3_panel_cfg_obj):
    """
    @brief Subtracts thin gridlines via max-composited templates, removes panel borders and clears bare thick vertical lines.
    @param arg0_od_img Float32 optical density image.
    @param arg1_box_mask Boolean exclusion mask from A_maskTitleAndInsetBoxes.
    @param arg2_grid_obj Calibrated 2D grid dictionary.
    @param arg3_panel_cfg_obj Panel configuration dictionary.
    @return Dictionary containing panel ROI crops ('roi_img', 'od_roi_img', 'gh_roi_img', 'rx0', 'ry0', 'thick_x_array').
    """
    #Convert from parameters
    od_img = arg0_od_img
    box_mask = arg1_box_mask
    grid_obj = arg2_grid_obj
    panel_cfg_obj = arg3_panel_cfg_obj if arg3_panel_cfg_obj is not None else {}

    #Declare local instance variables
    bare_75_array = None
    blur3_img = None
    bot_prof_array = None
    bot_strips_array = None
    c_left = int(panel_cfg_obj.get("c_left", 0))
    cal = self.calibrator_obj
    g_horiz_img = None
    g_total_sub_img = None
    g_vert_img = None
    h = 0
    h_only_sub_img = None
    h_profiles_array = []
    is_solid_mask = None
    l_flank_75_array = None
    r_bot = int(panel_cfg_obj.get("r_bot", 35))
    r_top = int(panel_cfg_obj.get("r_top", 0))
    ref_mask = None
    shift_map_obj = {}
    th_prof_array = None
    thick_set = None
    thick_x_array = []
    v_bridge_mask = None
    w = 0
    work_img = None
    x_left_panel = 0
    x_right_panel = 0
    x_start_px = 0.0
    xi75_abs = 0
    xref_array = None
    xs_all_array = None
    y_bot_px = 0
    y_top_px = 0
    yb1_75 = 0
    yb2_75 = 0
    year_start = int(panel_cfg_obj.get("year_start", 0))
    yref_array = None
    ys_bot_array = None
    ys_top_array = None

    if od_img is None or box_mask is None or grid_obj is None:
      logging.error("CurveExtractor.B_removeGridAndBorders received None input.")
      return None

    h, w = od_img.shape
    x_left_panel = max(0, int(round(grid_obj["x_cols"][c_left])) - 8)
    x_right_panel = min(w - 1, int(round(grid_obj["x_cols"][22])) + 8)
    y_top_px = max(0, int(round(min(cal.F_rowToPixelY(grid_obj, r_top, x_left_panel), cal.F_rowToPixelY(grid_obj, r_top, x_right_panel)))) - 8)
    y_bot_px = min(h - 1, int(round(max(cal.F_rowToPixelY(grid_obj, r_bot, x_left_panel), cal.F_rowToPixelY(grid_obj, r_bot, x_right_panel)))) + 8)

    work_img = od_img.copy()
    work_img[box_mask] = 0.0
    g_horiz_img = np.zeros_like(od_img)
    g_vert_img = np.zeros_like(od_img)
    xs_all_array = np.arange(x_left_panel, x_right_panel)

    for i in range(r_top + 1, r_bot):
      local_ys = np.round([cal.F_rowToPixelY(grid_obj, i, x) for x in xs_all_array]).astype(int)
      local_valid = ~box_mask[np.clip(local_ys, 0, h - 1), xs_all_array]
      if int(local_valid.sum()) > 40:
        local_xv = xs_all_array[local_valid]
        local_yv = local_ys[local_valid]
        local_meds = [float(np.median(od_img[np.clip(local_yv + x, 0, h - 1), local_xv])) for x in range(-3, 4)]
        local_shift = int(np.argmax(local_meds)) - 3 if max(local_meds) > 40.0 else 0
        local_yv = local_yv + local_shift
        local_strips = np.stack([od_img[np.clip(local_yv + x, 0, h - 1), local_xv] for x in range(-3, 4)], axis=0)
        h_profiles_array.append((i, local_shift, np.percentile(local_strips, self.grid_percentile, axis=1)))

    if len(h_profiles_array) > 0:
      th_prof_array = np.median(np.stack([item[2] for item in h_profiles_array], axis=0), axis=0)
      shift_map_obj = {item[0]: item[1] for item in h_profiles_array}
      for i in range(r_top + 1, r_bot):
        local_sh = shift_map_obj.get(i, 0)
        local_ys = np.round([cal.F_rowToPixelY(grid_obj, i, x) for x in xs_all_array]).astype(int) + local_sh
        for x in range(7):
          local_yy = np.clip(local_ys + (x - 3), 0, h - 1)
          g_horiz_img[local_yy, xs_all_array] = np.maximum(g_horiz_img[local_yy, xs_all_array], th_prof_array[x])

    thick_set = set(self.thick_cols_array) | {c_left}
    for i in range(c_left, 23):
      if i in thick_set:
        continue
      local_xi = int(round(grid_obj["x_cols"][i]))
      local_yb1 = max(0, int(round(cal.F_rowToPixelY(grid_obj, r_top, local_xi))) + 6)
      local_yb2 = min(h - 1, int(round(cal.F_rowToPixelY(grid_obj, r_bot, local_xi))) - 6)
      if local_yb2 > local_yb1 + 25:
        local_ys_v = np.arange(local_yb1, local_yb2)
        local_valid_v = ~box_mask[local_ys_v, local_xi]
        if int(local_valid_v.sum()) > 25:
          local_yv = local_ys_v[local_valid_v]
          local_v_strips = np.stack([od_img[local_yv, np.clip(local_xi + x, 0, w - 1)] for x in range(-3, 4)], axis=0)
          local_v_prof = np.percentile(local_v_strips, self.grid_percentile_vert, axis=1)
          for x in range(7):
            local_xx = np.clip(local_xi + (x - 3), 0, w - 1)
            local_va = max(0, local_yb1 - 8)
            local_vb = min(h - 1, local_yb2 + 8)
            g_vert_img[local_va:local_vb, local_xx] = np.maximum(g_vert_img[local_va:local_vb, local_xx], local_v_prof[x])

    g_total_sub_img = np.maximum(self.grid_subtract_scale*g_horiz_img, self.grid_subtract_vert*g_vert_img)
    blur3_img = cv2.blur(work_img, (3, 3))
    h_only_sub_img = np.maximum(0.0, work_img - self.grid_subtract_scale*g_horiz_img)
    v_bridge_mask = np.zeros_like(box_mask)
    for i in range(c_left, 23):
      if i in thick_set:
        continue
      local_xi = int(round(grid_obj["x_cols"][i]))
      if 6 <= local_xi < w - 6:
        local_lf = cv2.dilate(h_only_sub_img[:, local_xi - 6:local_xi - 3].max(axis=1, keepdims=True), np.ones((35, 1), dtype=np.uint8))[:, 0]
        local_rf = cv2.dilate(h_only_sub_img[:, local_xi + 4:local_xi + 7].max(axis=1, keepdims=True), np.ones((35, 1), dtype=np.uint8))[:, 0]
        v_bridge_mask[:, max(0, local_xi - 3):min(w, local_xi + 4)] |= ((local_lf > 85.0) & (local_rf > 85.0))[:, None]

    is_solid_mask = (
      ((blur3_img > np.maximum(g_horiz_img, g_vert_img) + 42.0) & (work_img > self.solid_min_od)) |
      (v_bridge_mask & (h_only_sub_img > 85.0))
    )
    work_img = np.where(is_solid_mask, work_img, np.maximum(0.0, work_img - g_total_sub_img))
    work_img = np.where(is_solid_mask & v_bridge_mask, h_only_sub_img, work_img)
    work_img[box_mask] = 0.0

    ys_top_array = np.round([cal.F_rowToPixelY(grid_obj, r_top, x) for x in xs_all_array]).astype(int)
    for i in range(len(xs_all_array)):
      work_img[y_top_px:min(h, max(0, ys_top_array[i] + 9)), xs_all_array[i]] = 0.0

    ys_bot_array = np.round([cal.F_rowToPixelY(grid_obj, r_bot, x) for x in xs_all_array]).astype(int)
    if r_bot < 35:
      local_bmeds = [float(np.median(od_img[np.clip(ys_bot_array + i, 0, h - 1), xs_all_array])) for i in range(-4, 5)]
      if max(local_bmeds) > 80.0:
        ys_bot_array = ys_bot_array + (int(np.argmax(local_bmeds)) - 4)
    x_start_px = cal.D_yearToPixelX(grid_obj, year_start)
    ref_mask = xs_all_array <= x_start_px - 10
    if int(ref_mask.sum()) < 25:
      ref_mask = (xs_all_array >= grid_obj["x_cols"][20] + 15) | (xs_all_array <= x_start_px - 10)
    if int(ref_mask.sum()) < 25:
      ref_mask = xs_all_array >= grid_obj["x_cols"][19] + 15
    xref_array = xs_all_array[ref_mask]
    yref_array = ys_bot_array[ref_mask]
    bot_strips_array = np.stack([od_img[np.clip(yref_array + i, 0, h - 1), xref_array] for i in range(-6, 7)], axis=0)
    bot_prof_array = np.percentile(bot_strips_array, 25.0, axis=1)
    for i in range(13):
      local_d = i - 6
      local_yy = np.clip(ys_bot_array + local_d, 0, h - 1)
      if r_bot == 35 and local_d < -1:
        continue
      if local_d < -2 and float(bot_prof_array[i]) < 95.0:
        continue
      if local_d == -2 and float(bot_prof_array[i]) < 65.0:
        continue
      local_scale = 1.05 if (local_d == -1 and r_bot == 35) else self.baseline_subtract_scale
      work_img[local_yy, xs_all_array] = np.maximum(0.0, work_img[local_yy, xs_all_array] - local_scale*bot_prof_array[i])
    for i in range(len(xs_all_array)):
      local_yb_cut = ys_bot_array[i] if r_bot == 35 else ys_bot_array[i] - 2
      work_img[min(h, max(0, local_yb_cut)):y_bot_px + 1, xs_all_array[i]] = 0.0

    xi75_abs = int(round(grid_obj["x_1975"]))
    work_img[y_top_px:y_bot_px + 1, min(x_right_panel, xi75_abs + 3):x_right_panel] = 0.0
    yb1_75 = max(0, int(round(cal.F_rowToPixelY(grid_obj, r_top, xi75_abs))) + 4)
    yb2_75 = min(h - 1, int(round(cal.F_rowToPixelY(grid_obj, r_bot, xi75_abs))))
    if yb2_75 > yb1_75 + 4 and 6 <= xi75_abs < w - 4:
      l_flank_75_array = cv2.dilate(work_img[yb1_75:yb2_75 + 1, xi75_abs - 7:xi75_abs - 3].max(axis=1, keepdims=True), np.ones((19, 1), dtype=np.uint8))[:, 0]
      bare_75_array = np.where(l_flank_75_array <= 85.0)[0] + yb1_75
      work_img[bare_75_array[:, None], np.arange(max(0, xi75_abs - 2), min(w, xi75_abs + 4))[None, :]] = 0.0

    for i in sorted(thick_set):
      if c_left <= i <= 22:
        thick_x_array.append(float(grid_obj["x_cols"][i]))

    for i in range(len(thick_x_array)):
      local_xc = thick_x_array[i]
      local_xi = int(round(local_xc))
      local_yb1 = max(0, int(round(cal.F_rowToPixelY(grid_obj, r_top, local_xi))) + 4)
      local_yb2 = min(h - 1, int(round(cal.F_rowToPixelY(grid_obj, r_bot, local_xi))) - 1)
      if local_yb2 > local_yb1 + 4 and 6 <= local_xi < w - 6:
        local_l_flank = cv2.dilate(work_img[local_yb1:local_yb2 + 1, local_xi - 6:local_xi - 3].max(axis=1, keepdims=True), np.ones((13, 1), dtype=np.uint8))[:, 0]
        local_r_flank = cv2.dilate(work_img[local_yb1:local_yb2 + 1, local_xi + 4:local_xi + 7].max(axis=1, keepdims=True), np.ones((13, 1), dtype=np.uint8))[:, 0]
        if abs(local_xc - x_start_px) < 4.5:
          local_l_dot = np.maximum(0.0, od_img[local_yb1:local_yb2 + 1, local_xi - 4:local_xi - 2].max(axis=1) - g_horiz_img[local_yb1:local_yb2 + 1, local_xi - 4:local_xi - 2].max(axis=1))
          local_bridge = (local_r_flank > 85.0) & ((local_l_dot > 65.0) | (local_r_flank > 130.0))
        else:
          local_bridge = (local_l_flank > self.thick_bridge_od) & (local_r_flank > self.thick_bridge_od)
        local_bare = np.where(~local_bridge)[0] + local_yb1
        work_img[local_bare[:, None], np.arange(max(0, local_xi - 3), min(w, local_xi + 4))[None, :]] = 0.0

    for i in range(c_left + 1, 22):
      if i in thick_set or grid_obj["x_cols"][i] <= x_start_px + 6.0:
        continue
      local_xi = int(round(grid_obj["x_cols"][i]))
      local_yb2 = min(h - 1, int(round(cal.F_rowToPixelY(grid_obj, r_bot, local_xi))))
      local_yb1 = max(0, int(round(cal.F_rowToPixelY(grid_obj, r_top, local_xi))) + 4, local_yb2 - 28)
      if local_yb2 > local_yb1 + 4 and 6 <= local_xi < w - 6:
        local_lf_t = cv2.dilate(work_img[local_yb1:local_yb2 + 1, local_xi - 6:local_xi - 3].max(axis=1, keepdims=True), np.ones((5, 1), dtype=np.uint8))[:, 0]
        local_rf_t = cv2.dilate(work_img[local_yb1:local_yb2 + 1, local_xi + 4:local_xi + 7].max(axis=1, keepdims=True), np.ones((5, 1), dtype=np.uint8))[:, 0]
        local_bare_t = np.where((local_lf_t <= 60.0) & (local_rf_t <= 60.0))[0] + local_yb1
        work_img[local_bare_t[:, None], np.arange(max(0, local_xi - 3), min(w, local_xi + 4))[None, :]] = 0.0

    thick_x_array.append(float(grid_obj["x_1975"]))

    #Return statement
    return {
      "gh_roi_img": g_horiz_img[y_top_px:y_bot_px + 1, x_left_panel:x_right_panel], "od_roi_img": od_img[y_top_px:y_bot_px + 1, x_left_panel:x_right_panel],
      "roi_img": work_img[y_top_px:y_bot_px + 1, x_left_panel:x_right_panel].copy(), "rx0": x_left_panel, "ry0": y_top_px,
      "thick_x_array": thick_x_array, "x_start_px": x_start_px
    }


  def C_isTextGlyphComponent (self, arg0_sub_mask, arg1_touch_baseline):
    """
    @brief Determines whether a binary connected component represents printed numerals/labels rather than a curve ribbon.
    @param arg0_sub_mask Boolean 2D bounding-box mask of the connected component.
    @param arg1_touch_baseline Boolean indicating whether the component sits adjacent to the panel baseline.
    @return True if the component is a text label/glyph; False if it contains a curve segment.
    """
    #Convert from parameters
    sub_mask = arg0_sub_mask
    touch_baseline = bool(arg1_touch_baseline)

    #Declare local instance variables
    ch = 0
    col_p88 = 0.0
    col_runs_array = None
    col_sum_array = None
    cw = 0
    er_mask = None
    has_ribbon = False
    l_sub = None
    n_sub = 0
    row_p88 = 0.0
    row_runs_array = None
    row_sum_array = None
    st_sub = None

    if sub_mask is None or sub_mask.ndim != 2:
      return False

    ch, cw = sub_mask.shape
    er_mask = cv2.erode(sub_mask.astype(np.uint8), np.ones((2, 2), dtype=np.uint8))
    if not touch_baseline and int(er_mask.sum()) < int(0.22*float(sub_mask.sum())):
      return True
    if 40 <= cw <= 240 and ch <= 45 and float(np.median(sub_mask.sum(axis=0))) <= 2.0:
      return True
    if cw >= self.glyph_max_w or ch >= self.glyph_max_h:
      return False
    if touch_baseline and ch <= 16:
      return False
    if not touch_baseline and ch <= 4 and int(sub_mask.sum()) < int(2.3*cw):
      return True

    if ch <= 42 and cw <= 60:
      col_sum_array = sub_mask.sum(axis=0)
      row_sum_array = sub_mask.sum(axis=1)
      col_p88 = float(np.percentile(col_sum_array, 88.0))
      row_p88 = float(np.percentile(row_sum_array, 88.0))
      col_runs_array = (sub_mask[1:, :] > sub_mask[:-1, :]).sum(axis=0) + sub_mask[0, :].astype(np.int32)
      row_runs_array = (sub_mask[:, 1:] > sub_mask[:, :-1]).sum(axis=1) + sub_mask[:, 0].astype(np.int32)
      if ch <= 33 and int((col_runs_array >= 2).sum()) >= 4:
        return True
      if min(col_p88, row_p88) > 13.5 and int((col_runs_array >= 2).sum()) >= 5:
        return True
      if int((col_runs_array >= 2).sum()) >= 7 and int((row_runs_array >= 2).sum()) >= 7:
        return True

    if not touch_baseline:
      if int(er_mask.sum()) == 0:
        return True
      n_sub, l_sub, st_sub, _ = cv2.connectedComponentsWithStats(er_mask, connectivity=8)
      for i in range(1, n_sub):
        local_sx, local_sy, local_sw, local_sh, local_area = st_sub[i]
        if local_area < 6:
          continue
        if local_sh > self.glyph_er_max_h or local_sw > self.glyph_er_max_w:
          has_ribbon = True
          break
        if (local_sw >= 30 or (local_sw >= 18 and local_sh >= 18 and float(np.hypot(local_sw, local_sh)) >= 28.5)) and local_area >= int(2.2*local_sw):
          local_piece = (l_sub[local_sy:local_sy + local_sh, local_sx:local_sx + local_sw] == i)
          local_p_sum = local_piece.sum(axis=0)
          local_p_runs = (local_piece[1:, :] > local_piece[:-1, :]).sum(axis=0) + local_piece[0, :].astype(np.int32)
          local_p_max = 13.5 if local_sh >= 18 else 9.5
          if int((local_p_runs >= 2).sum()) <= 2 and float(np.percentile(local_p_sum, 88.0)) <= local_p_max:
            has_ribbon = True
            break
      if not has_ribbon:
        return True

    #Return statement
    return False


  def D_filterGlyphComponents (self, arg0_clean_obj, arg1_grid_obj, arg2_panel_cfg_obj):
    """
    @brief Filters out printed milestone labels and rescues isolated bracketed milestone dots at 1875 and 1925.
    @param arg0_clean_obj Dictionary returned by B_removeGridAndBorders.
    @param arg1_grid_obj Calibrated 2D grid dictionary.
    @param arg2_panel_cfg_obj Panel configuration dictionary.
    @return Dictionary containing 'curve_roi_img', 'bonus_roi_img', 'x_1875', 'x_1925', 'x_start_roi', 'x_1975_roi'.
    """
    #Convert from parameters
    clean_obj = arg0_clean_obj
    grid_obj = arg1_grid_obj
    panel_cfg_obj = arg2_panel_cfg_obj if arg2_panel_cfg_obj is not None else {}

    #Declare local instance variables
    all_digit_boxes = []
    all_small_pts = []
    bin_mask = None
    bonus_roi_img = None
    cal = self.calibrator_obj
    centroids_array = None
    curve_mask = None
    curve_roi_img = None
    dot_mask = None
    keep_curve_array = None
    keep_dot_array = None
    labels_img = None
    num_cc = 0
    r_bot = float(panel_cfg_obj.get("r_bot", 35))
    roi_img = None
    rx0 = 0
    ry0 = 0
    stats_array = None
    x_1875 = 0.0
    x_1925 = 0.0
    x_1975_roi = 0.0
    x_start_roi = 0.0

    if clean_obj is None or grid_obj is None:
      logging.error("CurveExtractor.D_filterGlyphComponents received None input.")
      return None

    roi_img = clean_obj["roi_img"]
    rx0 = clean_obj["rx0"]
    ry0 = clean_obj["ry0"]
    bin_mask = (roi_img > self.bin_threshold).astype(np.uint8)
    if r_bot < 35.0:
      for i in range(max(0, int(round(grid_obj["x_cols"][18] - rx0))), bin_mask.shape[1]):
        local_ybi = int(round(cal.F_rowToPixelY(grid_obj, r_bot, rx0 + i) - ry0))
        if int(bin_mask[max(0, local_ybi - 9):min(bin_mask.shape[0], local_ybi + 2), i].sum()) <= 2:
          bin_mask[max(0, local_ybi - 4):min(bin_mask.shape[0], local_ybi + 2), i] = 0
    num_cc, labels_img, stats_array, centroids_array = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)

    x_1875 = 0.5*(grid_obj["x_cols"][19] + grid_obj["x_cols"][20]) - rx0
    x_1925 = 0.5*(grid_obj["x_cols"][20] + grid_obj["x_cols"][21]) - rx0
    x_start_roi = clean_obj["x_start_px"] - rx0
    x_1975_roi = grid_obj["x_1975"] - rx0

    keep_curve_array = np.zeros(num_cc, dtype=bool)
    keep_dot_array = np.zeros(num_cc, dtype=bool)

    for i in range(1, num_cc):
      local_lx, local_ly, local_cw, local_ch, local_area = stats_array[i]
      local_y_bot = cal.F_rowToPixelY(grid_obj, r_bot, rx0 + local_lx + 0.5*local_cw) - ry0
      local_touch_base = (local_ly + local_ch >= local_y_bot - 9.0)
      if local_cw >= self.curve_min_w or local_ch >= self.curve_min_h or (local_cw >= 19 and local_ch >= 19 and float(np.hypot(local_cw, local_ch)) >= 31.0):
        local_sub = (labels_img[local_ly:local_ly + local_ch, local_lx:local_lx + local_cw] == i)
        if self.C_isTextGlyphComponent(local_sub, local_touch_base):
          all_digit_boxes.append((i, local_lx, local_ly, local_cw, local_ch))
          continue
        keep_curve_array[i] = True
      else:
        if 11 <= local_ch <= 44 and 3 <= local_cw <= 28 and local_area >= 12:
          all_digit_boxes.append((i, local_lx, local_ly, local_cw, local_ch))
        if 2 <= local_cw <= 16 and 2 <= local_ch <= 16 and 12 <= local_area <= 85:
          all_small_pts.append((i, float(centroids_array[i][0]), float(centroids_array[i][1])))

    for i in range(1, num_cc):
      if keep_curve_array[i]:
        continue
      local_lx, local_ly, local_cw, local_ch, local_area = stats_array[i]
      local_cx, local_cy = centroids_array[i]
      local_y_bot = cal.F_rowToPixelY(grid_obj, r_bot, rx0 + local_cx) - ry0
      if 2 <= local_cw <= 13 and 2 <= local_ch <= 13 and local_area >= self.dot_min_area and local_cy < local_y_bot - 4.5:
        local_near_bracket = (abs(local_cx - x_1875) <= self.dot_rescue_r) or (abs(local_cx - x_1925) <= self.dot_rescue_r)
        if local_near_bracket:
          local_has_l = False
          local_has_r = False
          local_is_dec = False
          for x in range(len(all_digit_boxes)):
            local_bi, local_dx0, local_dy0, local_dw0, local_dh0 = all_digit_boxes[x]
            if local_bi == i:
              continue
            if (-3.0 <= local_dx0 - local_cx <= 14.2) and (local_dy0 + 0.30*local_dh0 <= local_cy <= local_dy0 + local_dh0 + 5.0):
              local_is_dec = True
              break
            if (-3.0 <= local_dx0 - local_cx <= 16.0) and (local_dy0 + 0.15*local_dh0 <= local_cy <= local_dy0 + local_dh0 + 5.0):
              local_has_r = True
            if (-3.0 <= local_cx - (local_dx0 + local_dw0) <= 13.0) and (local_dy0 + 0.35*local_dh0 <= local_cy <= local_dy0 + local_dh0 + 5.0):
              local_has_l = True
            if (-2.0 <= local_cy - (local_dy0 + local_dh0) <= 12.0) and (local_dx0 + 0.20*local_dw0 <= local_cx <= local_dx0 + local_dw0 + 4.5):
              local_is_dec = True
              break
          if local_has_l and local_has_r:
            local_is_dec = True
          if not local_is_dec:
            for x in range(len(all_small_pts)):
              local_oi, local_ox, local_oy = all_small_pts[x]
              if local_oi != i and 4.0 <= float(np.hypot(local_cx - local_ox, local_cy - local_oy)) <= 18.5:
                local_is_dec = True
                break
          if not local_is_dec:
            keep_dot_array[i] = True

    curve_mask = cv2.dilate((keep_curve_array[labels_img]).astype(np.uint8), np.ones((5, 5), dtype=np.uint8)) > 0
    dot_mask = cv2.dilate((keep_dot_array[labels_img]).astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) > 0
    curve_roi_img = np.where(curve_mask | dot_mask, roi_img, 0.0)
    bonus_roi_img = np.where(dot_mask, np.float32(self.dot_bonus_od), np.float32(0.0))

    #Return statement
    return {
      "bonus_roi_img": bonus_roi_img, "curve_roi_img": curve_roi_img, "gh_roi_img": clean_obj["gh_roi_img"],
      "od_roi_img": clean_obj["od_roi_img"], "rx0": rx0, "ry0": ry0, "thick_x_array": clean_obj["thick_x_array"],
      "x_1875": x_1875, "x_1925": x_1925, "x_1975_roi": x_1975_roi, "x_cols": grid_obj["x_cols"],
      "x_start_px": clean_obj["x_start_px"], "x_start_roi": x_start_roi
    }


  def E_traceViterbiSeam (self, arg0_filtered_obj):
    """
    @brief Runs C-vectorised sliding-window dynamic programming with quadratic air-jump penalties to trace the curve seam.
    @param arg0_filtered_obj Dictionary returned by D_filterGlyphComponents.
    @return Dictionary containing 'seam_sub_array', 'sub_img', 'in_thick_band_array', 'xs_track_start' and 'ya_crop'.
    """
    #Convert from parameters
    filtered_obj = arg0_filtered_obj
    if filtered_obj is None:
      logging.error("CurveExtractor.E_traceViterbiSeam received None filtered_obj.")
      return None

    #Declare local instance variables
    abs_offsets_array = None
    air_csum_array = None
    air_thresh_array = None
    back_mat = None
    bonus_roi_img = filtered_obj["bonus_roi_img"]
    curve_roi_img = filtered_obj["curve_roi_img"]
    dp_mat = None
    emission_img = None
    end_em_array = None
    in_thick_band_array = None
    init_em_array = None
    is_air_mat = None
    nz_rows_array = None
    pair_ink_mat = None
    pair_x_abs_array = None
    raw_bonus_img = None
    raw_sub_img = None
    rx0 = filtered_obj["rx0"]
    s_win = 0
    seam_sub_array = None
    step_pen_array = None
    sub_bonus_img = None
    sub_img = None
    thick_x_array = filtered_obj["thick_x_array"]
    track_h = 0
    track_w = 0
    vgrid_dist_array = None
    w_offsets_array = None
    x_cols_array = filtered_obj["x_cols"]
    xs_track_end = 0
    xs_track_start = max(0, int(round(filtered_obj["x_start_roi"])))
    y_base_idx_array = None
    y_cur = 0
    ya_crop = 0
    yb_crop = 0

    xs_track_end = min(curve_roi_img.shape[1] - 1, max(xs_track_start + 2, int(round(filtered_obj["x_1975_roi"])) - 4))
    raw_sub_img = curve_roi_img[:, xs_track_start:xs_track_end + 1]
    raw_bonus_img = bonus_roi_img[:, xs_track_start:xs_track_end + 1]

    nz_rows_array = np.where(raw_sub_img.max(axis=1) > 65.0)[0]
    if len(nz_rows_array) >= 2:
      ya_crop = max(0, int(nz_rows_array[0]) - 10)
      yb_crop = min(raw_sub_img.shape[0] - 1, int(nz_rows_array[-1]) + 10)
    else:
      ya_crop = 0
      yb_crop = raw_sub_img.shape[0] - 1

    sub_img = raw_sub_img[ya_crop:yb_crop + 1, :]
    sub_bonus_img = raw_bonus_img[ya_crop:yb_crop + 1, :]
    track_h, track_w = sub_img.shape
    emission_img = cv2.GaussianBlur(sub_img, (1, 5), 0.0) + sub_bonus_img

    pair_ink_mat = np.maximum(sub_img[:, :-1], sub_img[:, 1:])
    pair_x_abs_array = float(rx0 + xs_track_start) + 0.5 + np.arange(max(1, track_w - 1), dtype=np.float64)
    vgrid_dist_array = np.min(np.abs(pair_x_abs_array[:, None] - x_cols_array[None, :]), axis=1)
    air_thresh_array = np.where(vgrid_dist_array <= 3.0, self.air_ink_vgrid, self.air_ink_threshold).astype(np.float32)
    is_air_mat = (pair_ink_mat < air_thresh_array[None, :]).astype(np.float32)
    air_csum_array = np.vstack([np.zeros((1, max(1, track_w - 1)), dtype=np.float32), np.cumsum(is_air_mat, axis=0)])

    s_win = max(1, min(self.v_max_step, track_h - 1))
    dp_mat = np.full((track_h, track_w), -1e12, dtype=np.float32)
    back_mat = np.zeros((track_h, track_w), dtype=np.int32)

    init_em_array = sub_img[:, :min(12, track_w)].max(axis=1)
    dp_mat[:, 0] = emission_img[:, 0] + 2.2*init_em_array

    w_offsets_array = np.arange(-s_win, s_win + 1, dtype=np.int32)
    abs_offsets_array = np.abs(w_offsets_array).astype(np.float32)[None, :]
    step_pen_array = np.float32(self.v_step_alpha)*abs_offsets_array
    y_base_idx_array = np.arange(track_h, dtype=np.int32)

    in_thick_band_array = np.zeros(track_w, dtype=bool)
    for i in range(len(thick_x_array) - 1):
      local_xc_t = (thick_x_array[i] - rx0) - xs_track_start
      if 6.0 < local_xc_t < track_w - 4.0:
        local_t1 = max(1, int(np.floor(local_xc_t - 4.0)))
        local_t2 = min(track_w - 2, int(np.ceil(local_xc_t + 4.0)))
        in_thick_band_array[local_t1:local_t2 + 1] = True

    for i in range(1, track_w):
      local_x_roi = xs_track_start + i
      local_prev_pad = np.pad(dp_mat[:, i - 1], (s_win, s_win), constant_values=-1e12)
      local_win_prev = sliding_window_view(local_prev_pad, 2*s_win + 1)

      local_in_ko = (
        abs(local_x_roi - filtered_obj["x_1875"]) <= self.knockout_half_w or
        abs(local_x_roi - filtered_obj["x_1925"]) <= self.knockout_half_w or
        in_thick_band_array[i]
      )
      local_beta = np.float32(self.v_air_beta_ko if local_in_ko else self.v_air_beta)
      local_gamma = np.float32(self.v_air_gamma_ko if local_in_ko else self.v_air_gamma)

      local_cs_col = air_csum_array[:, i - 1]
      local_cs_mid = 0.5*(local_cs_col[:-1] + local_cs_col[1:])
      local_cs_pad = np.pad(local_cs_mid, (s_win, s_win), mode="edge")
      local_win_cs = sliding_window_view(local_cs_pad, 2*s_win + 1)
      local_air_mat = np.abs(local_win_cs - local_cs_mid[:, None])

      local_cand_mat = local_win_prev - (step_pen_array + local_beta*local_air_mat + local_gamma*(local_air_mat*local_air_mat))
      local_best_wi = np.argmax(local_cand_mat, axis=1)
      local_best_s = local_cand_mat[y_base_idx_array, local_best_wi]
      local_best_p = np.clip(y_base_idx_array + w_offsets_array[local_best_wi], 0, track_h - 1)

      local_self_val = np.float32(0.0 if in_thick_band_array[i] else self.v_air_self_pen)
      local_air_self = np.where(sub_img[:, i] < 60.0, local_self_val, np.float32(0.0))
      local_em_col = np.float32(0.25)*emission_img[:, i] if in_thick_band_array[i] else emission_img[:, i]
      dp_mat[:, i] = local_best_s + local_em_col + local_air_self
      back_mat[:, i] = local_best_p

    end_em_array = (sub_img[:, max(0, track_w - 4):track_w] + sub_bonus_img[:, max(0, track_w - 4):track_w]).max(axis=1)
    y_cur = int(np.argmax(dp_mat[:, -1] + 1.8*end_em_array))
    seam_sub_array = np.zeros(track_w, dtype=np.int32)
    seam_sub_array[-1] = y_cur
    for i in range(track_w - 1, 0, -1):
      y_cur = int(back_mat[y_cur, i])
      seam_sub_array[i - 1] = y_cur

    #Return statement
    return {"in_thick_band_array": in_thick_band_array, "seam_sub_array": seam_sub_array, "sub_img": sub_img, "xs_track_start": xs_track_start, "ya_crop": ya_crop}


  def F_sampleAnnualPopulation (self, arg0_seam_obj, arg1_filtered_obj, arg2_grid_obj, arg3_panel_cfg_obj):
    """
    @brief Computes unbiased sub-pixel curve centroids, locks the 1975 milestone dot and evaluates every integer year to 1975.
    @param arg0_seam_obj Dictionary returned by E_traceViterbiSeam.
    @param arg1_filtered_obj Dictionary returned by D_filterGlyphComponents.
    @param arg2_grid_obj Calibrated 2D grid dictionary.
    @param arg3_panel_cfg_obj Panel configuration dictionary.
    @return Annual population dictionary mapping integer year strings to millions.
    """
    #Convert from parameters
    seam_obj = arg0_seam_obj
    filtered_obj = arg1_filtered_obj
    grid_obj = arg2_grid_obj
    panel_cfg_obj = arg3_panel_cfg_obj if arg3_panel_cfg_obj is not None else {}
    if seam_obj is None or filtered_obj is None or grid_obj is None:
      logging.error("CurveExtractor.F_sampleAnnualPopulation received None input.")
      return {}

    #Declare local instance variables
    all_pre_pairs = []
    annual_pop_obj = {}
    cal = self.calibrator_obj
    gh_roi_img = filtered_obj["gh_roi_img"]
    in_thick_band_array = seam_obj["in_thick_band_array"]
    od_roi_img = filtered_obj["od_roi_img"]
    pchip_fn = None
    pre_anchors_obj = panel_cfg_obj.get("pre_anchors", None)
    r_bot = float(panel_cfg_obj.get("r_bot", 35))
    r_top = float(panel_cfg_obj.get("r_top", 0))
    rx0 = filtered_obj["rx0"]
    ry0 = filtered_obj["ry0"]
    seam_sub_array = seam_obj["seam_sub_array"]
    sub_img = seam_obj["sub_img"]
    track_h, track_w = sub_img.shape
    u_idx_array = None
    unit_per_row = float(panel_cfg_obj.get("unit_per_row", 1.0))
    valid_xs_array = []
    valid_ys_array = []
    vx_u_array = None
    vy_u_array = None
    x_1975_abs = float(grid_obj["x_1975"])
    x_cols_array = grid_obj["x_cols"]
    x_start_px = float(filtered_obj["x_start_px"])
    xi75 = 0
    xs_track_start = seam_obj["xs_track_start"]
    y_1975_final = 0.0
    y_max_p = 0.0
    y_min_p = 0.0
    y_pred_75 = 0.0
    ya_crop = seam_obj["ya_crop"]
    year_start = int(panel_cfg_obj.get("year_start", 0))

    for i in range(track_w):
      if in_thick_band_array[i]:
        continue
      local_x_abs = float(rx0 + xs_track_start + i)
      local_min_vdist = float(np.min(np.abs(x_cols_array - local_x_abs)))

      local_yc = int(seam_sub_array[i])
      local_col_v = sub_img[:, i]
      if local_col_v[max(0, local_yc - 4):min(track_h, local_yc + 5)].max() < 70.0:
        continue

      local_ts1, local_ts2 = max(0, i - 4), min(track_w - 1, i + 4)
      local_slope = abs(float(seam_sub_array[local_ts2]) - float(seam_sub_array[local_ts1]))/max(1.0, float(local_ts2 - local_ts1))
      local_yl = float(seam_sub_array[max(0, i - 12)])
      local_yr = float(seam_sub_array[min(track_w - 1, i + 12)])

      local_ya_raw = local_yc
      for x in range(local_yc - 1, -1, -1):
        if local_col_v[x] >= 65.0:
          local_ya_raw = x
        else:
          break
      local_yb_raw = local_yc
      for x in range(local_yc + 1, track_h):
        if local_col_v[x] >= 65.0:
          local_yb_raw = x
        else:
          break

      local_ynbr = 0.5*(local_yl + local_yr)
      local_up_exc = local_ynbr - local_ya_raw
      local_dn_exc = local_yb_raw - local_ynbr
      local_run_h = local_yb_raw - local_ya_raw + 1
      local_not_vgrid = (local_min_vdist > 3.5)

      local_is_up = (
        local_not_vgrid and
        (local_run_h >= self.minkowski_min_run) and
        (local_yc < min(local_yl, local_yr) - 2.5) and
        (local_ya_raw < min(local_yl, local_yr) - self.minkowski_min_exc) and
        (local_up_exc > local_dn_exc + self.minkowski_min_exc) and
        (local_up_exc < self.minkowski_max_exc) and
        (i < track_w - 3)
      )
      local_is_dn = (
        local_not_vgrid and
        (local_run_h >= self.minkowski_min_run) and
        (local_yc > max(local_yl, local_yr) + 2.5) and
        (local_yb_raw > max(local_yl, local_yr) + self.minkowski_min_exc) and
        (local_dn_exc > local_up_exc + self.minkowski_min_exc) and
        (local_dn_exc < self.minkowski_max_exc) and
        (i < track_w - 3)
      )

      if local_is_up:
        local_y_sub = float(local_ya_raw) + self.pen_radius_px
      elif local_is_dn:
        local_y_sub = float(local_yb_raw) - self.pen_radius_px
      else:
        local_max_half = max(6, int(round(self.slope_chord_factor*local_slope + self.slope_chord_base)))
        local_y1 = max(0, max(local_ya_raw, local_yc - local_max_half) - 1)
        local_y2 = min(track_h, min(local_yb_raw, local_yc + local_max_half) + 2)
        local_w_vec = local_col_v[local_y1:local_y2]
        if float(local_w_vec.sum()) > 70.0:
          local_y_sub = float((np.arange(local_y1, local_y2)*local_w_vec).sum()/local_w_vec.sum())
        else:
          continue

      valid_xs_array.append(local_x_abs)
      valid_ys_array.append(local_y_sub + ry0 + ya_crop)

    if len(valid_xs_array) >= 3:
      local_vx = np.array(valid_xs_array, dtype=np.float64)
      local_vy = np.array(valid_ys_array, dtype=np.float64)
      local_tail = (local_vx >= x_1975_abs - 13.0) & (local_vx <= x_1975_abs - 3.5)
      if int(local_tail.sum()) < 3:
        local_tail = (local_vx >= x_1975_abs - 20.0) & (local_vx <= x_1975_abs - 3.5)
      if int(local_tail.sum()) >= 3:
        local_m, local_b = np.polyfit(local_vx[local_tail], local_vy[local_tail], 1)
        y_pred_75 = float(local_m*x_1975_abs + local_b)
      else:
        local_m = 0.0
        y_pred_75 = float(local_vy[-1])

      y_min_p = cal.F_rowToPixelY(grid_obj, r_top, x_1975_abs) + 1.0
      y_max_p = cal.F_rowToPixelY(grid_obj, r_bot, x_1975_abs) - 1.0
      y_pred_75 = max(y_min_p, min(y_max_p, y_pred_75))

      xi75 = int(round(filtered_obj["x_1975_roi"]))
      local_max_up = max(self.endpoint_search_px, min(115, int(round(0.45*(y_max_p - y_min_p)))))
      local_up_search = max(self.endpoint_search_px, min(local_max_up, int(round(self.endpoint_search_px + 6.0*max(0.0, -local_m)))))
      local_y_top_guard = min(y_pred_75 - 6.0, y_min_p + 11.0)
      local_ys_a = max(2, int(round(local_y_top_guard - ry0)), int(round((y_pred_75 - ry0) - local_up_search)))
      local_ys_b = min(od_roi_img.shape[0] - 2, int(round((y_pred_75 - ry0) + self.endpoint_search_px)))
      if local_ys_b > local_ys_a + 4 and 5 <= xi75 < od_roi_img.shape[1] - 5:
        local_od_clean = np.maximum(0.0, od_roi_img[local_ys_a:local_ys_b + 1, :] - 1.05*gh_roi_img[local_ys_a:local_ys_b + 1, :])
        local_dot_prof = np.minimum(local_od_clean[:, xi75 - 4:xi75 - 2].max(axis=1), local_od_clean[:, xi75 + 3:xi75 + 5].max(axis=1))
        local_spine = local_od_clean[:, xi75 - 3:xi75 + 4].max(axis=1)
        for x in range(min(len(local_spine) - 1, max(2, int(round(y_pred_75 - ry0)) - local_ys_a - 4)), 1, -1):
          if local_spine[x] < 25.0 and local_spine[x - 1] < 25.0 and local_spine[x - 2] < 25.0:
            local_dot_prof[:x + 1] = 0.0
            break
        local_prox = np.maximum(0.40, 1.0 - 0.0045*np.abs(np.arange(local_ys_a, local_ys_b + 1) - (y_pred_75 - ry0)))
        local_pk = int(np.argmax(local_dot_prof*local_prox))
        if float(local_dot_prof[local_pk]) > self.dot_flank_min_od:
          local_p1, local_p2 = max(0, local_pk - 3), min(len(local_dot_prof), local_pk + 4)
          local_wd = local_dot_prof[local_p1:local_p2]
          y_1975_final = max(y_min_p, min(y_max_p, ry0 + local_ys_a + float((np.arange(local_p1, local_p2)*local_wd).sum()/local_wd.sum())))
        else:
          y_1975_final = y_pred_75
      else:
        y_1975_final = y_pred_75

      valid_xs_array.append(x_1975_abs)
      valid_ys_array.append(y_1975_final)
      if valid_xs_array[0] > x_start_px + 0.5:
        valid_xs_array.insert(0, x_start_px)
        valid_ys_array.insert(0, valid_ys_array[0])

    if len(valid_xs_array) >= 2:
      vx_u_array, u_idx_array = np.unique(np.array(valid_xs_array, dtype=np.float64), return_index=True)
      vy_u_array = np.array(valid_ys_array, dtype=np.float64)[u_idx_array]
      pchip_fn = PchipInterpolator(vx_u_array, vy_u_array, extrapolate=True)
    elif len(valid_xs_array) == 1:
      local_y_const = float(valid_ys_array[0])
      pchip_fn = lambda x_val: local_y_const
    else:
      local_y_const = float(ry0 + ya_crop + seam_sub_array[0])
      pchip_fn = lambda x_val: local_y_const

    for i in range(year_start, self.end_year_cap + 1):
      local_xp = cal.D_yearToPixelX(grid_obj, float(i))
      local_yp = float(pchip_fn(local_xp))
      local_pop = cal.E_pixelYToPopulation(grid_obj, local_yp, local_xp, {"r_bot": r_bot, "unit_per_row": unit_per_row})
      annual_pop_obj[str(i)] = round(local_pop, self.rounding_decimals)

    if isinstance(pre_anchors_obj, dict) and len(pre_anchors_obj) > 0:
      for i in pre_anchors_obj:
        all_pre_pairs.append((int(i), float(pre_anchors_obj[i])))
      all_pre_pairs.append((year_start, float(annual_pop_obj[str(year_start)])))
      all_pre_pairs.sort(key=lambda pair: pair[0])
      local_pre_yrs = np.array([pair[0] for pair in all_pre_pairs], dtype=np.float64)
      local_pre_vals = np.array([pair[1] for pair in all_pre_pairs], dtype=np.float64)
      local_pre_interp = PchipInterpolator(local_pre_yrs, local_pre_vals, extrapolate=True)
      local_ordered_obj = {}
      for i in range(int(all_pre_pairs[0][0]), year_start):
        local_ordered_obj[str(i)] = round(max(0.0, float(local_pre_interp(float(i)))), self.rounding_decimals)
      for i in range(year_start, self.end_year_cap + 1):
        local_ordered_obj[str(i)] = annual_pop_obj[str(i)]
      annual_pop_obj = local_ordered_obj

    #Return statement
    return annual_pop_obj

