import cv2
import logging
import numpy as np
import pymupdf


class GridCalibrator:
  """
  @brief Renders right-hand McEvedy & Jones chart pages and calibrates non-continuous X-axis breaks and 2D Y-grid meshes.
  """

  def __init__ (self, arg0_config_obj):
    """
    @brief Initialises GridCalibrator with rendering and axis geometry parameters from pipeline_config.json5.
    @param arg0_config_obj Parsed configuration dictionary.
    """
    #Convert from parameters
    config_obj = arg0_config_obj if arg0_config_obj is not None else {}

    #Declare local instance variables
    render_cfg_obj = config_obj.get("render", {})
    x_cfg_obj = config_obj.get("x_axis", {})
    y_cfg_obj = config_obj.get("y_axis", {})

    self.bg_blur_sigma = float(render_cfg_obj.get("background_blur_sigma", 15.0))
    self.bg_dilate_px = int(render_cfg_obj.get("background_dilate_px", 51))
    self.column_count = int(x_cfg_obj.get("column_count", 23))
    self.column_years_array = list(x_cfg_obj.get("column_years", []))
    self.dpi = int(render_cfg_obj.get("dpi", 300))
    self.end_year_cap = int(x_cfg_obj.get("end_year_cap", 1975))
    self.pitch_max_px = float(x_cfg_obj.get("pitch_search_max_px", 50.4))
    self.pitch_min_px = float(x_cfg_obj.get("pitch_search_min_px", 49.4))
    self.pitch_steps = int(x_cfg_obj.get("pitch_search_steps", 51))
    self.right_page_ratio = float(render_cfg_obj.get("right_page_x_start_ratio", 0.5))
    self.row_count = int(y_cfg_obj.get("row_count", 35))
    self.x0_max_px = int(x_cfg_obj.get("x0_search_max_px", 355))
    self.x0_min_px = int(x_cfg_obj.get("x0_search_min_px", 165))
    self.y0_max_px = int(y_cfg_obj.get("y0_search_max_px", 255))
    self.y0_min_px = int(y_cfg_obj.get("y0_search_min_px", 105))


  def A_renderAndNormalisePage (self, arg0_pdf_doc, arg1_page_index):
    """
    @brief Clips the right-hand chart page from a two-page PDF spread and computes background-normalised optical density.
    @param arg0_pdf_doc Open PyMuPDF document handle.
    @param arg1_page_index Zero-based PDF spread page index.
    @return Dictionary containing 'gray_img' (uint8) and 'od_img' (float32 optical density 0..255).
    """
    #Convert from parameters
    pdf_doc = arg0_pdf_doc
    page_index = int(arg1_page_index)

    #Declare local instance variables
    bg_img = None
    clip_rect = None
    dilate_kernel = None
    gray_img = None
    od_img = None
    page_obj = None
    page_rect = None
    pix_obj = None

    if pdf_doc is None or page_index < 0 or page_index >= len(pdf_doc):
      logging.error("GridCalibrator.A_renderAndNormalisePage received invalid page_index %s.", page_index)
      return None

    page_obj = pdf_doc[page_index]
    page_rect = page_obj.rect
    clip_rect = pymupdf.Rect(page_rect.width*self.right_page_ratio, 0.0, page_rect.width, page_rect.height)
    pix_obj = page_obj.get_pixmap(dpi=self.dpi, clip=clip_rect, colorspace=pymupdf.csGRAY)
    gray_img = np.frombuffer(pix_obj.samples, dtype=np.uint8).reshape(pix_obj.height, pix_obj.width)

    dilate_kernel = np.ones((self.bg_dilate_px, self.bg_dilate_px), dtype=np.uint8)
    bg_img = cv2.GaussianBlur(cv2.dilate(gray_img, dilate_kernel), (0, 0), self.bg_blur_sigma)
    od_img = np.clip(bg_img.astype(np.float32) - gray_img.astype(np.float32), 0.0, 255.0)

    #Return statement
    return {
      "gray_img": gray_img,
      "od_img": od_img
    }


  def B_detectVerticalLattice (self, arg0_od_img):
    """
    @brief Locates all 23 vertical grid columns (BC400..AD2000 across the 1000 and 1500 scale breaks) plus the 1975 end line.
    @param arg0_od_img Float32 optical density image of the right-hand chart page.
    @return Dictionary with sub-pixel 'x_cols' (23,), 'x_1975', 'dx' and 'x0'.
    """
    #Convert from parameters
    od_img = arg0_od_img

    #Declare local instance variables
    best_idx = None
    c_idx_array = None
    d75 = 0.0
    denom = 0.0
    dx = 0.0
    dxs_array = None
    h = 0
    idx_grid = None
    pk75 = 0
    poly_x_array = None
    prof75_array = None
    vp_array = None
    w = 0
    x0 = 0.0
    x0_span = 0
    x0s_array = None
    x1975_est = 0.0
    x_1975 = 0.0
    x_cols_array = None
    xi75 = 0
    y_scan_bot = 0
    y_scan_top = 0

    if od_img is None or od_img.ndim != 2:
      logging.error("GridCalibrator.B_detectVerticalLattice received invalid od_img.")
      return None

    h, w = od_img.shape
    y_scan_top = min(h - 2, 200)
    y_scan_bot = min(h - 1, 1850)
    vp_array = cv2.dilate(
      od_img[y_scan_top:y_scan_bot, :].mean(axis=0).astype(np.float32).reshape(1, -1),
      np.ones((1, 3), dtype=np.uint8)
    )[0]

    dxs_array = np.linspace(self.pitch_min_px, self.pitch_max_px, self.pitch_steps)[:, None, None]
    x0s_array = np.arange(self.x0_min_px, self.x0_max_px)[None, :, None]
    x0_span = self.x0_max_px - self.x0_min_px
    idx_grid = np.clip(np.round(x0s_array + np.arange(self.column_count)[None, None, :]*dxs_array).astype(int), 0, w - 1)
    best_idx = np.unravel_index(vp_array[idx_grid].sum(axis=2).argmax(), (self.pitch_steps, x0_span))
    dx = float(dxs_array[best_idx[0], 0, 0])
    x0 = float(x0s_array[0, best_idx[1], 0])

    x_cols_array = np.zeros(self.column_count, dtype=np.float64)
    for i in range(self.column_count):
      local_xc_est = x0 + i*dx
      local_xi = int(round(local_xc_est))
      if i <= 12:
        local_prof = np.median(
          od_img[max(0, y_scan_bot - 450):y_scan_bot, max(0, local_xi - 4):min(w, local_xi + 5)],
          axis=0
        )
      else:
        local_prof = np.median(
          od_img[y_scan_top:y_scan_bot, max(0, local_xi - 4):min(w, local_xi + 5)],
          axis=0
        )
      local_pk = int(np.argmax(local_prof))
      if 0 < local_pk < len(local_prof) - 1:
        local_denom = local_prof[local_pk - 1] - 2.0*local_prof[local_pk] + local_prof[local_pk + 1]
        if abs(local_denom) > 1e-3:
          local_delta = 0.5*(local_prof[local_pk - 1] - local_prof[local_pk + 1])/local_denom
          local_delta = max(-0.5, min(0.5, float(local_delta)))
        else:
          local_delta = 0.0
      else:
        local_delta = 0.0
      local_x_raw = (local_xi - 4) + local_pk + local_delta
      if abs(local_x_raw - local_xc_est) <= 2.8 and local_prof[local_pk] > 45.0:
        x_cols_array[i] = local_x_raw
      else:
        x_cols_array[i] = local_xc_est

    c_idx_array = np.arange(self.column_count, dtype=np.float64)
    poly_x_array = np.polyval(np.polyfit(c_idx_array, x_cols_array, 2), c_idx_array)
    x_cols_array = np.where(np.abs(x_cols_array - poly_x_array) < 1.8, 0.75*x_cols_array + 0.25*poly_x_array, poly_x_array)

    x1975_est = 0.5*(x_cols_array[21] + x_cols_array[22])
    xi75 = int(round(x1975_est))
    prof75_array = np.median(od_img[y_scan_top:y_scan_bot, max(0, xi75 - 5):min(w, xi75 + 6)], axis=0)
    pk75 = int(np.argmax(prof75_array))
    if 0 < pk75 < len(prof75_array) - 1:
      denom = prof75_array[pk75 - 1] - 2.0*prof75_array[pk75] + prof75_array[pk75 + 1]
      if abs(denom) > 1e-3:
        d75 = max(-0.5, min(0.5, float(0.5*(prof75_array[pk75 - 1] - prof75_array[pk75 + 1])/denom)))
    if prof75_array[pk75] > 80.0:
      x_1975 = float((xi75 - 5) + pk75 + d75)
    else:
      x_1975 = float(x1975_est)

    #Return statement
    return {
      "dx": dx,
      "x0": x0,
      "x_1975": x_1975,
      "x_cols": x_cols_array
    }


  def C_detectHorizontalMesh (self, arg0_od_img, arg1_vert_lattice_obj):
    """
    @brief Calibrates all 36 horizontal grid row lines (0..35) plus quadratic spine-curl tilt across the chart width.
    @param arg0_od_img Float32 optical density image.
    @param arg1_vert_lattice_obj Output dictionary from B_detectVerticalLattice.
    @return Complete 2D grid calibration dictionary ready for coordinate transforms and curve extraction.
    """
    #Convert from parameters
    od_img = arg0_od_img
    vert_lattice_obj = arg1_vert_lattice_obj

    #Declare local instance variables
    best_j = None
    dy = 0.0
    dys_array = None
    h = 0
    hp_array = None
    idy_grid = None
    line_count = 0
    poly_y_array = None
    r_arr = None
    ref_shift = 0.0
    tilt_dys_array = []
    tilt_poly_array = None
    tilt_xs_array = []
    w = 0
    x_cols_array = None
    xl_ref = 0
    xr_ref = 0
    y0 = 0.0
    y0_span = 0
    y0s_array = None
    y35_ref = 0.0
    y_rows_ref_array = None

    if od_img is None or vert_lattice_obj is None:
      logging.error("GridCalibrator.C_detectHorizontalMesh received None input.")
      return None

    h, w = od_img.shape
    line_count = self.row_count + 1
    x_cols_array = vert_lattice_obj["x_cols"]
    xl_ref = max(0, int(round(x_cols_array[13])))
    xr_ref = min(w, int(round(x_cols_array[21])))

    hp_array = cv2.dilate(
      np.median(od_img[:, xl_ref:xr_ref], axis=1).astype(np.float32).reshape(1, -1),
      np.ones((1, 3), dtype=np.uint8)
    )[0]
    dys_array = np.linspace(self.pitch_min_px, self.pitch_max_px, self.pitch_steps)[:, None, None]
    y0s_array = np.arange(self.y0_min_px, self.y0_max_px)[None, :, None]
    y0_span = self.y0_max_px - self.y0_min_px
    idy_grid = np.clip(np.round(y0s_array + np.arange(line_count)[None, None, :]*dys_array).astype(int), 0, h - 1)
    best_j = np.unravel_index(hp_array[idy_grid].sum(axis=2).argmax(), (self.pitch_steps, y0_span))
    dy = float(dys_array[best_j[0], 0, 0])
    y0 = float(y0s_array[0, best_j[1], 0])

    y_rows_ref_array = np.zeros(line_count, dtype=np.float64)
    for i in range(line_count):
      local_yr_est = y0 + i*dy
      local_yi = int(round(local_yr_est))
      local_prof = np.median(od_img[max(0, local_yi - 4):min(h, local_yi + 5), xl_ref:xr_ref], axis=1)
      local_pk = int(np.argmax(local_prof))
      if 0 < local_pk < len(local_prof) - 1:
        local_denom = local_prof[local_pk - 1] - 2.0*local_prof[local_pk] + local_prof[local_pk + 1]
        if abs(local_denom) > 1e-3:
          local_delta = max(-0.5, min(0.5, float(0.5*(local_prof[local_pk - 1] - local_prof[local_pk + 1])/local_denom)))
        else:
          local_delta = 0.0
      else:
        local_delta = 0.0
      local_y_raw = (local_yi - 4) + local_pk + local_delta
      if abs(local_y_raw - local_yr_est) <= 2.8 and local_prof[local_pk] > 40.0:
        y_rows_ref_array[i] = local_y_raw
      else:
        y_rows_ref_array[i] = local_yr_est

    r_arr = np.arange(line_count, dtype=np.float64)
    poly_y_array = np.polyval(np.polyfit(r_arr, y_rows_ref_array, 2), r_arr)
    y_rows_ref_array = np.where(
      np.abs(y_rows_ref_array - poly_y_array) < 1.8,
      0.75*y_rows_ref_array + 0.25*poly_y_array,
      poly_y_array
    )

    y35_ref = y_rows_ref_array[self.row_count]
    for i in range(self.column_count - 1):
      local_xa = max(0, int(round(x_cols_array[i] + 8)))
      local_xb = min(w, int(round(x_cols_array[i + 1] - 8)))
      if local_xb > local_xa + 5:
        local_yi = int(round(y35_ref))
        local_prof = np.median(od_img[max(0, local_yi - 6):min(h, local_yi + 7), local_xa:local_xb], axis=1)
        local_pk = int(np.argmax(local_prof))
        if local_prof[local_pk] > 90.0:
          tilt_xs_array.append(0.5*(local_xa + local_xb))
          tilt_dys_array.append((local_yi - 6 + local_pk) - y35_ref)

    if len(tilt_xs_array) >= 6:
      tilt_poly_array = np.polyfit(np.array(tilt_xs_array), np.array(tilt_dys_array), 2)
      ref_shift = float(np.polyval(tilt_poly_array, 0.5*(xl_ref + xr_ref)))
    else:
      tilt_poly_array = np.array([0.0, 0.0, 0.0], dtype=np.float64)
      ref_shift = 0.0

    #Return statement
    return {
      "dx": vert_lattice_obj["dx"],
      "dy": dy,
      "ref_shift": ref_shift,
      "tilt_poly": tilt_poly_array,
      "x_1975": vert_lattice_obj["x_1975"],
      "x_cols": x_cols_array,
      "y_rows_ref": y_rows_ref_array
    }


  def D_yearToPixelX (self, arg0_grid_obj, arg1_year):
    """
    @brief Maps a historical calendar year (-400..1975) to sub-pixel X across piecewise-linear X-axis breaks (1000, 1500, 1975).
    @param arg0_grid_obj Calibrated grid dictionary from C_detectHorizontalMesh.
    @param arg1_year Calendar year (AD1 is represented as 0).
    @return Sub-pixel X coordinate on the right-hand chart image.
    """
    #Convert from parameters
    grid_obj = arg0_grid_obj
    year = float(arg1_year)

    #Declare local instance variables
    knots_x_array = None
    knots_yr_array = None

    if grid_obj is None:
      logging.error("GridCalibrator.D_yearToPixelX received None grid_obj.")
      return 0.0

    knots_yr_array = self.column_years_array[:-1] + [self.end_year_cap, self.column_years_array[-1]]
    knots_x_array = list(grid_obj["x_cols"][:-1]) + [grid_obj["x_1975"], grid_obj["x_cols"][-1]]

    #Return statement
    return float(np.interp(year, knots_yr_array, knots_x_array))


  def E_pixelYToPopulation (self, arg0_grid_obj, arg1_y_px, arg2_x_px, arg3_options):
    """
    @brief Converts a sub-pixel (x_px, y_px) point on a panel into millions using tilt-corrected 2D mesh interpolation.
    @param arg0_grid_obj Calibrated grid dictionary.
    @param arg1_y_px Sub-pixel Y coordinate.
    @param arg2_x_px Sub-pixel X coordinate.
    @param arg3_options Panel options dict containing 'r_bot' and 'unit_per_row'.
    @return Non-negative population in millions.
    """
    #Convert from parameters
    grid_obj = arg0_grid_obj
    y_px = float(arg1_y_px)
    x_px = float(arg2_x_px)
    options = arg3_options if arg3_options is not None else {}

    #Declare local instance variables
    line_count = 0
    r_bot = float(options.get("r_bot", self.row_count))
    r_float = 0.0
    rows_above_bot = 0.0
    tilt_px = 0.0
    unit_per_row = float(options.get("unit_per_row", 1.0))
    y_corr = 0.0
    y_rows_array = None

    if grid_obj is None:
      logging.error("GridCalibrator.E_pixelYToPopulation received None grid_obj.")
      return 0.0

    y_rows_array = grid_obj["y_rows_ref"]
    line_count = len(y_rows_array)
    tilt_px = float(np.polyval(grid_obj["tilt_poly"], x_px)) - grid_obj["ref_shift"]
    y_corr = y_px - tilt_px
    r_float = float(np.interp(y_corr, y_rows_array, np.arange(line_count, dtype=np.float64)))
    rows_above_bot = r_bot - r_float

    #Return statement
    return max(0.0, rows_above_bot*unit_per_row)


  def F_rowToPixelY (self, arg0_grid_obj, arg1_row_float, arg2_x_px):
    """
    @brief Evaluates the tilt-adjusted pixel Y coordinate of fractional grid row arg1_row_float at horizontal position arg2_x_px.
    @param arg0_grid_obj Calibrated grid dictionary.
    @param arg1_row_float Fractional grid row index in [0 .. 35].
    @param arg2_x_px Horizontal pixel position.
    @return Sub-pixel Y coordinate.
    """
    #Convert from parameters
    grid_obj = arg0_grid_obj
    row_float = float(arg1_row_float)
    x_px = float(arg2_x_px)

    #Declare local instance variables
    frac = 0.0
    max_r0 = 0
    r0 = 0
    tilt_px = 0.0
    y_base = 0.0
    y_rows_array = None

    if grid_obj is None:
      logging.error("GridCalibrator.F_rowToPixelY received None grid_obj.")
      return 0.0

    y_rows_array = grid_obj["y_rows_ref"]
    max_r0 = len(y_rows_array) - 2
    r0 = int(max(0, min(max_r0, np.floor(row_float))))
    frac = row_float - r0
    y_base = y_rows_array[r0]*(1.0 - frac) + y_rows_array[r0 + 1]*frac
    tilt_px = float(np.polyval(grid_obj["tilt_poly"], x_px)) - grid_obj["ref_shift"]

    #Return statement
    return float(y_base + tilt_px)

