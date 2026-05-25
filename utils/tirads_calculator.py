"""
ACR TI-RADS (Thyroid Imaging Reporting and Data System) Calculator
Based on: ACR TI-RADS White Paper (Tessler et al., JACR 2017)
https://doi.org/10.1016/j.jacr.2017.01.046

Scoring system used by radiologists worldwide to assess malignancy risk
of thyroid nodules from ultrasound features.

5 categories, each scored independently:
  1. Composition
  2. Echogenicity
  3. Shape
  4. Margin
  5. Echogenic Foci

Total points → TI-RADS Level → Malignancy risk % → FNA recommendation
"""


# ── Scoring tables (ACR TI-RADS 2017) ────────────────────────────

COMPOSITION_SCORES = {
    'cystic_or_almost_cystic': 0,
    'spongiform':               0,
    'mixed_cystic_solid':       1,
    'solid_or_almost_solid':    2,
}

ECHOGENICITY_SCORES = {
    'anechoic':                 0,
    'hyperechoic_or_isoechoic': 1,
    'hypoechoic':               2,
    'very_hypoechoic':          3,
}

SHAPE_SCORES = {
    'wider_than_tall': 0,
    'taller_than_wide': 3,
}

MARGIN_SCORES = {
    'smooth':                   0,
    'ill_defined':              0,
    'lobulated_or_irregular':   2,
    'extra_thyroidal_extension': 3,
}

# Echogenic foci — can select MULTIPLE, scores are additive
ECHOGENIC_FOCI_SCORES = {
    'none_or_large_comet_tail': 0,
    'macrocalcifications':      1,
    'peripheral_calcifications': 2,
    'punctate_echogenic_foci':  3,
}

# TI-RADS level lookup by total points
def get_tirads_level(points: int) -> dict:
    if points == 0:
        return {
            'level': 'TR1',
            'label': 'Benign',
            'risk': '< 1%',
            'risk_pct': 0.5,
            'recommendation': 'No FNA needed',
            'color': '#198754',   # green
            'badge': 'success',
        }
    elif points <= 2:
        return {
            'level': 'TR2',
            'label': 'Not Suspicious',
            'risk': '< 1.5%',
            'risk_pct': 1.5,
            'recommendation': 'No FNA needed',
            'color': '#198754',
            'badge': 'success',
        }
    elif points <= 4:
        return {
            'level': 'TR3',
            'label': 'Mildly Suspicious',
            'risk': '< 5%',
            'risk_pct': 5,
            'recommendation': 'FNA if ≥ 2.5 cm; Follow if ≥ 1.5 cm',
            'color': '#ffc107',   # yellow
            'badge': 'warning',
        }
    elif points <= 6:
        return {
            'level': 'TR4',
            'label': 'Moderately Suspicious',
            'risk': '5–20%',
            'risk_pct': 12,
            'recommendation': 'FNA if ≥ 1.5 cm; Follow if ≥ 1 cm',
            'color': '#fd7e14',   # orange
            'badge': 'warning',
        }
    else:  # >= 7
        return {
            'level': 'TR5',
            'label': 'Highly Suspicious',
            'risk': '> 20%',
            'risk_pct': 35,
            'recommendation': 'FNA if ≥ 1 cm; Follow if ≥ 0.5 cm',
            'color': '#dc3545',   # red
            'badge': 'danger',
        }


def calculate_tirads(data: dict) -> dict:
    """
    Calculate ACR TI-RADS score from ultrasound features.

    Parameters
    ----------
    data : dict with keys:
        composition       : str  (one of COMPOSITION_SCORES keys)
        echogenicity      : str  (one of ECHOGENICITY_SCORES keys)
        shape             : str  (one of SHAPE_SCORES keys)
        margin            : str  (one of MARGIN_SCORES keys)
        echogenic_foci    : list (subset of ECHOGENIC_FOCI_SCORES keys)
        nodule_size_cm    : float (largest dimension in cm)
        patient_age       : int  (optional, for context)
        patient_sex       : str  (optional, for context)

    Returns
    -------
    dict with full scoring breakdown, TI-RADS level, risk, recommendation
    """
    errors = []

    # ── Score each category ───────────────────────────────────────
    composition = data.get('composition', '')
    comp_score  = COMPOSITION_SCORES.get(composition, None)
    if comp_score is None:
        errors.append('Invalid composition value: ' + str(composition))
        comp_score = 0

    echogenicity = data.get('echogenicity', '')
    echo_score   = ECHOGENICITY_SCORES.get(echogenicity, None)
    if echo_score is None:
        errors.append('Invalid echogenicity value: ' + str(echogenicity))
        echo_score = 0

    # Special rule: if composition is cystic/spongiform, echogenicity = 0
    if composition in ('cystic_or_almost_cystic', 'spongiform'):
        echo_score = 0

    shape       = data.get('shape', '')
    shape_score = SHAPE_SCORES.get(shape, None)
    if shape_score is None:
        errors.append('Invalid shape value: ' + str(shape))
        shape_score = 0

    margin       = data.get('margin', '')
    margin_score = MARGIN_SCORES.get(margin, None)
    if margin_score is None:
        errors.append('Invalid margin value: ' + str(margin))
        margin_score = 0

    foci_list  = data.get('echogenic_foci', ['none_or_large_comet_tail'])
    if not foci_list:
        foci_list = ['none_or_large_comet_tail']
    foci_score = sum(ECHOGENIC_FOCI_SCORES.get(f, 0) for f in foci_list)

    total_points = comp_score + echo_score + shape_score + margin_score + foci_score

    # ── TI-RADS level ─────────────────────────────────────────────
    level_info = get_tirads_level(total_points)

    # ── Size-based FNA recommendation ────────────────────────────
    nodule_size = float(data.get('nodule_size_cm', 0) or 0)
    fna_recommended = False
    follow_up       = False

    level = level_info['level']
    if level == 'TR3':
        if nodule_size >= 2.5:
            fna_recommended = True
        elif nodule_size >= 1.5:
            follow_up = True
    elif level == 'TR4':
        if nodule_size >= 1.5:
            fna_recommended = True
        elif nodule_size >= 1.0:
            follow_up = True
    elif level == 'TR5':
        if nodule_size >= 1.0:
            fna_recommended = True
        elif nodule_size >= 0.5:
            follow_up = True

    # ── Final verdict ─────────────────────────────────────────────
    if level_info['risk_pct'] < 5:
        verdict = 'Likely Benign'
        verdict_color = '#198754'
    elif level_info['risk_pct'] < 20:
        verdict = 'Indeterminate — Further Evaluation Needed'
        verdict_color = '#fd7e14'
    else:
        verdict = 'Suspicious for Malignancy'
        verdict_color = '#dc3545'

    # ── Score breakdown ───────────────────────────────────────────
    breakdown = [
        {'category': 'Composition',     'value': composition.replace('_', ' ').title(),    'score': comp_score},
        {'category': 'Echogenicity',    'value': echogenicity.replace('_', ' ').title(),   'score': echo_score},
        {'category': 'Shape',           'value': shape.replace('_', ' ').title(),          'score': shape_score},
        {'category': 'Margin',          'value': margin.replace('_', ' ').title(),         'score': margin_score},
        {'category': 'Echogenic Foci',  'value': ', '.join(f.replace('_', ' ').title() for f in foci_list), 'score': foci_score},
    ]

    return {
        'success':          True,
        'total_points':     total_points,
        'tirads_level':     level_info['level'],
        'tirads_label':     level_info['label'],
        'malignancy_risk':  level_info['risk'],
        'risk_pct':         level_info['risk_pct'],
        'color':            level_info['color'],
        'badge':            level_info['badge'],
        'recommendation':   level_info['recommendation'],
        'fna_recommended':  fna_recommended,
        'follow_up':        follow_up,
        'verdict':          verdict,
        'verdict_color':    verdict_color,
        'nodule_size_cm':   nodule_size,
        'breakdown':        breakdown,
        'errors':           errors,
        # Human-readable summary
        'summary': (
            f"TI-RADS {level_info['level']} ({level_info['label']}) — "
            f"Malignancy risk {level_info['risk']} — "
            f"{'FNA recommended' if fna_recommended else 'Follow-up recommended' if follow_up else 'No immediate action needed'}"
        ),
    }


# ── Option descriptions for the UI ───────────────────────────────

UI_OPTIONS = {
    'composition': [
        {'value': 'cystic_or_almost_cystic', 'label': 'Cystic or Almost Cystic',  'score': 0, 'desc': '>= 90% fluid'},
        {'value': 'spongiform',              'label': 'Spongiform',               'score': 0, 'desc': '>= 50% tiny cysts'},
        {'value': 'mixed_cystic_solid',      'label': 'Mixed Cystic and Solid',   'score': 1, 'desc': 'Partial fluid, partial solid'},
        {'value': 'solid_or_almost_solid',   'label': 'Solid or Almost Solid',    'score': 2, 'desc': 'Minimal or no fluid'},
    ],
    'echogenicity': [
        {'value': 'anechoic',                 'label': 'Anechoic',                  'score': 0, 'desc': 'No internal echoes (fluid)'},
        {'value': 'hyperechoic_or_isoechoic', 'label': 'Hyperechoic or Isoechoic', 'score': 1, 'desc': 'Same or brighter than thyroid'},
        {'value': 'hypoechoic',               'label': 'Hypoechoic',               'score': 2, 'desc': 'Darker than thyroid'},
        {'value': 'very_hypoechoic',          'label': 'Very Hypoechoic',          'score': 3, 'desc': 'Darker than strap muscles'},
    ],
    'shape': [
        {'value': 'wider_than_tall', 'label': 'Wider than Tall (Horizontal)', 'score': 0, 'desc': 'AP < transverse dimension'},
        {'value': 'taller_than_wide', 'label': 'Taller than Wide (Vertical)', 'score': 3, 'desc': 'AP > transverse dimension'},
    ],
    'margin': [
        {'value': 'smooth',                    'label': 'Smooth',                     'score': 0, 'desc': 'Uniform, well-defined edge'},
        {'value': 'ill_defined',               'label': 'Ill-Defined',                'score': 0, 'desc': 'Cannot trace edge clearly'},
        {'value': 'lobulated_or_irregular',    'label': 'Lobulated or Irregular',     'score': 2, 'desc': 'Bumpy or jagged edge'},
        {'value': 'extra_thyroidal_extension', 'label': 'Extra-Thyroidal Extension',  'score': 3, 'desc': 'Extends beyond thyroid capsule'},
    ],
    'echogenic_foci': [
        {'value': 'none_or_large_comet_tail', 'label': 'None or Large Comet-Tail Artifacts', 'score': 0, 'desc': 'No suspicious foci'},
        {'value': 'macrocalcifications',      'label': 'Macrocalcifications',               'score': 1, 'desc': 'Large bright spots with shadow'},
        {'value': 'peripheral_calcifications','label': 'Peripheral (Rim) Calcifications',   'score': 2, 'desc': 'Calcifications at nodule edge'},
        {'value': 'punctate_echogenic_foci',  'label': 'Punctate Echogenic Foci',           'score': 3, 'desc': 'Tiny bright spots (microcalcifications)'},
    ],
}
