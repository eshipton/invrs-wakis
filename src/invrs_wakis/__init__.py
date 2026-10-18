"""Wakis (FIT time-domain, exact discrete adjoint) backend for invrs-gym challenges."""

from invrs_wakis._design import PixelDesign
from invrs_wakis.ceviche import (
    CevicheGeometry,
    CevicheSimulation,
    SimOptions,
    WakisCevicheComponent,
    wakis_ceviche_challenge,
)

__version__ = "0.1.0"
