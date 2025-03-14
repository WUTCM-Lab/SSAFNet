from .apc import APCHead
from .aspp import ASPPHead
from .aspp_plus import ASPPPlusHead
from .da import DAHead
from .dnl import DNLHead
from .fcfpn import FCFPNHead
from .fcn import FCNHead
from .gc import GCHead
from .psa import PSAHead
from .psp import PSPHead
from .unet import UNetHead
from .uper import UPerHead
from .seg import SegHead
from .cefpn import CEFPNHead
from .mlp import MLPHead
from .edge import EdgeHead
from .bs_attn import BSHead
from .bs_fuse import BSFuseHead

__all__ = [
    'APCHead', 'ASPPHead', 'ASPPPlusHead', 'DAHead', 'DNLHead', 'FCFPNHead', 'FCNHead',
    'GCHead', 'PSAHead', 'PSPHead', 'UNetHead', 'UPerHead', 'SegHead', 'CEFPNHead', 'MLPHead', 'EdgeHead',
    'BSHead', 'BSFuseHead'
]