"""
@Author: Conghao Wong
@Date: 2023-08-08 15:52:46
@LastEditors: Conghao Wong
@LastEditTime: 2024-05-30 10:53:52
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import qpid as qpid

from . import original_models
from .__args import PhysicalCircleArgs, SocialCircleArgs
from .ev_sc import EVSCModel, EVSCStructure
from .ev_scp import EVSCPlusModel, EVSCPlusStructure
from .msn_sc import MSNSCModel, MSNSCStructure
from .msn_scp import MSNSCPlusModel, MSNSCPlusStructure
from .trans_sc import TransformerSCModel, TransformerSCStructure
from .trans_scp import TransformerSCPlusModel, TransformerSCPlusStructure
from .v_sc import VSCModel, VSCStructure
from .v_scp import VSCPlusModel, VSCPlusStructure

# Add new args
qpid.register_args(SocialCircleArgs, 'SocialCircle Args')
qpid.register_args(PhysicalCircleArgs, 'PhysicalCircle Args')
qpid.add_arg_alias(alias=['--sc', '-sc', '--socialCircle'],
                   command=['--model', 'MKII', '--loads'],
                   pattern='{},speed')

# Register Circle-based models
qpid.register(
    # SocialCircle Models
    evsc=[EVSCStructure, EVSCModel],
    vsc=[VSCStructure, VSCModel],
    msnsc=[MSNSCStructure, MSNSCModel],
    transsc=[TransformerSCStructure, TransformerSCModel],

    # SocialCircle+ Models (SocialCircle + PhysicalCircle)
    evspc=[EVSCPlusStructure, EVSCPlusModel],
    vspc=[VSCPlusStructure, VSCPlusModel],
    msnspc=[MSNSCPlusStructure, MSNSCPlusModel],
    transspc=[TransformerSCPlusStructure, TransformerSCPlusModel],

    evscp=[EVSCPlusStructure, EVSCPlusModel],
    vscp=[VSCPlusStructure, VSCPlusModel],
    msnscp=[MSNSCPlusStructure, MSNSCPlusModel],
    transscp=[TransformerSCPlusStructure, TransformerSCPlusModel],
)
