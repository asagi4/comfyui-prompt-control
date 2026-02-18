import os
import logging

log = logging.getLogger("comfyui-prompt-control")

if os.environ.get("PC_USE_OLD_PARSER", "0") != "0":
    log.info("Using new parser implementation. Set PC_USE_OLD_PARSER=1 to use old parser instead")
    from .parser_parsy import parse_prompt_schedules
else:
    from .parser_lark import parse_prompt_schedules
