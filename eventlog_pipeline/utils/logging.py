import logging
from pathlib import Path

def setup_logger(log_dir: Path , verbose: bool = True):
    
    
    logger = logging.getLogger("repaireventlog")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # For Console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    
    
    # For the log file
    fh = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    
    
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger
