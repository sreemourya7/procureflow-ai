import os
from pathlib import Path
from .api import create_app

app = create_app(Path(os.getenv("PROCUREFLOW_ROOT", Path.cwd())))
