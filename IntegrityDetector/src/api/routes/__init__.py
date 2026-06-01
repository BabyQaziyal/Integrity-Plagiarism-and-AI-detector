"""Flask blueprints."""
from src.api.routes.students import bp as students_bp
from src.api.routes.submissions import bp as submissions_bp

ALL_BLUEPRINTS = [students_bp, submissions_bp]
