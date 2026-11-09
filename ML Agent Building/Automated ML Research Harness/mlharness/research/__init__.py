"""V2 research phase: four researchers, a shared critic, the revision loop, portfolio and gate."""
from .codegen import CodeEngineer
from .critic import ResearchCritic
from .gate import ImplementationGate
from .loop import ResearchLoop
from .portfolio import baseline_methods, select_portfolio
from .proposals import check_prediction, claim_hygiene, completeness, normalize_proposal
from .researchers import ROLES, ResearcherAgent, method_menu

__all__ = ["CodeEngineer", "ImplementationGate", "ROLES", "ResearchCritic", "ResearchLoop", "ResearcherAgent",
           "baseline_methods", "check_prediction", "claim_hygiene", "completeness", "method_menu",
           "normalize_proposal", "select_portfolio"]
