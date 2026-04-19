"""Experiment runners and metrics."""

from src.experiments.alpha_beta_trace import (
	AlphaBetaTraceMetrics,
	AlphaBetaTraceNode,
	AlphaBetaTraceResult,
	AlphaBetaTracer,
	build_alpha_beta_markdown,
	generate_alpha_beta_example,
	load_state_from_text_file,
	render_trace_tree,
	state_from_grid,
	trace_to_graphviz_dot,
	write_alpha_beta_markdown,
	write_graphviz_dot,
)
from src.experiments.metrics import MatchMetrics, TournamentMetrics


_LAZY_RUNNER_EXPORTS = {
	"RunnerConfig",
	"play_game",
	"play_match",
	"AgentExperimentConfig",
	"ExperimentScenario",
	"ExperimentMatchRecord",
	"ExperimentSummary",
	"ExperimentSuiteOutput",
	"run_experiment_scenario",
	"summarize_experiment_scenario",
	"run_experiment_suite",
	"save_experiment_outputs",
	"print_experiment_summary_tables",
	"generate_summary_plots",
	"build_default_experiment_scenarios",
	"build_and_run_default_experiment_suite",
}


def __getattr__(name: str):
	if name in _LAZY_RUNNER_EXPORTS:
		from src.experiments import runner as runner_module

		return getattr(runner_module, name)

	raise AttributeError(f"module 'src.experiments' has no attribute '{name}'")

__all__ = [
	"MatchMetrics",
	"TournamentMetrics",
	"RunnerConfig",
	"play_game",
	"play_match",
	"AgentExperimentConfig",
	"ExperimentScenario",
	"ExperimentMatchRecord",
	"ExperimentSummary",
	"ExperimentSuiteOutput",
	"run_experiment_scenario",
	"summarize_experiment_scenario",
	"run_experiment_suite",
	"save_experiment_outputs",
	"print_experiment_summary_tables",
	"generate_summary_plots",
	"build_default_experiment_scenarios",
	"build_and_run_default_experiment_suite",
	"AlphaBetaTraceMetrics",
	"AlphaBetaTraceNode",
	"AlphaBetaTraceResult",
	"AlphaBetaTracer",
	"state_from_grid",
	"load_state_from_text_file",
	"render_trace_tree",
	"build_alpha_beta_markdown",
	"write_alpha_beta_markdown",
	"trace_to_graphviz_dot",
	"write_graphviz_dot",
	"generate_alpha_beta_example",
]
