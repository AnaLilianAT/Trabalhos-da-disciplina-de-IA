import tempfile
import unittest
from pathlib import Path

from src.experiments.runner import (
    AgentExperimentConfig,
    ExperimentScenario,
    build_default_experiment_scenarios,
    run_experiment_scenario,
    save_experiment_outputs,
)


class TestExperimentRunner(unittest.TestCase):
    def _small_scenario(self, alternate_colors: bool = True) -> ExperimentScenario:
        return ExperimentScenario(
            scenario_name="unit_small_experiment",
            agent_a=AgentExperimentConfig.minimax(
                depth=2,
                evaluation_name="balanced",
                label="MinimaxTest",
            ),
            agent_b=AgentExperimentConfig.mcts(
                num_simulations=30,
                rollout_policy="random",
                label="MCTSTest",
            ),
            board_size=4,
            scoring_mode="standard",
            matches=2,
            alternate_colors=alternate_colors,
            base_seed=2026,
            max_turns=200,
        )

    def test_run_experiment_scenario_returns_records_and_summary(self) -> None:
        scenario = self._small_scenario()

        records, summary = run_experiment_scenario(scenario)

        self.assertEqual(len(records), scenario.matches)
        self.assertEqual(summary.matches, scenario.matches)
        self.assertEqual(summary.board_size, scenario.board_size)
        self.assertEqual(summary.scoring_mode, scenario.scoring_mode)
        self.assertAlmostEqual(
            summary.agent_a_win_rate + summary.agent_b_win_rate + summary.draw_rate,
            1.0,
            places=6,
        )
        self.assertGreaterEqual(summary.avg_time_per_match_seconds, 0.0)
        self.assertGreaterEqual(summary.avg_time_per_move_seconds, 0.0)

    def test_alternate_colors_between_matches(self) -> None:
        scenario = self._small_scenario(alternate_colors=True)

        records, _summary = run_experiment_scenario(scenario)

        self.assertEqual(records[0].agent_a_color, "BLACK")
        self.assertEqual(records[1].agent_a_color, "WHITE")

    def test_reproducible_with_controlled_seed(self) -> None:
        scenario = self._small_scenario(alternate_colors=True)

        records_a, summary_a = run_experiment_scenario(scenario)
        records_b, summary_b = run_experiment_scenario(scenario)

        compact_a = [
            (
                row.winner_label,
                row.agent_a_score,
                row.agent_b_score,
                row.turns,
                row.total_passes,
            )
            for row in records_a
        ]
        compact_b = [
            (
                row.winner_label,
                row.agent_a_score,
                row.agent_b_score,
                row.turns,
                row.total_passes,
            )
            for row in records_b
        ]

        self.assertEqual(compact_a, compact_b)
        self.assertEqual(summary_a.agent_a_wins, summary_b.agent_a_wins)
        self.assertEqual(summary_a.agent_b_wins, summary_b.agent_b_wins)
        self.assertEqual(summary_a.draws, summary_b.draws)

    def test_save_experiment_outputs_writes_csv_files(self) -> None:
        scenario = self._small_scenario()
        records, summary = run_experiment_scenario(scenario)

        with tempfile.TemporaryDirectory() as temp_dir:
            output = save_experiment_outputs(
                match_records=records,
                summaries=[summary],
                output_dir=temp_dir,
                generate_plots=False,
            )

            matches_path = Path(output.matches_csv_path)
            summary_path = Path(output.summary_csv_path)

            self.assertTrue(matches_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertGreater(matches_path.stat().st_size, 0)
            self.assertGreater(summary_path.stat().st_size, 0)

    def test_default_suite_contains_required_scenarios(self) -> None:
        scenarios = build_default_experiment_scenarios(
            matches_per_configuration=1,
            base_seed=42,
            board_sizes=(4, 6),
            scoring_modes=("standard", "weighted"),
        )

        names = {scenario.scenario_name for scenario in scenarios}
        required = {
            "minimax_balanced_vs_mcts_random",
            "minimax_balanced_vs_mcts_rollout_simple",
            "minimax_balanced_vs_mcts_rollout_successor_eval",
            "minimax_balanced_vs_mcts_rollout_topk",
            "mcts_random_vs_rollout_simple",
            "mcts_rollout_simple_vs_rollout_successor_eval",
            "mcts_rollout_successor_eval_vs_rollout_topk",
        }

        self.assertTrue(required.issubset(names))

        weighted_core = [
            scenario
            for scenario in scenarios
            if scenario.scoring_mode == "weighted"
            and scenario.scenario_name == "minimax_balanced_vs_mcts_random"
        ]
        self.assertTrue(weighted_core)

        board_sweep = [
            scenario
            for scenario in scenarios
            if scenario.scenario_name.startswith("board_size_sweep_")
        ]
        self.assertGreaterEqual(len(board_sweep), 2)


if __name__ == "__main__":
    unittest.main()
