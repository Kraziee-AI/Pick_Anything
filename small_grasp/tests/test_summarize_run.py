"""CPU-only regression tests for saved evaluation audit and stage denominators."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from small_grasp import summarize_run as summary


class SummaryTests(unittest.TestCase):
    def setUp(self):
        (summary.ROOT / 'outputs').mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=summary.ROOT / 'outputs')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'evaluation'
        self.run.mkdir()
        self.trial = self.run / 'single' / 'trial_00_seed42'
        self.trial.mkdir(parents=True)
        (self.trial / 'proposals').mkdir()
        self.write(self.root / 'rl_source_before.json', {})
        self.root_patch = patch.object(summary, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.config = {'seeds': [42], 'scenes': ['single']}
        self.row = {'trial': 0, 'seed': 42, 'instruction': 'pear', 'target_key': 'pear',
                    'counts': {'exported': 3}, 'grounding': {'backend': 'dino', 'detected': True,
                    'instruction': 'pear', 'live_language_detection': True,
                    'mask_refinement': {'accepted': True, 'method': 'generic_observed_object'}},
                    'success': True, 'terminal': {'success': True}, 'grasp_lift_success': True,
                    'robot_executed': True, 'failure_stage': None, 'chosen': {'world_pocket_m': [1, 2, 3]}}

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value))

    def evaluate(self, rows=None, **extra):
        rows = [self.row] if rows is None else rows
        self.write(self.run / 'run.json', self.config)
        self.write(self.run / 'single' / 'results.json', dict(episodes=len(rows),
                   successes=sum(row.get('success') is True for row in rows), rows=rows, **extra))
        return summary.summarize(self.run)

    def verified_grounding(self):
        mask = self.trial / 'target_mask.png'
        mask.write_bytes(b'mask bytes')
        perception = {'target_detection': {'backend': 'dino', 'detected': True, 'instruction': 'pear'},
                      'target_segmentation': {'backend': 'sam', 'accepted': True},
                      'target_quality': {'accepted': True, 'method': 'generic_observed_object'},
                      'target_mask_sha256': summary.sha256(mask)}
        source = self.trial / 'perception.json'
        self.write(source, perception)
        grounding = {'backend': 'saved_verified_perception', 'instruction': 'pear',
                     'detection': perception['target_detection'], 'segmentation': perception['target_segmentation'],
                     'mask_refinement': perception['target_quality'], 'mask_source': str(mask),
                     'mask_sha256': summary.sha256(mask), 'provenance_file': str(source),
                     'provenance_sha256': summary.sha256(source), 'live_language_detection': True}
        self.row['grounding'] = grounding
        self.write(self.trial / 'proposals' / 'result.json', {'grounding': grounding})
        return grounding

    def test_historical_schema_and_no_invented_passes(self):
        report = self.evaluate()
        self.assertEqual(report['evaluation_role'], 'historical')
        self.assertIsNone(report['source_snapshot_verified'])
        self.assertEqual(report['audits']['segmentation_model']['status'], 'not_recorded')
        self.assertEqual(report['groups']['single']['stages']['task']['successes'], 1)
        text = summary.markdown_report(report)
        self.assertNotIn('橘子尺寸', text)
        self.assertIn('不擅自归类', text)

    def test_new_saved_sam_schema(self):
        self.verified_grounding()
        self.config.update(evaluation_role='smoke', segmentation_backend='sam', collision_check='off')
        report = self.evaluate()
        self.assertEqual(report['audits']['saved_perception_provenance']['status'], 'passed')
        self.assertEqual(report['groups']['single']['rows'][0]['segmentation_backend'], 'sam')
        self.assertTrue(all(x['successes'] == 1 for x in report['groups']['single']['stages'].values()))
        self.assertIn('本轮为冒烟', summary.markdown_report(report))

    def test_tampered_mask_invalidates_provenance_and_detection(self):
        self.verified_grounding()
        (self.trial / 'target_mask.png').write_bytes(b'changed mask')
        report = self.evaluate()
        self.assertEqual(report['audit_status'], 'failed')
        self.assertEqual(report['groups']['single']['stages']['detection']['unverified'], 1)
        self.assertEqual(report['groups']['single']['stages']['mask_quality']['unverified'], 1)

    def test_row_proposal_mismatch_fails(self):
        self.verified_grounding()
        self.row['grounding']['segmentation']['backend'] = 'box'
        report = self.evaluate()
        self.assertEqual(report['audits']['saved_perception_provenance']['status'], 'failed')

    def test_scanned_view_uses_trial_root_proposal(self):
        self.verified_grounding()
        view = self.trial / 'view_01'
        view.mkdir()
        for name in ('target_mask.png', 'perception.json'):
            (self.trial / name).rename(view / name)
        grounding = self.row['grounding']
        grounding.update(mask_source=str(view / 'target_mask.png'), provenance_file=str(view / 'perception.json'))
        self.write(self.trial / 'proposals' / 'result.json', {'grounding': grounding})
        self.row['selected_view'] = str(view)
        report = self.evaluate()
        self.assertEqual(report['audits']['saved_perception_provenance']['status'], 'passed')

    def test_all_seeds_remain_denominator_with_early_failure_and_missing_row(self):
        self.config['seeds'] = [42, 43, 44]
        bad = copy.deepcopy(self.row)
        bad.update(trial=1, seed=43, counts={'exported': 0}, chosen=None, robot_executed=False,
                   success=False, terminal=None, grasp_lift_success=False, failure_stage='empty_target_mask')
        bad['grounding']['detected'] = False
        bad['grounding']['mask_refinement']['accepted'] = False
        report = self.evaluate([self.row, bad])
        group = report['groups']['single']
        self.assertEqual(group['episodes'], 3)
        self.assertEqual(group['observed_episodes'], 2)
        self.assertEqual(group['stages']['task']['rate'], 1 / 3)
        self.assertEqual(group['stages']['reachable']['failed_or_unreached'], 1)
        self.assertEqual(group['stages']['reachable']['unverified'], 1)
        self.assertEqual(group['failure_counts']['missing_result'], 1)

    def test_gt_is_independent_of_real_language(self):
        gt = copy.deepcopy(self.row)
        gt.update(trial=1, seed=43, mask_backend='gt')
        self.config['seeds'] = [42, 43]
        report = self.evaluate([self.row, gt])
        cohorts = report['groups']['single']['cohorts']
        self.assertEqual(cohorts['real_language']['episodes'], 1)
        self.assertEqual(cohorts['gt_diagnostic']['episodes'], 1)
        self.assertEqual(cohorts['gt_diagnostic']['stages']['detection']['successes'], 0)

    def test_saved_detection_without_live_provenance_is_unverified(self):
        self.row['grounding'] = {'backend': 'saved', 'detection': {'backend': 'dino', 'detected': True},
                                 'live_language_detection': False}
        report = self.evaluate()
        self.assertIn('unverified_saved_or_unknown', report['groups']['single']['cohorts'])
        self.assertEqual(report['groups']['single']['stages']['detection']['successes'], 0)

    def test_capsule_failure_does_not_become_reachability_or_collision_pass(self):
        self.row.update(chosen=None, ik_attempts=[], robot_executed=False, success=False,
                        terminal=None, grasp_lift_success=False, failure_stage='collision_check_unavailable',
                        collision_check='capsule', collision_evidence={'error': 'soft limits'})
        group = self.evaluate()['groups']['single']
        self.assertEqual(group['stages']['reachable']['unverified'], 1)
        self.assertEqual(group['stages']['execution']['successes'], 0)
        self.assertEqual(group['rows'][0]['collision_evidence']['error'], 'soft limits')

    def test_source_asset_and_sam_hash_mismatches_are_explicit(self):
        source = self.run / 'source'
        source.mkdir()
        (source / 'a.py').write_text('source')
        (self.root / 'asset').write_text('asset')
        model = self.root / 'sam'
        model.mkdir()
        (model / 'model').write_bytes(b'weights')
        self.config.update(source_sha256={'a.py': summary.sha256(source / 'a.py')},
                           asset_sha256={'asset': summary.sha256(self.root / 'asset')},
                           segmentation_model={'path': str(model), 'files_sha256': {'model': summary.sha256(model / 'model')}})
        self.assertEqual(self.evaluate()['audits']['segmentation_model']['status'], 'passed')
        (source / 'a.py').write_text('tamper')
        (self.root / 'asset').write_text('tamper')
        (model / 'model').write_bytes(b'tamper')
        report = self.evaluate()
        for key in ('source_snapshot', 'assets', 'segmentation_model'):
            self.assertEqual(report['audits'][key]['status'], 'failed')

    def test_no_results_is_unverified_and_role_is_not_guessed(self):
        self.config['evaluation_role'] = 'holdout'
        report = self.evaluate([])
        self.assertEqual(report['audits']['task_scoring']['status'], 'not_recorded')
        self.assertEqual(report['groups']['single']['stages']['task']['unverified'], 1)
        self.assertIn('未独立核验', summary.markdown_report(report))

    def test_duplicate_seed_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.evaluate([self.row, self.row])

    def test_truth_diagnostics_do_not_change_scores(self):
        self.write(self.trial / 'evaluation_state.json', {'objects': {'pear': [100, 100, 100]}})
        group = self.evaluate()['groups']['single']
        self.assertGreater(group['rows'][0]['post_execution_gt_diagnostics']['initial_grasp_point_error_m'], 100)
        self.assertEqual(group['stages']['task']['successes'], 1)

    def test_success_without_terminal_is_not_a_verified_task_pass(self):
        self.row['terminal'] = None
        report = self.evaluate()
        self.assertEqual(report['audit_status'], 'failed')
        self.assertEqual(report['groups']['single']['stages']['task']['successes'], 0)


if __name__ == '__main__':
    unittest.main()
