"""Audit saved evaluation evidence and summarize stages over all configured trials."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = ('detection', 'mask_quality', 'candidates', 'reachable', 'execution', 'stable_lift', 'task')


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit_hashes(base, manifest):
    """Missing evidence never constitutes a vacuous pass."""
    if not manifest:
        return {'status': 'not_recorded', 'files': 0, 'issues': []}
    issues = []
    for relative, digest in manifest.items():
        path = (base / relative).resolve()
        if not path.is_relative_to(base.resolve()):
            issues.append({'path': str(path), 'reason': 'outside_manifest_root'})
        elif not path.is_file():
            issues.append({'path': str(path), 'reason': 'missing'})
        elif sha256(path) != digest:
            issues.append({'path': str(path), 'reason': 'sha256_mismatch'})
    return {'status': 'failed' if issues else 'passed', 'files': len(manifest), 'issues': issues}


def bool_or_unknown(value):
    return value if isinstance(value, bool) else None


def cohort(row, result):
    grounding = row.get('grounding') or {}
    backend = row.get('mask_backend', result.get('mask_backend'))
    if backend == 'gt' or 'oracle' in Path(grounding.get('mask_source', '')).name:
        return 'gt_diagnostic'
    if backend == 'dino' or (grounding.get('live_language_detection') is True and
                            (grounding.get('detection') or grounding).get('backend') == 'dino'):
        return 'real_language'
    return 'unverified_saved_or_unknown'


def provenance_audit(grounding):
    if grounding.get('backend') != 'saved_verified_perception':
        return {'status': 'not_recorded', 'issues': []}
    issues = []
    for key, digest_key in [('mask_source', 'mask_sha256'), ('provenance_file', 'provenance_sha256')]:
        path = Path(grounding.get(key, ''))
        if not grounding.get(key) or not grounding.get(digest_key):
            issues.append(f'missing_{key}_or_hash')
        elif not path.is_file() or sha256(path) != grounding[digest_key]:
            issues.append(f'{key}_missing_or_hash_mismatch')
    source = read_json(Path(grounding.get('provenance_file', '')), {})
    for key, source_key in [('detection', 'target_detection'), ('segmentation', 'target_segmentation'),
                            ('mask_refinement', 'target_quality')]:
        if not grounding.get(key) or grounding[key] != source.get(source_key):
            issues.append(f'{key}_source_mismatch')
    if grounding.get('mask_sha256') != source.get('target_mask_sha256'):
        issues.append('source_mask_hash_mismatch')
    mask, source_path = Path(grounding.get('mask_source', '')), Path(grounding.get('provenance_file', ''))
    if mask.name != 'target_mask.png' or mask.parent.resolve() != source_path.parent.resolve():
        issues.append('mask_and_perception_are_not_same_view')
    if grounding.get('instruction') != (grounding.get('detection') or {}).get('instruction'):
        issues.append('detection_instruction_mismatch')
    if (grounding.get('detection') or {}).get('backend') != 'dino' or grounding.get('live_language_detection') is not True:
        issues.append('live_dino_detection_not_recorded')
    return {'status': 'failed' if issues else 'passed', 'issues': issues}


def stage_metrics(rows):
    n = len(rows)
    return {stage: {'successes': sum(row['stages'][stage] is True for row in rows),
                    'denominator': n,
                    'rate': sum(row['stages'][stage] is True for row in rows) / n if n else None,
                    'failed_or_unreached': sum(row['stages'][stage] is False for row in rows),
                    'unverified': sum(row['stages'][stage] is None for row in rows)} for stage in STAGES}


def summarize_row(directory, scene, row, result):
    trial = directory / scene / f"trial_{row['trial']:02d}_seed{row['seed']}"
    view = Path(row.get('selected_view') or trial)
    if not view.is_absolute():
        view = directory / view
    perception = read_json(view / 'perception.json', {})
    # Perception moves with the selected scan view; proposals stay at trial root.
    proposal = read_json(trial / 'proposals' / 'result.json', {})
    grounding = row.get('grounding') or proposal.get('grounding') or {}
    detection = grounding.get('detection') or (grounding if grounding.get('backend') == 'dino' else {})
    detection = detection or perception.get('target_detection', {})
    quality = grounding.get('mask_refinement') or perception.get('target_quality', {})
    segmentation = grounding.get('segmentation') or perception.get('target_segmentation', {})
    trace = read_json(trial / 'execution_trace.json', [])
    kind = cohort(row, result)
    provenance = provenance_audit(grounding)
    if grounding.get('backend') == 'saved_verified_perception' and proposal.get('grounding') != grounding:
        provenance['status'] = 'failed'
        provenance['issues'].append('row_and_proposal_grounding_mismatch')
    detected = bool_or_unknown(detection.get('detected')) if detection.get('backend') == 'dino' else None
    if kind != 'real_language' or provenance['status'] == 'failed':
        detected = None
    mask_ok = bool_or_unknown(quality.get('accepted'))
    if segmentation.get('accepted') is False:
        mask_ok = False
    if provenance['status'] == 'failed':
        mask_ok = None
    counts = row.get('counts', proposal.get('counts', {}))
    candidate_ok = counts['exported'] > 0 if 'exported' in counts else None
    attempts = row.get('ik_attempts')
    chosen = row.get('chosen') or {}
    # Chosen proves pregrasp/grasp IK selection, not collision checking.
    reachable = True if chosen else (False if attempts else None)
    if row.get('failure_stage') in ('no_valid_ik', 'no_reachable_candidate'):
        reachable = False
    stages = dict(zip(STAGES, (detected, mask_ok, candidate_ok, reachable,
                              bool_or_unknown(row.get('robot_executed')),
                              bool_or_unknown(row.get('grasp_lift_success')), bool_or_unknown(row.get('success')))))
    for previous, following in [('detection', 'mask_quality'), ('mask_quality', 'candidates'), ('candidates', 'reachable')]:
        if stages[previous] is False and stages[following] is None:
            stages[following] = False
    item = {key: row.get(key) for key in ('seed', 'trial', 'success', 'grasp_lift_success', 'failure_stage',
                                        'max_rise_m', 'terminal', 'instruction', 'target_key')}
    item.update(evidence=str(trial), selected_view=str(view), cohort=kind, stages=stages, counts=counts,
                grounding_backend=grounding.get('backend'), detection_backend=detection.get('backend'),
                segmentation_backend=segmentation.get('backend', row.get('segmentation_backend', result.get('segmentation_backend', 'not_recorded'))),
                quality_method=quality.get('method') or (quality.get('sphere') or {}).get('method'),
                provenance_audit=provenance, collision_check=row.get('collision_check', result.get('collision_check', 'not_recorded')),
                collision_evidence=row.get('collision_evidence'),
                max_held_lift_run_steps=max((sample.get('held_lift_run', 0) for sample in trace), default=0) if trace else None,
                checkpoint=proposal.get('checkpoint'), checkpoint_sha256=proposal.get('checkpoint_sha256'))
    generic_lift_only = scene == 'object-single' and row.get('score_schema') == 'target_contact_lift_8cm_5steps_v1'
    if generic_lift_only:
        stages['task'] = None
        item.update(score_schema=row['score_schema'], place_evaluated=False, place_success=None,
                    original_task_success=None,
                    generic_grasp_lift_success=row.get('generic_grasp_lift_success'),
                    generic_end_hold_success=row.get('generic_end_hold_success'))
    elif row.get('success') is True and not (row.get('terminal') or {}).get('success'):
        item['scoring_issue'] = 'task_success_without_terminal_success'
        stages['task'] = None
    # Truth is used only in an explicitly separate post-execution diagnostic.
    objects = read_json(view / 'evaluation_state.json', {}).get('objects', {})
    target = row.get('target_key', (result.get('language') or {}).get('target_key'))
    if target is None and row.get('instruction') == 'orange':
        target = 'orange'  # Old schema was orange-only.
    diagnostic = {}
    if chosen.get('world_pocket_m') is not None and target in objects:
        diagnostic['initial_grasp_point_error_m'] = math.dist(chosen['world_pocket_m'], objects[target])
    if row.get('destination_world_m') is not None and 'cardboard_box' in objects:
        diagnostic['box_estimate_xy_error_m'] = math.dist(row['destination_world_m'][:2], objects['cardboard_box'][:2])
    if diagnostic:
        item['post_execution_gt_diagnostics'] = diagnostic
    scan = read_json(trial / 'scan.json')
    if scan:
        item['scan_views'] = len(scan['views'])
    return item


def summarize(directory):
    directory = Path(directory).resolve()
    run = read_json(directory / 'run.json')
    if run is None:
        raise ValueError(f'Missing run.json: {directory}')
    role = run.get('evaluation_role', 'historical')
    if role not in ('smoke', 'regression', 'holdout', 'historical'):
        raise ValueError(f'Unknown evaluation_role: {role}')
    seeds = run['seeds']
    if len(set(seeds)) != len(seeds):
        raise ValueError('Duplicate configured seeds')
    audits = {'source_snapshot': audit_hashes(directory / 'source', run.get('source_sha256')),
              'assets': audit_hashes(ROOT, run.get('asset_sha256')),
              'original_rl_sources': audit_hashes(ROOT.parent, read_json(ROOT / 'rl_source_before.json'))}
    model = run.get('segmentation_model') or {}
    audits['segmentation_model'] = audit_hashes(Path(model.get('path', '.')), model.get('files_sha256'))
    report = {'input': str(directory), 'evaluation_role': role, 'seeds': seeds, 'language': run.get('language'),
              'segmentation_backend': run.get('segmentation_backend', 'not_recorded'),
              'collision_check': run.get('collision_check', 'not_recorded'), 'segmentation_model': model or None,
              'audits': audits, 'groups': {},
              'notes': ['All configured seeds remain in each scene denominator, including missing results.',
                        'Unknown/unreached stages contribute no successes; unknown counts are explicit.',
                        'Detection/mask acceptance is model/control evidence, not independent localization accuracy.',
                        'Candidates and IK reachability do not prove collision-free execution.',
                        'GT diagnostics and unverified masks are excluded from real-language cohort metrics.',
                        'GT distances are post-execution diagnostics only.',
                        'evaluation_role is declared metadata; seed independence is not independently verified.']}
    checkpoints = {}
    for scene in run['scenes']:
        result = read_json(directory / scene / 'results.json', {})
        source_rows = result.get('rows', [])
        observed_seeds = [row['seed'] for row in source_rows]
        if len(set(observed_seeds)) != len(observed_seeds) or not set(observed_seeds) <= set(seeds):
            raise ValueError(f'{scene}: duplicate/unconfigured row seeds')
        if result and result.get('episodes') != len(source_rows):
            raise ValueError(f'{scene}: episode count differs from recorded rows')
        if result.get('successes', 0) != sum(row.get('success') is True for row in source_rows):
            raise ValueError(f'{scene}: success count differs from recorded rows')
        rows = [summarize_row(directory, scene, row, result) for row in source_rows]
        for index, seed in enumerate(seeds):
            if seed not in observed_seeds:
                rows.append({'seed': seed, 'trial': index, 'success': None, 'grasp_lift_success': None,
                             'failure_stage': 'missing_result', 'cohort': cohort({}, result),
                             'stages': {stage: None for stage in STAGES}})
        rows.sort(key=lambda row: seeds.index(row['seed']))
        group = {key: result.get(key, 'not_recorded') for key in ('controller', 'grasp_point', 'contact_frame', 'mask_backend', 'cuda_visible_devices', 'metric')}
        metrics = stage_metrics(rows)
        group.update(episodes=len(seeds), observed_episodes=len(source_rows), successes=metrics['task']['successes'],
                     success_rate=metrics['task']['rate'], grasp_lift_successes=metrics['stable_lift']['successes'], stages=metrics,
                     failure_counts=dict(Counter(row['failure_stage'] or 'unspecified' for row in rows if row['success'] is not True)),
                     cohorts={kind: {'episodes': len(subset), 'stages': stage_metrics(subset)}
                              for kind in ('real_language', 'gt_diagnostic', 'unverified_saved_or_unknown')
                              if (subset := [row for row in rows if row['cohort'] == kind])}, rows=rows)
        if scene == 'object-single':
            group.update(score_schema=result.get('score_schema'), place_evaluated=False,
                         original_task_success=None, place_success=None,
                         generic_grasp_lift_successes=sum(row.get('generic_grasp_lift_success') is True for row in rows),
                         generic_end_hold_successes=sum(row.get('generic_end_hold_success') is True for row in rows))
        report['groups'][scene] = group
        for row in rows:
            if row.get('checkpoint') and row.get('checkpoint_sha256'):
                path, digest = str(Path(row['checkpoint']).resolve()), row['checkpoint_sha256']
                if path in checkpoints and checkpoints[path] != digest:
                    raise ValueError('Conflicting checkpoint hashes for the same path')
                checkpoints[path] = digest
                if result.get('checkpoint_sha256') and result['checkpoint_sha256'] != digest:
                    raise ValueError(f'{scene}: result/proposal checkpoint hash mismatch')
    audits['grasp_checkpoints'] = audit_hashes(ROOT.parent, checkpoints)
    provenance = [row['provenance_audit'] for group in report['groups'].values() for row in group['rows'] if 'provenance_audit' in row]
    audits['saved_perception_provenance'] = {'status': 'failed' if any(x['status'] == 'failed' for x in provenance)
                                           else 'passed' if provenance and all(x['status'] == 'passed' for x in provenance)
                                           else 'not_recorded',
                                           'verified_trials': sum(x['status'] == 'passed' for x in provenance),
                                           'unverified_trials': sum(x['status'] == 'not_recorded' for x in provenance),
                                           'issues': [x for x in provenance if x['status'] == 'failed']}
    scoring_issues = [row['scoring_issue'] for group in report['groups'].values() for row in group['rows'] if 'scoring_issue' in row]
    scored = any(row['stages']['task'] is not None for group in report['groups'].values() for row in group['rows'])
    audits['task_scoring'] = {'status': 'failed' if scoring_issues else 'passed' if scored else 'not_recorded',
                            'issues': scoring_issues}
    report['audit_status'] = 'failed' if any(a['status'] == 'failed' for a in audits.values()) else 'passed_recorded_checks_only'
    for name, key in [('source_snapshot_verified', 'source_snapshot'), ('asset_hashes_verified', 'assets'), ('original_rl_sources_unchanged', 'original_rl_sources')]:
        report[name] = {'passed': True, 'failed': False, 'not_recorded': None}[audits[key]['status']]
    return report


def markdown_report(report):
    lines = ['# 语言抓取评估证据汇总', '', f"评估角色：`{report['evaluation_role']}`；seeds：`{report['seeds']}`。",
             f"分割配置：`{report['segmentation_backend']}`；整臂碰撞配置：`{report['collision_check']}`。", '',
             '所有配置回合计入分母；缺失或未验证阶段不计成功。掩码质量是程序接受判据，不是人工标注准确率。', '',
             '| 场景 | 证据类别 | 回合数 | 检测 | 掩码有效 | 候选 | IK可达 | 已执行 | 稳定抓起 | 原任务 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for scene, group in report['groups'].items():
        for kind, values in group['cohorts'].items():
            cells = [f"{values['stages'][stage]['successes']}/{values['episodes']} (未知{values['stages'][stage]['unverified']})" for stage in STAGES]
            lines.append(f"| {scene} | {kind} | {values['episodes']} | " + ' | '.join(cells) + ' |')
    lines += ['', 'GT 诊断与未验证保存掩码独立列出，不并入真实语言成功率。IK 可达不等同于整臂碰撞检查通过。', '', '各场景记录的评分与控制配置：', '']
    for scene, group in report['groups'].items():
        lines.append(f"- {scene}: metric=`{group['metric']}`, controller=`{group['controller']}`, grasp_point=`{group['grasp_point']}`, contact_frame=`{group['contact_frame']}`；已保存{group['observed_episodes']}/{group['episodes']}回合。")
    role_note = {'smoke': '本轮为冒烟，不能作为统计可靠性或留出集泛化结论。',
                 'regression': '本轮标记为回归，不作为独立留出集泛化结论。',
                 'holdout': '本轮标记为留出集；本汇总器未独立核验种子是否参与过调参。',
                 'historical': '历史记录未声明评估角色，不擅自归类为回归或留出集。'}
    lines += ['', role_note[report['evaluation_role']], '', '## 完整性检查', '', '| 检查 | 状态 |', '|---|---|']
    for key, audit in report['audits'].items():
        lines.append(f"| {key} | {audit['status']} |")
    lines += ['', 'passed 仅表示已执行的检查通过；not_recorded 表示缺证据。源快照哈希只核对记录中的文件。各项异常见 SUMMARY.json。',
              '', '## 逐回合结果', '', '| 场景 | seed | 类别 | 原任务 | 稳定抓起 | 失败阶段 |', '|---|---:|---|---|---|---|']
    for scene, group in report['groups'].items():
        for row in group['rows']:
            lines.append(f"| {scene} | {row['seed']} | {row['cohort']} | {row['success']} | {row['grasp_lift_success']} | {row['failure_stage'] or '—'} |")
    lines += ['', '逐回合阶段、来源、碰撞证据、事后真值诊断及文件路径见 SUMMARY.json。']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT):
        parser.error('output must stay inside small_grasp')
    report = summarize(args.input)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'SUMMARY.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    (output / 'REPORT.md').write_text(markdown_report(report))
    print(json.dumps({'evaluation_role': report['evaluation_role'], 'audit_status': report['audit_status'],
                      'groups': {scene: {key: group[key] for key in ('episodes', 'successes', 'grasp_lift_successes', 'failure_counts')}
                                 for scene, group in report['groups'].items()}}, ensure_ascii=False))
    if report['audit_status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
