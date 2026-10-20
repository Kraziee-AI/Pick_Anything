"""Audit frozen 10-trial evaluations and the separate 3-seed failure controls."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import backproject

ROOT = Path(__file__).resolve().parent


def main():
    output = ROOT / 'outputs/20261015_failure_diagnosis'
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Refusing to overwrite an existing diagnosis report')
    directories = {'dino_first_contact': ROOT / 'outputs/20261015_eval10_fixed',
                   'gt_first_contact': ROOT / 'outputs/20261015_diag_gt_first',
                   'gt_full_close': ROOT / 'outputs/20261015_diag_gt_full',
                   'dino_full_close': ROOT / 'outputs/20261015_diag_dino_full'}
    report = {'groups': {}, 'pair_checks': [], 'perception_checks': [], 'placement_checks': [],
              'original_rl_hashes_unchanged': False,
              'notes': ['Full task success and stable held lift 8cm for 5 steps are separate metrics.',
                        'Only dino_first_contact has 10 trials per scene; controls use seeds42/43/44.',
                        'GT is an oracle diagnostic, not language-driven performance.',
                        'Each trial has one plan and one chosen grasp, without retries.']}
    all_rows = {}
    for name, directory in directories.items():
        metadata = json.loads((directory / 'run.json').read_text())
        for relative, digest in metadata['source_sha256'].items():
            assert hashlib.sha256((directory / 'source' / relative).read_bytes()).hexdigest() == digest
        report['groups'][name] = {}
        for scene in ('single', 'multi'):
            result = json.loads((directory / scene / 'results.json').read_text())
            expected = 10 if name == 'dino_first_contact' else 3
            assert result['episodes'] == expected
            assert len({row['seed'] for row in result['rows']}) == expected
            assert result['checkpoint_sha256'] == '60680087c61cba2b6791614fef1519071e294f6dcaf99b3f581bb95f7c51a868'
            assert result['cuda_visible_devices'] == '0'
            all_rows[name, scene] = result['rows']
            summary = {key: result[key] for key in ('episodes', 'successes', 'grasp_lift_successes')}
            summary['failure_counts'] = dict(Counter(row['failure_stage'] for row in result['rows']))
            summary['rows'] = []
            for row in result['rows']:
                trial = directory / scene / f"trial_{row['trial']:02d}_seed{row['seed']}"
                trace = json.loads((trial / 'execution_trace.json').read_text()) if (trial / 'execution_trace.json').exists() else []
                max_run = max((sample['held_lift_run'] for sample in trace), default=0)
                summary['rows'].append({'seed': row['seed'], 'failure_stage': row['failure_stage'],
                                       'grasp_lift_success': row['grasp_lift_success'],
                                       'max_rise_m': row.get('max_rise_m'), 'max_held_8cm_run_steps': max_run,
                                       'candidate_count': row['counts']['exported'], 'evidence': str(trial)})
            report['groups'][name][scene] = summary
    for scene in ('single', 'multi'):
        for trial, seed in enumerate((42, 43, 44)):
            label = f'trial_{trial:02d}_seed{seed}'
            paths = [directories[group] / scene / label for group in ('gt_first_contact', 'gt_full_close')]
            states = [json.loads((path / 'evaluation_state.json').read_text()) for path in paths]
            joint_delta = float(np.max(np.abs(np.array(states[0]['joint_position']) - states[1]['joint_position'])))
            object_deltas = {key: float(np.max(np.abs(np.array(states[0]['objects'][key]) - states[1]['objects'][key])))
                             for key in states[0]['objects']}
            object_delta = max(object_deltas.values())
            assert joint_delta < 1e-5 and object_deltas['orange'] < 1e-5 and object_deltas['cardboard_box'] < 1e-5
            rows = [all_rows[group, scene][trial] for group in ('gt_first_contact', 'gt_full_close')]
            target_delta = None
            if all(row.get('chosen') for row in rows):
                target_delta = float(np.max(np.abs(np.array(rows[0]['chosen']['world_pocket_m']) - rows[1]['chosen']['world_pocket_m'])))
                assert target_delta < 1e-5
            report['pair_checks'].append({'scene': scene, 'seed': seed, 'max_joint_delta_rad': joint_delta,
                                          'max_object_delta_m': object_delta, 'per_object_delta_m': object_deltas,
                                          'selected_point_delta_m': target_delta})
            mask = np.array(Image.open(paths[0] / 'oracle_mask.png')) > 0
            dino = np.array(Image.open(directories['dino_first_contact'] / scene / label / 'proposals/target_mask.png')) > 0
            union = np.logical_or(mask, dino).sum()
            row = {'scene': scene, 'seed': seed, 'oracle_pixels': int(mask.sum()),
                   'dino_oracle_box_mask_iou': float(np.logical_and(mask, dino).sum() / union) if union else None}
            if not mask.any():
                transform = np.array(json.loads((paths[0] / 'world_from_camera.json').read_text()))
                camera = (np.array(states[0]['objects']['orange']) - transform[:3, 3]) @ transform[:3, :3]
                k = json.loads((paths[0] / 'intrinsics.json').read_text())
                row['orange_projected_centre_px'] = [float(camera[0] / camera[2] * k['fx'] + k['cx']),
                                                       float(camera[1] / camera[2] * k['fy'] + k['cy'])]
            report['perception_checks'].append(row)
    for group in ('gt_full_close', 'dino_full_close'):
        for row in all_rows[group, 'single']:
            if row['failure_stage'] != 'transport_ik':
                continue
            trial = directories[group] / 'single' / f"trial_{row['trial']:02d}_seed{row['seed']}"
            detection = json.loads((trial / 'box_detection.json').read_text())
            k = json.loads((trial / 'intrinsics.json').read_text())
            depth = np.load(trial / 'depth.npy')
            cloud, valid = backproject(depth, k)
            transform = np.array(json.loads((trial / 'world_from_camera.json').read_text()))
            mask = np.zeros(depth.shape, dtype=bool)
            x0, y0 = np.floor(detection['box_xyxy'][:2]).astype(int)
            x1, y1 = np.ceil(detection['box_xyxy'][2:]).astype(int)
            mask[max(0, y0):min(depth.shape[0], y1), max(0, x0):min(depth.shape[1], x1)] = True
            points = cloud[valid & mask] @ transform[:3, :3].T + transform[:3, 3]
            median = np.median(points, axis=0)
            state = json.loads((trial / 'evaluation_state.json').read_text())
            report['placement_checks'].append({'group': group, 'seed': row['seed'],
                'box_crop_median_world_m': median.tolist(), 'true_box_root_world_m': state['objects']['cardboard_box'],
                'commanded_transport_pocket_world_m': (median + [0, 0, .1]).tolist(),
                'failure_stage': row['failure_stage']})
    hashes = json.loads((ROOT / 'rl_source_before.json').read_text())
    assert all(hashlib.sha256((ROOT.parent / path).read_bytes()).hexdigest() == digest for path, digest in hashes.items())
    report['original_rl_hashes_unchanged'] = True
    (output / 'SUMMARY.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    lines = ['# 独立GraspNet实际执行结果与失败对照', '',
             '物理GPU0，Conda isaaclab_61，固定epoch18 GraspNet，指令orange。', '',
             '## 正式固定配置：各10次', '',
             '| 场景 | 完整任务成功 | 稳定夹持抬升8cm | 失败阶段 |', '|---|---:|---:|---|']
    for scene in ('single', 'multi'):
        data = report['groups']['dino_first_contact'][scene]
        lines.append(f"| {scene} | {data['successes']}/10 | {data['grasp_lift_successes']}/10 | {data['failure_counts']} |")
    lines += ['', '单橘子完整任务要求原环境放盒/撤爪成功；多物体完整任务就是原环境正确',
              '夹持抬升8cm持续5步。每次一次规划与一个选中候选，无检测/无候选/不可达均为',
              '端到端失败；原PPO是连续控制，因此两种控制器不能当作同算法直接比较。', '',
              '## 配对诊断：seed42/43/44，各3次', '',
              '| 掩码与闭合方式 | 单橘子稳定抓起 | 单橘子放盒成功 | 多物体稳定抓起 |', '|---|---:|---:|---:|']
    for group in directories:
        a, b = report['groups'][group]['single'], report['groups'][group]['multi']
        first3 = all_rows[group, 'single'][:3]
        rows_multi = all_rows[group, 'multi'][:3]
        lines.append(f"| {group} | {sum(row['grasp_lift_success'] for row in first3)}/3 | {sum(row['success'] for row in first3)}/3 | {sum(row['grasp_lift_success'] for row in rows_multi)}/3 |")
    lines += ['', 'dino_first_contact取正式评估的前三次作为参考，不重复计为额外诊断回合；其余',
              '三组共额外18次。GT是明确标记的真值投影框，不是实际语言检测。', '',
              '## 已确认的原因与边界', '',
              '1. **首次接触停止影响夹持。** 只取消停止条件，GT与DINO单橘子稳定抬升均由',
              '   0/3变为1/3；GT seed44连续54步在夹持条件下升高超过8cm，最大10.07cm。',
              '   单橘子组初始关节/物体位置和选中抓取点误差均低于1e-5。持续闭合仍有失败，',
              '   也会发生掉落，不能据此称为已修复或推荐直接无条件大力夹紧。',
              '2. **感知与视野有独立问题。** 多物体seed43的DINO框与GT投影框IoU=0；GT能',
              '   提供可执行候选，但仍未抓稳。seed44橘子投影中心约(-124,-36)，在256×192',
              '   腕摄视野外，GT掩码为空，DINO仍给出了其他区域的框。换抓取模型不能补足',
              '   当前图像没有观测到的目标几何。',
              '3. **抓起后的放盒估计也失败。** GT持续闭合seed44已经稳定抓起，随后transport_ik',
              '   失败；盒子框内点云中位数世界Z约0.235m，而盒子根位置Z约0.041m。这个',
              '   中位数没有对应盒子，进而请求了约0.335m的搬运高度。DINO持续闭合成功抓起',
              '   的回合同样记录transport_ik失败。未做正确放盒目标替换对照，不宣称这能',
              '   单独解释全部搬运失败。', '',
              '本次未证明GraspNet全部候选都正确，未完成SO101夹指网格/抓取位姿/接触预压',
              '的独立消融。小样本对照用于定位原因，不能把1/3外推为稳定总体成功率。', '',
              '多物体配对的橘子、盒子和关节匹配，但小黄鸭静置位置存在最大约0.61mm差异，',
              '未将所有背景刚体声称为逐位相同；逐项误差保存在SUMMARY.json。', '',
              '## 建议修复顺序', '',
              '先补目标可见性检查与主动扫描、误框拒绝；再为SO101设计持续接触后的夹持',
              '保持控制；放盒阶段使用目标点云聚类/排除机械臂前景并验证可达性。继续使用',
              '固定模型和同一批种子评估各项修改，保持正式结果与诊断结果分开。', '',
              '## 证据', '',
              '各组source/与run.json保存实际源码快照和SHA；每trial有RGB-D、掩码、模型',
              '候选、episode.json与execution_trace.json。详细计数、54步抬升证据、初始状态',
              '和抓取点配对误差、掩码IoU及放盒点云估计见同目录SUMMARY.json。',
              '现有RL的96个Python文件SHA与新增小模型之前一致；原训练代码未改。']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({group: {scene: {key: data[key] for key in ('episodes', 'successes', 'grasp_lift_successes')}
                            for scene, data in scenes.items()} for group, scenes in report['groups'].items()}, indent=2))


if __name__ == '__main__':
    main()
