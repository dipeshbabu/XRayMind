import json
import runpy
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from xraymind import cli, explainability, inference

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def research_images(tmp_path, monkeypatch):
    class SmallModel(torch.nn.Module):
        pathologies = ['Atelectasis', 'Cardiomegaly']

        def forward(self, image):
            score = torch.sigmoid(image.mean(dim=(1, 2, 3)) / 1024)
            return torch.stack([score, 1 - score], dim=1)

    model = SmallModel().eval()
    monkeypatch.setattr(inference, 'load_model', lambda name: model)
    monkeypatch.setattr(explainability, 'load_model', lambda name: model)
    for index in range(4):
        pixels = np.full((24, 32), 40 + index * 50, dtype=np.uint8)
        pixels[8:16, 8:16] += 20
        Image.fromarray(pixels).save(tmp_path / f'{index}.png')
    pd.DataFrame({'image': [f'{i}.png' for i in range(4)], 'Atelectasis': [0, 0, 1, 1],
                  'Cardiomegaly': [1, 1, 0, 0], 'sex': ['F', 'M', 'F', 'M']}).to_csv(tmp_path / 'labels.csv', index=False)
    return tmp_path


@pytest.mark.parametrize('method', ['saliency', 'input_x_gradient', 'integrated_gradients'])
def test_attribution_methods_produce_finite_heatmaps(research_images, method):
    result = explainability.compute_attribution(research_images / '0.png', 'Atelectasis', 'test', method=method)
    assert result.shape == (224, 224)
    assert np.isfinite(result).all()
    assert result.min() >= 0 and result.max() <= 1


def test_cli_prediction_report_packet_and_tta(research_images):
    root = research_images
    image = str(root / '0.png')
    assert cli.main(['predict', '--image', image, '--out', str(root / 'prediction.json'), '--html', str(root / 'prediction.html')]) == 0
    result = json.loads((root / 'prediction.json').read_text())
    assert len(result['predictions']) == 2
    assert cli.main(['report', '--image', image, '--json', str(root / 'report.json'), '--html', str(root / 'report.html'),
                     '--original', str(root / 'original.png'), '--heatmap', str(root / 'heatmap.png'), '--method', 'saliency']) == 0
    assert (root / 'heatmap.png').is_file()
    assert "src='original.png'" in (root / 'report.html').read_text()
    assert cli.main(['tta', '--image', image, '--out', str(root / 'tta.json')]) == 0
    assert json.loads((root / 'tta.json').read_text())['num_augmentations'] == 4
    assert cli.main(['packet', '--image', image, '--out-dir', str(root / 'packet'), '--method', 'saliency']) == 0
    with ZipFile(root / 'packet.zip') as archive:
        assert {'prediction.json', 'report.html', 'original_preview.png', 'overlay.png', 'side_by_side.png', 'manifest.json'} <= set(archive.namelist())
        assert json.loads(archive.read('manifest.json')) == json.loads((root / 'packet/manifest.json').read_text())
        assert "src='side_by_side.png'" in archive.read('report.html').decode()


def run_script(monkeypatch, name, args):
    path = ROOT / 'scripts' / name
    monkeypatch.setattr(sys, 'argv', [str(path), *map(str, args)])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(path), run_name='__main__')
    assert result.value.code == 0


def test_benchmark_selective_conformal_pipeline(research_images, monkeypatch):
    root = research_images
    benchmark = root / 'benchmark'
    run_script(monkeypatch, 'benchmark_models.py', [
        '--image-dir', root, '--labels', root / 'labels.csv', '--out-dir', benchmark,
        '--subgroups', 'sex', '--min-group-size', '2', '--save-plots', '--selective', '--bootstrap', '5',
    ])
    manifest = json.loads((benchmark / 'run_manifest.json').read_text())
    assert Path(manifest['outputs']['leaderboard']).is_file()
    metrics = pd.read_csv(manifest['outputs']['combined_metrics'])
    assert set(metrics['label']) == {'Atelectasis', 'Cardiomegaly'}
    predictions = next(iter(manifest['outputs']['predictions'].values()))
    run_script(monkeypatch, 'selective_prediction.py', [
        '--labels', root / 'labels.csv', '--predictions', predictions, '--subgroups', 'sex',
        '--out-dir', root / 'selective', '--coverage-grid', '0.5', '1.0',
    ])
    assert set(pd.read_csv(root / 'selective/selective_summary.csv')['coverage']) == {0.5, 1.0}
    run_script(monkeypatch, 'conformal_predict.py', [
        '--labels', root / 'labels.csv', '--predictions', predictions, '--out-dir', root / 'conformal',
    ])
    assert set(pd.read_csv(root / 'conformal/conformal_summary.csv')['label']) == {'Atelectasis', 'Cardiomegaly'}
