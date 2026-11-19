import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision
import torchxrayvision as xrv

ROOT = Path(__file__).resolve().parents[1]


def notebook_source(path):
    notebook = json.loads(path.read_text(encoding='utf-8'))
    return '\n'.join(
        line for cell in notebook['cells'] if cell['cell_type'] == 'code'
        for line in ''.join(cell['source']).splitlines()
        if not line.startswith(('!', '%'))
    )


@pytest.mark.parametrize('path', sorted((ROOT / 'notebooks').glob('*.ipynb')), ids=lambda p: p.name)
def test_notebook_code_compiles(path):
    compile(notebook_source(path), str(path), 'exec')


@pytest.mark.parametrize('relative_path', [
    'src/image_utils.py',
    'notebooks/gradio_interface_prediction_and_explainability_v1.py',
    'notebooks/gradio_interface_prediction_and_explainability_v1.ipynb',
])
@pytest.mark.parametrize('shape', [(32, 32), (32, 32, 3)])
def test_notebook_prediction_accepts_grayscale_and_color(relative_path, shape, monkeypatch):
    path = ROOT / relative_path
    source = notebook_source(path) if path.suffix == '.ipynb' else path.read_text()
    definitions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)]
    namespace = {'np': np, 'torch': torch, 'torchvision': torchvision, 'xrv': xrv}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)

    class Model(torch.nn.Module):
        pathologies = ['test']

        def forward(self, image):
            assert not self.training
            assert image.shape == (1, 1, 224, 224)
            return torch.tensor([[0.75]])

    monkeypatch.setattr(xrv.models, 'DenseNet', lambda **kwargs: Model())
    assert namespace['predict'](np.full(shape, 128, dtype=np.uint8), 'test-model') == {'test': 0.75}
