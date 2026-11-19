import importlib.util
from pathlib import Path

import numpy as np


def test_gradio_builds_and_reviews_case(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('GRADIO_ANALYTICS_ENABLED', 'False')
    path = Path(__file__).resolve().parents[1] / 'app.py'
    spec = importlib.util.spec_from_file_location('xraymind_test_app', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        config = module.demo.get_config_file()
        assert config['components']
        assert config['dependencies']
        assert 'Upload' in module.run_xraymind(None, 'test', 5, False)[-1]
        from xraymind import cases
        def prediction(*args, **kwargs):
            return {'top_findings': [{'label': 'A', 'probability': 0.8}],
                    'uncertainty': {'max_probability': 0.8, 'low_confidence': False}}
        monkeypatch.setattr(cases, 'predict_image', prediction)
        created = module.create_review_case(np.zeros((8, 8), dtype=np.uint8), 'test', 5, 'routine', 'test')
        assert 'Created case #1' in created[0]
        reviewed = module.review_case(1, 'agree', 'reader', 'reviewed')
        assert 'Saved' in reviewed[0]
        assert 'reviews: 1' in module.load_case(1)[0]
    finally:
        module.demo.close()
