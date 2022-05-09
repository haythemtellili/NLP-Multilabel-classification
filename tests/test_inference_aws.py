import pytest
import sys
import json
sys.path.insert(0, "src")

import inference_aws

def test_inference_aws():
    # single url
    request_body = json.dumps({"url":"https://dictionnaire.reverso.net/francais-arabe/"})
    # load model
    model_artifacts = inference_aws.model_fn('models')

    data, mask = inference_aws.input_fn(request_body, 'application/json')

    output = inference_aws.predict_fn((data, mask), model_artifacts)

    assert output is not None

