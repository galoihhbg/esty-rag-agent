import os
os.environ["TESTING"] = "1"  # ensure main runs in test mode
import pytest
from main import PredictRequest

def test_parse_config_from_string():
    config_str = '[{"name":"Color","type":"text"},{"name":"Nickname","type":"text"}]'
    pr = PredictRequest(user_input="blue- 2022", config_json=config_str, color_list="Blue, White")
    assert isinstance(pr.config_json, list)
    assert pr.config_json[0]["name"] == "Color"
    assert pr.color_list == ["Blue", "White"]

def test_parse_config_from_list():
    config_list = [{"name":"Color","type":"text"},{"name":"Nickname","type":"text"}]
    pr = PredictRequest(user_input="blue- 2022", config_json=config_list, color_list=["Blue","White"])
    assert isinstance(pr.config_json, list)
    assert pr.color_list == ["Blue", "White"]

def test_invalid_config_raises():
    with pytest.raises(Exception):
        PredictRequest(user_input="x", config_json='{"name":"x"}', color_list="")