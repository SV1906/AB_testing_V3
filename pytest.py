import pytest
from calculator import Flask_App, base_data, basic_Sample_Size, evan_Millers
import pandas as pd
from pathlib import Path

@pytest.fixture
def client():
    Flask_App.config['TESTING'] = True
    with Flask_App.test_client() as client:
        yield client

def test_homepage(client):
    response = client.get('/')
    assert response.status_code == 200

def test_index(client):
    response = client.get('/Index')
    assert response.status_code == 200

def test_base_data():
    test_data = pd.DataFrame({
        'feature1': [1, 0, 1, 1],
        'feature2': [0, 1, 1, 0]
    })
    selected_features = ['feature1']
    filtered_data = base_data(test_data, selected_features)
    assert all(filtered_data['feature1'] == 1)

def test_basic_sample_size():
    result = basic_Sample_Size(95, 5)  # Confidence interval 95, Margin of error 5
    assert result > 0

def test_evan_millers():
    result = evan_Millers(10, 5, 80, 5)  # Baseline rate, effect, power, significance level
    assert result > 0
