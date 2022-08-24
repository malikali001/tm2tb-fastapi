from datetime import datetime, timedelta

import jwt
import pytest
from dependencies import API_KEY
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def create_jwt():
    SECRET_KEY = "secret"
    json_data = {
        "id": "456",
        "exp": (datetime.now() + timedelta(minutes=5)).timestamp(),
    }
    return jwt.encode(payload=json_data, key=SECRET_KEY, algorithm="HS256")


SOURCE_TRANSCRIPT = {
    "meeting_id": "456",
    "lang_code": "en",
    "interpreter_audio": True,
    "text": "I have no preference if you think of a question in the middle, feel free to just raise your hand or interrupt. Either one is fine. OK, sounds great. Here we go. Soum I'll have to share it with the screen like this. Hope that's OK. So yeah, welcome to my presentation on Agile product development. Here I will discuss the framework of agile and how the technology team, and more specifically my own tech team, uses these practices.",
    "meta_data": "Je vais donc partager mon écran, Bonjour donc ma présentation du développement de produits adjaye. Je vais parler de notre cadre de développement agile et comment notre équipe de technologie.\nUtilise en fait ce logiciel. Nous avons quelques réunions ces dernières années.\nDonc vous pouvez voir comment vous allez voir comment les différentes en fait entre les différentes équipes.\nEt comment nous utilisons ce logiciel dans notre équipe, donc\nAujourd'hui, vous allez voir à l'ordre du jour.",
}

TARGET_TRANSCRIPT = {
    "meeting_id": "456",
    "lang_code": "fr",
    "interpreter_audio": True,
    "text": "Je vais donc partager mon écran, Bonjour donc ma présentation du développement de produits adjaye. Je vais parler de notre cadre de développement agile et comment notre équipe de technologie. Utilise en fait ce logiciel. Nous avons quelques réunions ces dernières années. Donc vous pouvez voir comment vous allez voir comment les différentes en fait entre les différentes équipes. Et comment nous utilisons ce logiciel dans notre équipe, donc Aujourd'hui, vous allez voir à l'ordre du jour, qu'est ce que elle a le Je vais donc vous parler de comment nous utilisons adjaye les terres, les mots, la terminologie que nous utilisons, les cadres de développement et, et cetera, et cetera. Donc commençons par le début, qu'est ce que j'aille à Gmail est un cadre de développement qui permet de découper. Un objectif final en plusieurs sprints où cycles et nous.",
    "meta_data": "Je vais donc partager mon écran, Bonjour donc ma présentation du développement de produits adjaye. Je vais parler de notre cadre de développement agile et comment notre équipe de technologie. Utilise en fait ce logiciel. Nous avons quelques réunions ces dernières années. Donc vous pouvez voir comment vous allez voir comment les différentes en fait entre les différentes équipes. Et comment nous utilisons ce logiciel dans notre équipe, donc Aujourd'hui, vous allez voir à l'ordre du jour, qu'est ce que elle a le Je vais donc vous parler de comment nous utilisons adjaye les terres, les mots, la terminologie que nous utilisons, les cadres de développement et, et cetera, et cetera. Donc commençons par le début, qu'est ce que j'aille à Gmail est un cadre de développement qui permet de découper. Un objectif final en plusieurs sprints où cycles et nous.",
}

headers = {"access_token": create_jwt()}


@pytest.fixture(scope="module", autouse=True)
def create_transcripts():
    response = client.post(
        "/api/v1_0/transcript-insert",
        json=SOURCE_TRANSCRIPT,
        headers={"access_token": API_KEY},
    )
    response = client.post(
        "/api/v1_0/transcript-insert",
        json=TARGET_TRANSCRIPT,
        headers={"access_token": API_KEY},
    )


def test_glossary_creation():
    payload = {"source_lang": "en", "target_lang": "fr"}
    response = client.post(
        f"/api/v1_0/meetings/456/glossary?access_token={API_KEY}",
        json=payload,
        headers=headers,
    )
    assert response.status_code == 200
    assert type(response.json()) == dict


def test_without_languages():
    payload = {}
    r = client.post(
        f"/api/v1_0/meetings/456/glossary?access_token={API_KEY}",
        json=payload,
        headers=headers,
    )
    assert r.status_code == 422
    assert r.json() == {
        "detail": [
            {
                "loc": ["body", "source_lang"],
                "msg": "field required",
                "type": "value_error.missing",
            },
            {
                "loc": ["body", "target_lang"],
                "msg": "field required",
                "type": "value_error.missing",
            },
        ]
    }


def test_without_target_language():
    payload = {"source_lang": "en"}
    r = client.post(
        f"/api/v1_0/meetings/456/glossary?access_token={API_KEY}",
        json=payload,
        headers=headers,
    )
    assert r.status_code == 422
    assert r.json() == {
        "detail": [
            {
                "loc": ["body", "target_lang"],
                "msg": "field required",
                "type": "value_error.missing",
            }
        ]
    }


def test_meeting_id_missing():
    payload = {"source_lang": "en", "target_lang": "fr"}
    r = client.post(
        f"/api/v1_0/meetings/glossary?access_token={API_KEY}",
        json=payload,
        headers=headers,
    )
    assert r.status_code == 404
    assert r.json() == {"detail": "Not Found"}


def test_with_invalid_id():
    payload = {"source_lang": "en", "target_lang": "fr"}
    r = client.post(
        f"/api/v1_0/meetings/3344/glossary?access_token={API_KEY}",
        json=payload,
        headers=headers,
    )
    assert r.status_code == 401
    assert r.json() == {"detail": "access denied"}
