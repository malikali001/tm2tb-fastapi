import datetime

import jwt
from dependencies import API_KEY
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def create_jwt():
    SECRET_KEY = "secret"
    json_data = {
        "id": "123",
        "exp": (datetime.datetime.now() + datetime.timedelta(minutes=5)).timestamp(),
    }
    return jwt.encode(payload=json_data, key=SECRET_KEY, algorithm="HS256")


headers = {"access_token": create_jwt()}

TRANSCRIPT = {
    "meeting_id": "123",
    "lang_code": "en",
    "interpreter_audio": True,
    "text": "I have no preference if you think of a question in the middle, feel free to just raise your hand or interrupt. Either one is fine. OK, sounds great. Here we go. Soum I'll have to share it with the screen like this. Hope that's OK. So yeah, welcome to my presentation on Agile product development. Here I will discuss the framework of agile and how the technology team, and more specifically my own tech team, uses these practices.",
    "meta_data": "Je vais donc partager mon écran, Bonjour donc ma présentation du développement de produits adjaye. Je vais parler de notre cadre de développement agile et comment notre équipe de technologie.\nUtilise en fait ce logiciel. Nous avons quelques réunions ces dernières années.\nDonc vous pouvez voir comment vous allez voir comment les différentes en fait entre les différentes équipes.\nEt comment nous utilisons ce logiciel dans notre équipe, donc\nAujourd'hui, vous allez voir à l'ordre du jour.",
}


def test_summary():
    response = client.post(
        "/api/v1_0/transcript-insert",
        json=TRANSCRIPT,
        headers={"access_token": API_KEY},
    )
    payload = {"lang": "en"}
    response = client.post(
        f"/api/v1_0/meetings/123/summary?access_token={API_KEY}",
        json=payload,
        headers=headers,
    )
    data = response.json()
    assert response.status_code == 200
    assert type(response.json()) == dict


def test_meeting_id_missing():
    payload = {"lang": "en"}
    r = client.post(
        "/api/v1_0/meetings/summary", json=payload, headers={"access_token": API_KEY}
    )
    assert r.status_code == 404
    assert r.json() == {"detail": "Not Found"}


def test_with_lang_missing():
    payload = {}
    r = client.post(
        "/api/v1_0/meetings/123/summary",
        json=payload,
        headers={"access_token": API_KEY},
    )
    assert r.status_code == 422
    assert r.json() == {
        "detail": [
            {
                "loc": ["body", "lang"],
                "msg": "field required",
                "type": "value_error.missing",
            }
        ]
    }
