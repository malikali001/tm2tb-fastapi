"""
TM2TB API summarization tests.
"""
from dependencies import API_KEY
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)
headers = {"access_token": API_KEY}


def test_basic_summarization():
    """
    GIVEN a bitext, language IDs and freq_min,
    WHEN the bitext is sent to /biterms,
    THEN try to extract biterms
    """
    texts = [
        "Please welcome Professor Michael Mcfaul, director of the Freeman Spogli Institute for International Studies and former U.S. ambassador to Russia. First thing I'm supposed to do is introduce myself, but God just did that. Welcome everyone to the keynote address for our conference today called Challenges to democracy in the digital information realm. Yes, I am Michael Mcfaul, director of Epsi professor of political science and a Hoover Fellow. The Cyber Policy Center, one of emphasized newest centers now run jointly with the Stanford Law School, is really the convener of our meeting today. It is I consider, if not one. It is one of, if not the most important issues of our time for anybody. That cares about democracy and I just wanna thank them all and especially thank Ambassador Eileen Donahoe, Larry Diamond and Tracy Nava choke for their leadership for getting us here today. I'm also thrilled to welcome back to the farm. It's third time, by the way, my old boss, President Barack Obama. Yes. Let me just say that again and get another plus like Barack Obama's here. So I love Stanford. I was an undergraduate here. I've been teaching since 1995. My kids are born here. They go to school here. We live on campus. I plan to die here. I love Stanford University. It's the greatest institution I think in the world. But but working for Barack Obama for five years was an honor and thrill of my lifetime. In fact, after those years on his team, it took me a little while to figure out what to do next. For a while, so my family will tell you and my son, Luke, I think is here. I don't see him as Luke here. I don't see him. OK, maybe he's not here. I thought he's gonna come I maybe I can't see him, but they could tell you you know, as I was in a bit of a funk, I want to tell you honestly, because working for this guy was just a fantastic experience through the good times and the bad times. And I sold it around for a while, cranked up Bruce Springsteen's glory days. And repeated endless"
    ]

    lang = "en"
    pload = {"texts": texts, "lang": lang, "summary_sentences_n": "15"}
    r = client.post("/api/v1_0/summary", headers=headers, json=pload)
    assert r.status_code == 200
    assert r.json() == {
        "summary": [
            "Please welcome Professor Michael Mcfaul, director of the Freeman "
            "Spogli Institute for International Studies and former U.S. "
            "ambassador to Russia.",
            "First thing I'm supposed to do is introduce myself, but God just "
            "did that.",
            "Welcome everyone to the keynote address for our conference today "
            "called Challenges to democracy in the digital information realm.",
            "Yes, I am Michael Mcfaul, director of Epsi professor of "
            "political science and a Hoover Fellow.",
            "The Cyber Policy Center, one of emphasized newest centers now "
            "run jointly with the Stanford Law School, is really the convener "
            "of our meeting today.",
            "It is one of, if not the most important issues of our time for "
            "anybody.",
            "That cares about democracy and I just wanna thank them all and "
            "especially thank Ambassador Eileen Donahoe, Larry Diamond and "
            "Tracy Nava choke for their leadership for getting us here today.",
            "I'm also thrilled to welcome back to the farm.",
            "I plan to die here.",
            "It's the greatest institution I think in the world.",
            "But but working for Barack Obama for five years was an honor and "
            "thrill of my lifetime.",
            "In fact, after those years on his team, it took me a little "
            "while to figure out what to do next.",
            "For a while, so my family will tell you and my son, Luke, I "
            "think is here.",
            "I thought he's gonna come I maybe I can't see him, but they "
            "could tell you you know, as I was in a bit of a funk, I want to "
            "tell you honestly, because working for this guy was just a "
            "fantastic experience through the good times and the bad times.",
            "And I sold it around for a while, cranked up Bruce Springsteen's "
            "glory days.",
        ]
    }
