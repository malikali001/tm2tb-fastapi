"""
TM2TB API unit tests.
"""
from fastapi.testclient import TestClient
from main import app
from dependencies import API_KEY


client = TestClient(app)
headers = {"access_token": API_KEY}


def test_basic_request():
    """
    GIVEN a bitext, language IDs and freq_min,
    WHEN the bitext is sent to /biterms,
    THEN try to extract biterms
    """
    src_texts = "Il panda gigante o panda maggiore è un mammifero appartenente alla famiglia degli orsi."
    tgt_texts = (
        "The giant panda or big panda is a mammal that belongs to the bear family."
    )
    src_lang = "it"
    tgt_lang = "en"
    freq_min = 1
    pload = {
        "src_texts": [src_texts],
        "tgt_texts": [tgt_texts],
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "freq_min": freq_min,
    }
    r = client.post("/api/v1_0/biterms", headers=headers, json=pload)
    assert r.status_code == 200
    assert r.json() == {
        "src_terms": ["panda gigante", "panda maggiore", "mammifero"],
        "src_labels": ["", "", ""],
        "src_frequencies": [1, 1, 1],
        "src_clusters": [0, 0, 0],
        "tgt_terms": ["giant panda", "big panda", "mammal"],
        "similarities": [0.976, 0.913, 0.914],
        "frequencies": [1, 1, 1],
        "ranks": [0.424, 0.398, 0.199],
        "origins": ["similarity", "similarity", "similarity"],
    }


def test_lang_detect():
    """
    GIVEN a bitext and freq_min (without language IDs),
    WHEN the bitext is sent to /biterms,
    THEN try to extract biterms and assert that languages are detected
    """
    src_texts = "Il panda gigante o panda maggiore è un mammifero appartenente alla famiglia degli orsi."
    tgt_texts = (
        "The giant panda or big panda is a mammal that belongs to the bear family."
    )
    freq_min = 1
    pload = {"src_texts": [src_texts], "tgt_texts": [tgt_texts], "freq_min": freq_min}
    r = client.post("/api/v1_0/biterms", params=headers, json=pload)
    assert r.status_code == 200
    assert r.json() == {
        "src_terms": ["panda gigante", "panda maggiore", "mammifero"],
        "src_labels": ["", "", ""],
        "src_frequencies": [1, 1, 1],
        "src_clusters": [0, 0, 0],
        "tgt_terms": ["giant panda", "big panda", "mammal"],
        "similarities": [0.976, 0.913, 0.914],
        "frequencies": [1, 1, 1],
        "ranks": [0.424, 0.398, 0.199],
        "origins": ["similarity", "similarity", "similarity"],
    }


def test_unsupported_lang():
    """
    GIVEN a bitext, an unsupported language ID and freq_min,
    WHEN the bitext is sent to /biterms,
    THEN try to extract biterms and assert that the app raises the appropriate error.
    """
    src_texts = "Il panda gigante o panda maggiore è un mammifero appartenente alla famiglia degli orsi."
    tgt_texts = (
        "The giant panda or big panda is a mammal that belongs to the bear family."
    )
    src_lang = "zh"
    tgt_lang = "en"
    freq_min = 1
    pload = {
        "src_texts": [src_texts],
        "tgt_texts": [tgt_texts],
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "freq_min": freq_min,
    }
    r = client.post("/api/v1_0/biterms", headers=headers, json=pload)
    assert r.status_code == 422
    assert r.json() == {"detail": "Model zh_core_news_md is not currently supported."}


def test_no_terms_above_freq_min():
    """
    GIVEN a bitext, language IDs and a freq_min of 5,
    WHEN the bitext is sent to /biterms,
    THEN try to extract biterms and assert that the response is valid, but empty.
    """
    src_texts = "Il panda gigante o panda maggiore è un mammifero appartenente alla famiglia degli orsi."
    tgt_texts = (
        "The giant panda or big panda is a mammal that belongs to the bear family."
    )
    src_lang = "it"
    tgt_lang = "en"
    freq_min = 5
    pload = {
        "src_texts": [src_texts],
        "tgt_texts": [tgt_texts],
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "freq_min": freq_min,
    }
    r = client.post("/api/v1_0/biterms", headers=headers, json=pload)
    assert r.status_code == 200
    assert r.json() == {
        "src_terms": [],
        "src_labels": [],
        "src_frequencies": [],
        "src_clusters": [],
        "tgt_terms": [],
        "similarities": [],
        "frequencies": [],
        "ranks": [],
        "origins": [],
    }


def test_empty_pload():
    """
    GIVEN a completely empty payload,
    WHEN the payload  is sent to /biterms,
    THEN assert that FastAPI returns a detailed error message.
    """
    r = client.post("/api/v1_0/biterms", headers=headers, json={})
    assert r.status_code == 422
    assert r.json() == {
        "detail": [
            {
                "loc": ["body", "src_texts"],
                "msg": "field required",
                "type": "value_error.missing",
            },
            {
                "loc": ["body", "tgt_texts"],
                "msg": "field required",
                "type": "value_error.missing",
            },
        ]
    }


def test_ngrams():
    """
    GIVEN a bitext and a span_range of (1, 1)
    WHEN the bitext is sent to /biterms,
    THEN try to extract biterms and assert that only unigrams are returned.
    """
    src_texts = [
        (
            "Proteins may be purified from other cellular components using a variety of "
            "techniques such as ultracentrifugation, precipitation, electrophoresis, and "
            "chromatography; the advent of genetic engineering has made possible a number "
            "of methods to facilitate purification. Methods commonly used to study "
            "protein structure and function include immunohistochemistry, site-directed "
            "mutagenesis, X-ray crystallography, nuclear magnetic resonance and mass "
            "spectrometry."
        )
    ]

    tgt_texts = [
        (
            "Las proteínas pueden purificarse a partir de otros componentes celulares "
            "mediante diversas técnicas como la ultracentrifugación, la precipitación, la "
            "electroforesis y la cromatografía; la llegada de la ingeniería genética ha "
            "hecho posible una serie de métodos para facilitar la purificación. Entre los "
            "métodos más utilizados para estudiar la estructura y la función de las "
            "proteínas se encuentran la inmunohistoquímica, la mutagénesis dirigida al "
            "lugar, la cristalografía de rayos X, la resonancia magnética nuclear y la "
            "espectrometría de masas."
        )
    ]

    src_lang = "en"
    tgt_lang = "es"

    # Test unigrams
    pload1 = {
        "src_texts": src_texts,
        "tgt_texts": tgt_texts,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "span_range": (1, 1),
    }

    r = client.post("/api/v1_0/biterms", headers=headers, json=pload1)
    assert r.status_code == 200
    assert r.json() == {
        "src_terms": [
            "electrophoresis",
            "protein",
            "spectrometry",
            "chromatography",
            "ultracentrifugation",
            "purification",
            "engineering",
            "crystallography",
            "mutagenesis",
            "components",
            "precipitation",
            "variety",
            "advent",
            "immunohistochemistry",
            "resonance",
        ],
        "src_labels": ["", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
        "src_frequencies": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "src_clusters": [1, 3, 1, 1, 1, 2, 0, 1, 0, 3, 2, 0, 2, 4, 0],
        "tgt_terms": [
            "electroforesis",
            "proteínas",
            "espectrometría",
            "cromatografía",
            "ultracentrifugación",
            "purificación",
            "ingeniería",
            "cristalografía",
            "mutagénesis",
            "componentes",
            "precipitación",
            "",
            "",
            "",
            "",
        ],
        "similarities": [
            0.96,
            0.929,
            0.919,
            0.925,
            0.965,
            0.982,
            0.973,
            0.942,
            0.959,
            0.984,
            0.981,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        "frequencies": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        "ranks": [
            0.216,
            0.215,
            0.204,
            0.192,
            0.185,
            0.145,
            0.139,
            0.138,
            0.134,
            0.112,
            0.029,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        "origins": [
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "",
            "",
            "",
            "",
        ],
    }

    # Test trigrams
    pload2 = {
        "src_texts": src_texts,
        "tgt_texts": tgt_texts,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "span_range": (1, 3),
    }
    r = client.post("/api/v1_0/biterms", headers=headers, json=pload2)
    assert r.status_code == 200
    assert r.json() == {
        "src_terms": [
            "ray crystallography",
            "electrophoresis",
            "genetic engineering",
            "mass spectrometry",
            "chromatography",
            "ultracentrifugation",
            "nuclear magnetic resonance",
            "purification",
            "cellular components",
            "precipitation",
            "Proteins",
            "variety of techniques",
            "advent",
            "number of methods",
            "protein structure",
            "immunohistochemistry",
            "mutagenesis",
        ],
        "src_labels": [
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        "src_frequencies": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "src_clusters": [1, 1, 4, 1, 1, 1, 1, 0, 3, 0, 3, 2, 0, 2, 3, 4, 4],
        "tgt_terms": [
            "cristalografía de rayos",
            "electroforesis",
            "ingeniería genética",
            "espectrometría de masas",
            "cromatografía",
            "ultracentrifugación",
            "resonancia magnética nuclear",
            "purificación",
            "componentes celulares",
            "precipitación",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        "similarities": [
            0.915,
            0.96,
            0.942,
            0.907,
            0.925,
            0.965,
            0.938,
            0.982,
            0.915,
            0.981,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        "frequencies": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        "ranks": [
            0.217,
            0.216,
            0.209,
            0.198,
            0.192,
            0.185,
            0.174,
            0.145,
            0.137,
            0.029,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        "origins": [
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
    }


def test_return_stopwords():
    """
    GIVEN a bitext, and parameter `filter_stopwords` set to False,
    WHEN the bitext is sent to /biterms,
    THEN try to extract biterms and assert that stopwords (e.g, 'family') are returned.
    """
    src_texts = "Il panda gigante o panda maggiore è un mammifero appartenente alla famiglia degli orsi."
    tgt_texts = (
        "The giant panda or big panda is a mammal that belongs to the bear family."
    )
    src_lang = "it"
    tgt_lang = "en"
    freq_min = 1
    pload = {
        "src_texts": [src_texts],
        "tgt_texts": [tgt_texts],
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "freq_min": freq_min,
        "filter_stopwords": False,
    }
    r = client.post("/api/v1_0/biterms", headers=headers, json=pload)
    assert r.status_code == 200
    assert r.json() == {
        "src_terms": ["panda gigante", "panda maggiore", "mammifero", "famiglia"],
        "src_labels": ["", "", "", ""],
        "src_frequencies": [1, 1, 1, 1],
        "src_clusters": [0, 0, 0, 0],
        "tgt_terms": ["giant panda", "big panda", "mammal", ""],
        "similarities": [0.976, 0.913, 0.914, 0.0],
        "frequencies": [1, 1, 1, 0],
        "ranks": [0.424, 0.398, 0.199, 0.0],
        "origins": ["similarity", "similarity", "similarity", ""],
    }


def test_similarity_min():
    """
    GIVEN a bitext, language IDs and a similarity min of .8,
    WHEN the bitext is sent to /biterms,
    THEN assert that matches with a similarity lower than default (.9) are returned
    """
    src_texts = "Il panda gigante o panda maggiore è un mammifero appartenente alla famiglia degli orsi."
    tgt_texts = (
        "The giant panda or big panda is a mammal that belongs to the bear family."
    )
    src_lang = "it"
    tgt_lang = "en"
    freq_min = 0.8
    pload = {
        "src_texts": [src_texts],
        "tgt_texts": [tgt_texts],
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "similarity_min": freq_min,
    }
    r = client.post("/api/v1_0/biterms", headers=headers, json=pload)
    assert r.status_code == 200
    assert r.json() == {
        "src_terms": ["panda gigante", "panda maggiore", "mammifero"],
        "src_labels": ["", "", ""],
        "src_frequencies": [1, 1, 1],
        "src_clusters": [0, 0, 0],
        "tgt_terms": ["giant panda", "big panda", "mammal"],
        "similarities": [0.976, 0.913, 0.914],
        "frequencies": [1, 1, 1],
        "ranks": [0.424, 0.398, 0.199],
        "origins": ["similarity", "similarity", "similarity"],
    }


def test_extract_biterms_invalid_key():
    """
    GIVEN a bitext, language IDs and freq_min,
    WHEN the access token is invalid,
    THEN return a 403 with a message indicating that the access token is invalid
    """
    src_texts = "Il panda gigante o panda maggiore è un mammifero appartenente alla famiglia degli orsi."
    tgt_texts = (
        "The giant panda or big panda is a mammal that belongs to the bear family."
    )
    src_lang = "it"
    tgt_lang = "en"
    freq_min = 1
    pload = {
        "src_texts": [src_texts],
        "tgt_texts": [tgt_texts],
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "freq_min": freq_min,
    }
    r = client.post("/api/v1_0/biterms", headers={"access_token": "1234"}, json=pload)
    assert r.status_code == 403
    assert r.json() == {"detail": "Could not validate credentials"}


def test_include_entities():
    """
    GIVEN a bitext (with some expected named entities)
    WHEN the bitext is sent to /biterms
    THEN assert that in this case 'Sichuan' and 'Cina' have a label of 'LOC'.
    """
    src_texts = (
        "Originario della Cina centrale, vive nelle regioni montuose del Sichuan."
    )
    tgt_texts = (
        "Native to central China, it lives in the mountainous regions of Sichuan."
    )
    src_lang = "it"
    tgt_lang = "en"
    freq_min = 1

    pload = {
        "src_texts": [src_texts],
        "tgt_texts": [tgt_texts],
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "freq_min": freq_min,
        "include_entities": True,
    }
    r = client.post("/api/v1_0/biterms", headers=headers, json=pload)
    assert r.status_code == 200
    assert r.json() == {
        "src_terms": ["Sichuan", "Cina", "regioni montuose"],
        "src_labels": ["LOC", "LOC", ""],
        "src_frequencies": [1, 1, 1],
        "src_clusters": [0, 0, 0],
        "tgt_terms": ["Sichuan", "China", ""],
        "similarities": [1.0, 0.95, 0.0],
        "frequencies": [1, 1, 0],
        "ranks": [0.344, 0.264, 0.0],
        "origins": ["similarity", "similarity", ""],
    }


def test_collapse_lemmas():
    """
    GIVEN a bitext (with some expected terms whose lemmas can be folded, ("Saxophone/saxophone/saxophones")
    WHEN the bitext is sent to /biterms
    THEN assert that the results are collapsed
    """
    src_texts = [
        (
            "The saxophone was invented by the Belgian instrument maker Adolphe Sax in "
            "the early 1840s. Saxophones are used in chamber music, such as saxophone "
            "quartets and other chamber combinations of instruments. Soprano and "
            "sopranino saxophones are usually constructed with a straight tube with a "
            "flared bell at the end, although some are made in the curved shape of the "
            "other saxophones. Alto and larger saxophones have a detachable curved neck "
            "and a U-shaped bend (the bow) that directs the tubing upward as it "
            "approaches the bell. There are rare examples of alto, tenor, and baritone "
            "saxophones with mostly straight bodies. The baritone, bass, and contrabass "
            "saxophones accommodate the length of the bore with extra bends in the tube. "
            "The fingering system for the saxophone is similar to the systems used for "
            "the oboe, the Boehm system clarinet, and the flute. Brevard's Saxophone "
            "Institute offers students the chance to work with world-renowned faculty on "
            "our 180-acre wooded campus in the beautiful mountains of western North "
            "Carolina. "
        )
    ]

    tgt_texts = [
        (
            "El saxofón fue inventado por el fabricante de instrumentos belga Adolphe Sax "
            "a principios de la década de 1840. Los saxofones se utilizan en la música de "
            "cámara, como los cuartetos de saxofones y otras combinaciones de "
            "instrumentos de cámara. Los saxofones sopranino y soprano suelen estar "
            "construidos con un tubo recto con una campana acampanada en el extremo, "
            "aunque algunos se fabrican con la forma curva de los demás saxofones. Los "
            "saxofones altos y mayores tienen un cuello curvo desmontable y una curva en "
            "forma de U (el arco) que dirige el tubo hacia arriba a medida que se acerca "
            "a la campana. Existen raros ejemplos de saxofones altos, tenores y barítonos "
            "con cuerpos mayoritariamente rectos. Los saxofones barítonos, bajos y "
            "contrabajos se adaptan a la longitud de la campana con curvas adicionales en "
            "el tubo. El sistema de digitación del saxofón es similar a los sistemas "
            "utilizados para el oboe, el clarinete con sistema Boehm y la flauta. El "
            "Instituto del Saxofón de Brevard ofrece a los estudiantes la oportunidad de "
            "trabajar con profesores de renombre mundial en nuestro campus arbolado de "
            "180 acres en las hermosas montañas del oeste de Carolina del Norte."
        )
    ]

    src_lang = "en"
    tgt_lang = "es"

    # Set `collapse_lemmas` as False, both "saxophones" and "saxophone" are returned
    pload1 = {
        "src_texts": src_texts,
        "tgt_texts": tgt_texts,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "collapse_lemmas": False,
    }
    r = client.post("/api/v1_0/biterms", headers=headers, json=pload1)
    assert r.status_code == 200
    assert r.json() == {
        "src_terms": [
            "saxophones",
            "saxophone",
            "sopranino saxophones",
            "saxophone quartets",
            "Soprano",
            "tenor",
            "oboe",
            "straight tube",
            "curved shape",
            "acre",
            "flute",
            "system clarinet",
            "fingering system",
            "baritone saxophones",
            "bore with extra bends",
            "contrabass saxophones",
            "bass",
            "straight bodies",
            "chamber music",
            "baritone",
            "rare examples of alto",
            "bend",
            "chamber combinations of instruments",
            "Saxophones",
            "early 1840s",
            "Belgian instrument maker",
            "beautiful mountains",
        ],
        "src_labels": [
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        "src_frequencies": [
            5,
            3,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        "src_clusters": [
            0,
            0,
            0,
            0,
            3,
            3,
            4,
            1,
            1,
            4,
            3,
            2,
            2,
            0,
            6,
            0,
            4,
            1,
            2,
            3,
            5,
            4,
            2,
            0,
            7,
            2,
            7,
        ],
        "tgt_terms": [
            "saxofones",
            "saxofón",
            "saxofones sopranino",
            "cuartetos de saxofones",
            "soprano",
            "tenores",
            "oboe",
            "tubo recto",
            "forma curva",
            "acres",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        "similarities": [
            0.981,
            0.962,
            0.984,
            0.928,
            0.906,
            0.912,
            1.0,
            0.948,
            0.947,
            0.93,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        "frequencies": [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        "ranks": [
            0.387,
            0.35,
            0.312,
            0.286,
            0.196,
            0.119,
            0.11,
            0.104,
            0.102,
            0.064,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        "origins": [
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
    }

    # Assert that lemma collapsation is True by default
    pload2 = {
        "src_texts": src_texts,
        "tgt_texts": tgt_texts,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
    }
    r = client.post("/api/v1_0/biterms", headers=headers, json=pload2)
    assert r.status_code == 200
    assert r.json() == {
        "src_terms": [
            "saxophones",
            "sopranino saxophones",
            "saxophone quartets",
            "Soprano",
            "tenor",
            "oboe",
            "straight tube",
            "curved shape",
            "acre",
            "straight bodies",
            "flute",
            "system clarinet",
            "fingering system",
            "bore with extra bends",
            "contrabass saxophones",
            "bass",
            "chamber combinations of instruments",
            "baritone saxophones",
            "baritone",
            "rare examples of alto",
            "bend",
            "chamber music",
            "early 1840s",
            "Belgian instrument maker",
            "beautiful mountains",
        ],
        "src_labels": [
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        "src_frequencies": [
            5,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        "src_clusters": [
            0,
            0,
            0,
            3,
            3,
            4,
            1,
            1,
            4,
            1,
            3,
            2,
            2,
            6,
            0,
            4,
            2,
            0,
            3,
            5,
            4,
            2,
            7,
            2,
            7,
        ],
        "tgt_terms": [
            "saxofones",
            "saxofones sopranino",
            "cuartetos de saxofones",
            "soprano",
            "tenores",
            "oboe",
            "tubo recto",
            "forma curva",
            "acres",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        "similarities": [
            0.981,
            0.984,
            0.928,
            0.906,
            0.912,
            1.0,
            0.948,
            0.947,
            0.93,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        "frequencies": [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        "ranks": [
            0.387,
            0.312,
            0.286,
            0.196,
            0.119,
            0.11,
            0.104,
            0.102,
            0.064,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        "origins": [
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
    }

    # Assert that lemma collapsation is True by default ("Saxophones" should not be returned).
    # Also include entities and assert that related terms are unaffected.
    pload3 = {
        "src_texts": src_texts,
        "tgt_texts": tgt_texts,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "include_entities": True,
    }
    r = client.post("/api/v1_0/biterms", headers=headers, json=pload3)
    assert r.status_code == 200
    assert r.json() == {
        "src_terms": [
            "saxophones",
            "Brevard's Saxophone Institute",
            "sopranino saxophones",
            "saxophone quartets",
            "Adolphe Sax",
            "Soprano",
            "tenor",
            "oboe",
            "straight tube",
            "curved shape",
            "North Carolina",
            "acre",
            "flute",
            "system clarinet",
            "baritone saxophones",
            "fingering system",
            "bore with extra bends",
            "contrabass saxophones",
            "bass",
            "straight bodies",
            "early 1840s",
            "baritone",
            "rare examples of alto",
            "bend",
            "chamber combinations of instruments",
            "chamber music",
            "Belgian instrument maker",
            "Alto",
            "beautiful mountains",
        ],
        "src_labels": [
            "",
            "ORG",
            "",
            "",
            "ORG",
            "",
            "",
            "",
            "",
            "",
            "GPE",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "PERSON",
            "",
        ],
        "src_frequencies": [
            5,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        "src_clusters": [
            1,
            1,
            1,
            1,
            8,
            6,
            6,
            3,
            4,
            3,
            2,
            3,
            0,
            7,
            1,
            0,
            5,
            1,
            0,
            4,
            2,
            0,
            8,
            3,
            7,
            7,
            7,
            8,
            2,
        ],
        "tgt_terms": [
            "saxofones",
            "Instituto del Saxofón de Brevard",
            "saxofones sopranino",
            "cuartetos de saxofones",
            "Adolphe Sax",
            "soprano",
            "tenores",
            "oboe",
            "tubo recto",
            "forma curva",
            "Carolina del Norte",
            "acres",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        "similarities": [
            0.981,
            0.956,
            0.984,
            0.928,
            1.0,
            0.906,
            0.912,
            1.0,
            0.948,
            0.947,
            0.957,
            0.93,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        "frequencies": [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        "ranks": [
            0.387,
            0.319,
            0.312,
            0.286,
            0.221,
            0.196,
            0.119,
            0.11,
            0.104,
            0.102,
            0.088,
            0.064,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        "origins": [
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
    }


def test_src_frequencies():
    """
    GIVEN a bitext,
    WHEN the bitext is sent to /biterms
    THEN assert that the source frequencies are correct.
    """
    src_texts = [
        (
            "Mexico, officially the United Mexican States, is a country in the southern "
            "portion of North America. It is bordered to the north by the United States; "
            "to the south and west by the Pacific Ocean; to the southeast by Guatemala, "
            "Belize, and the Caribbean Sea; and to the east by the Gulf of Mexico. Mexico "
            "is organized as a federation comprising 31 states and Mexico City, its "
            "capital. The prehistory of Mexico stretches back millennia. The earliest "
            "human artifacts in Mexico are chips of stone tools found near campfire "
            "remains in the Valley of Mexico and radiocarbon-dated to circa 10,000 years "
            "ago. Bullfighting came to Mexico 500 years ago with the arrival of the "
            "Spanish. Despite efforts by animal rights activists to outlaw it, "
            "bullfighting remains a popular sport in the country, and almost all large "
            "cities have bullrings. Plaza Mexico in Mexico City, which seats 45,000 "
            "people, is the largest bullring in the world."
        )
    ]

    tgt_texts = [
        (
            "Mexiko, offiziell die Vereinigten Mexikanischen Staaten, ist ein Land im "
            "südlichen Teil Nordamerikas. Es grenzt im Norden an die Vereinigten Staaten, "
            "im Süden und Westen an den Pazifischen Ozean, im Südosten an Guatemala, "
            "Belize und das Karibische Meer und im Osten an den Golf von Mexiko. Mexiko "
            "ist als Föderation organisiert, die 31 Bundesstaaten und die Hauptstadt "
            "Mexiko-Stadt umfasst. Die Vorgeschichte Mexikos reicht Jahrtausende zurück. "
            "Die frühesten menschlichen Artefakte in Mexiko sind Späne von "
            "Steinwerkzeugen, die in der Nähe von Lagerfeuerresten im Tal von Mexiko "
            "gefunden wurden und deren Radiokarbondatierung auf etwa 10.000 Jahre "
            "zurückgeht. Der Stierkampf kam vor 500 Jahren mit der Ankunft der Spanier "
            "nach Mexiko. Trotz der Bemühungen von Tierschützern, ihn zu verbieten, ist "
            "der Stierkampf nach wie vor ein beliebter Sport in Mexiko, und in fast allen "
            "großen Städten gibt es Stierkampfarenen. Die größte Stierkampfarena der Welt "
            "ist die Plaza Mexico in Mexiko-Stadt, die 45.000 Menschen Platz bietet."
        )
    ]

    src_lang = "en"
    tgt_lang = "de"

    pload = {
        "src_texts": src_texts,
        "tgt_texts": tgt_texts,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "include_entities": True,
    }

    r = client.post("/api/v1_0/biterms", headers=headers, json=pload)
    assert r.status_code == 200
    assert r.json() == {
        "src_terms": [
            "Mexico",
            "Mexico City",
            "Plaza Mexico",
            "earliest human artifacts",
            "popular sport",
            "Belize",
            "southern portion",
            "Guatemala",
            "chips of stone tools",
            "efforts by animal rights activists",
            "Bullfighting",
            "radiocarbon",
            "campfire",
            "Spanish",
            "prehistory",
            "federation",
            "southeast",
            "North America",
            "bullrings",
        ],
        "src_labels": [
            "GPE",
            "GPE",
            "GPE",
            "",
            "",
            "GPE",
            "",
            "GPE",
            "",
            "",
            "",
            "",
            "",
            "NORP",
            "",
            "",
            "",
            "LOC",
            "",
        ],
        "src_frequencies": [5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "src_clusters": [0, 0, 0, 4, 3, 0, 1, 0, 4, 4, 2, 5, 2, 3, 1, 1, 1, 1, 2],
        "tgt_terms": [
            "Mexiko",
            "Mexiko-Stadt",
            "Plaza Mexico",
            "frühesten menschlichen Artefakte",
            "beliebter Sport",
            "Belize",
            "südlichen Teil",
            "Guatemala",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        "similarities": [
            0.98,
            0.964,
            1.0,
            0.905,
            0.949,
            1.0,
            0.958,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        "frequencies": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "ranks": [
            0.443,
            0.37,
            0.276,
            0.204,
            0.193,
            0.154,
            0.071,
            0.068,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        "origins": [
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
    }


def test_return_unmatched_terms():
    """
    GIVEN a bitext
    WHEN the bitext is sent to /biterms
    THEN assert that the return_unmatched_terms parameter works correctly.
    """
    src_texts = (
        "Originario della Cina centrale, vive nelle regioni montuose del Sichuan."
    )
    tgt_texts = (
        "Native to central China, it lives in the mountainous regions of Sichuan."
    )
    src_lang = "it"
    tgt_lang = "en"
    freq_min = 1

    pload1 = {
        "src_texts": [src_texts],
        "tgt_texts": [tgt_texts],
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "freq_min": freq_min,
        "include_entities": True,
    }
    r = client.post("/api/v1_0/biterms", headers=headers, json=pload1)
    assert r.status_code == 200
    assert r.json() == {
        "src_terms": ["Sichuan", "Cina", "regioni montuose"],
        "src_labels": ["LOC", "LOC", ""],
        "src_frequencies": [1, 1, 1],
        "src_clusters": [0, 0, 0],
        "tgt_terms": ["Sichuan", "China", ""],
        "similarities": [1.0, 0.95, 0.0],
        "frequencies": [1, 1, 0],
        "ranks": [0.344, 0.264, 0.0],
        "origins": ["similarity", "similarity", ""],
    }

    pload2 = {
        "src_texts": [src_texts],
        "tgt_texts": [tgt_texts],
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "freq_min": freq_min,
        "include_entities": True,
        "return_unmatched_terms": False,
    }
    r = client.post("/api/v1_0/biterms", headers=headers, json=pload2)
    assert r.status_code == 200
    assert r.json() == {
        "src_terms": ["Sichuan", "Cina"],
        "src_labels": ["LOC", "LOC"],
        "src_frequencies": [1, 1],
        "src_clusters": [0, 0],
        "tgt_terms": ["Sichuan", "China"],
        "similarities": [1.0, 0.95],
        "frequencies": [1, 1],
        "ranks": [0.344, 0.264],
        "origins": ["similarity", "similarity"],
    }


def test_mt_unmatched_terms():
    """
    GIVEN a bitext
    WHEN the bitext is sent to /biterms,
    THEN machine translate unmatched src terms and assert that they are flagged correctly.
    """
    src_texts = [
        (
            "Proteins may be purified from other cellular components using a variety of "
            "techniques such as ultracentrifugation, precipitation, electrophoresis, and "
            "chromatography; the advent of genetic engineering has made possible a number "
            "of methods to facilitate purification. Methods commonly used to study "
            "protein structure and function include immunohistochemistry, site-directed "
            "mutagenesis, X-ray crystallography, nuclear magnetic resonance and mass "
            "spectrometry."
        )
    ]

    tgt_texts = [
        (
            "Las proteínas pueden purificarse a partir de otros componentes celulares "
            "mediante diversas técnicas como la ultracentrifugación, la precipitación, la "
            "electroforesis y la cromatografía; la llegada de la ingeniería genética ha "
            "hecho posible una serie de métodos para facilitar la purificación. Entre los "
            "métodos más utilizados para estudiar la estructura y la función de las "
            "proteínas se encuentran la inmunohistoquímica, la mutagénesis dirigida al "
            "lugar, la cristalografía de rayos X, la resonancia magnética nuclear y la "
            "espectrometría de masas."
        )
    ]

    src_lang = "en"
    tgt_lang = "es"

    pload = {
        "src_texts": src_texts,
        "tgt_texts": tgt_texts,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "mt_unmatched_terms": True,
    }

    r = client.post("/api/v1_0/biterms", headers=headers, json=pload)
    assert r.status_code == 200
    assert r.json() == {
        "src_terms": [
            "electrophoresis",
            "mass spectrometry",
            "chromatography",
            "ultracentrifugation",
            "nuclear magnetic resonance",
            "purification",
            "cellular components",
            "precipitation",
            "Proteins",
            "variety of techniques",
            "advent of genetic engineering",
            "number of methods",
            "protein structure",
            "immunohistochemistry",
            "mutagenesis",
            "ray crystallography",
        ],
        "src_labels": ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
        "src_frequencies": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "src_clusters": [1, 1, 1, 1, 1, 0, 4, 0, 4, 2, 3, 2, 4, 4, 3, 1],
        "tgt_terms": [
            "electroforesis",
            "espectrometría de masas",
            "cromatografía",
            "ultracentrifugación",
            "resonancia magnética nuclear",
            "purificación",
            "componentes celulares",
            "precipitación",
            "Proteínas",
            "variedad de técnicas",
            "advenimiento de la ingeniería genética",
            "número de métodos",
            "estructura proteica",
            "inmunohistoquímica",
            "mutagénesis",
            "cristalografía de rayos",
        ],
        "similarities": [
            0.96,
            0.907,
            0.925,
            0.965,
            0.938,
            0.982,
            0.915,
            0.981,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        "frequencies": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "ranks": [
            0.216,
            0.198,
            0.192,
            0.185,
            0.174,
            0.145,
            0.137,
            0.029,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        "origins": [
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "similarity",
            "MT",
            "MT",
            "MT",
            "MT",
            "MT",
            "MT",
            "MT",
            "MT",
        ],
    }
