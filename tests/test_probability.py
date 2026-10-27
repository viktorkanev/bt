import numpy as np

from footy.models.probability import OrderedLogit


def test_ordered_logit_monotone():
    rng = np.random.default_rng(1)
    x = rng.normal(size=5000)
    latent = 0.9 * x + rng.logistic(size=5000)
    y = np.digitize(latent, [-0.6, 0.4])  # 0=A, 1=D, 2=H
    ol = OrderedLogit().fit(x[:, None], y)
    p = ol.predict_proba(np.array([[-2.0], [0.0], [2.0]]))
    assert np.allclose(p.sum(axis=1), 1)
    assert p[2, 0] > p[1, 0] > p[0, 0]  # P(home) rises with x
    assert p[0, 2] > p[2, 2]
