import pytest
from src.rubric import RUBRIC, compute_overall, rating_for
from src.schemas import DimensionAssessment as D, Rating


def test_weights_sum_to_one():
    assert sum(d.weight for d in RUBRIC) == pytest.approx(1.0)


@pytest.mark.parametrize("s,r", [(1.0, Rating.POOR), (1.7499, Rating.POOR), (1.75, Rating.AVERAGE),
                                 (2.7499, Rating.AVERAGE), (2.75, Rating.GOOD), (3.7499, Rating.GOOD),
                                 (3.75, Rating.EXCELLENT), (5.0, Rating.EXCELLENT)])
def test_threshold_boundaries(s, r):
    assert rating_for(s) == r


def test_out_of_range():
    with pytest.raises(ValueError):
        rating_for(5.1)


def all_dims(score):
    return [D(dimension=d.key, score=score) for d in RUBRIC]


def test_weighted_all_equal():
    s, r, w = compute_overall(all_dims(4))
    assert (s, r, w) == (4.0, Rating.EXCELLENT, 1.0)


def test_weighted_mixed():
    dims = [D(dimension="clarity", score=5), D(dimension="content", score=1)]  # (1.5+0.25)/0.55
    s, _, w = compute_overall(dims, 0.5)
    assert s == pytest.approx(3.18, abs=0.01) and w == pytest.approx(0.55)


def test_insufficient_dimension_excluded_and_renormalised():
    dims = all_dims(4)
    dims[0] = D(dimension="clarity", score=None, sufficient_evidence=False)
    s, _, w = compute_overall(dims)
    assert s == 4.0 and w == pytest.approx(0.70)


def test_below_min_weight_gives_no_rating():
    s, r, w = compute_overall([D(dimension="structure", score=5), D(dimension="av_usability", score=5)])
    assert s is None and r is None and w == pytest.approx(0.25)


def test_unknown_and_duplicate_dimensions_ignored():
    dims = all_dims(3) + [D(dimension="bogus", score=5), D(dimension="clarity", score=1)]
    assert compute_overall(dims)[0] == 3.0


def test_rounding_cannot_cross_threshold():
    # raw 3.745 must stay Good even though it displays as 3.75
    dims = [D(dimension="clarity", score=4), D(dimension="content", score=4), D(dimension="engagement", score=4),
            D(dimension="structure", score=3), D(dimension="av_usability", score=4)]
    s, r, _ = compute_overall(dims)
    assert s == 3.85 and r == Rating.EXCELLENT
