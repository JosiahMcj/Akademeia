import pytest
from akademeia.data.leakage import check_leakage

def test_clean_passes():
    check_leakage(pool_texts=["I:rev abc;O:cba$"],
                  bench_texts=["I:rev zzz;O:zzz$"],
                  unseen_exprs=["U=m*g*h"])

def test_verbatim_item_hard_exits():
    s = "I:F=m*a;m=33;a=41;O:1353$"
    with pytest.raises(SystemExit):
        check_leakage(pool_texts=[s], bench_texts=[s])

def test_unseen_expr_in_pool_hard_exits():
    with pytest.raises(SystemExit):
        check_leakage(pool_texts=["I:U=m*g*h;m=2;g=3;h=4;O:24$"],
                      bench_texts=["I:J=f*t;f=50;t=60;O:3000$"],
                      unseen_exprs=["U=m*g*h"])

def test_shared_template_prefix_is_allowed():
    # heldrange shares template strings with the pool BY DESIGN (Amendment 1);
    # only full-item or unseen-expr contact may trip
    check_leakage(pool_texts=["I:F=m*a;m=3;a=5;O:15$"],
                  bench_texts=["I:F=m*a;m=45;a=63;O:2835$"])
