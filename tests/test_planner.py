import pandas as pd
import pytest

from careaccess.planner import export_csv, plan_outreach


def frame():
    return pd.DataFrame({'appointment_id':['TEST-02','TEST-01','TEST-03'], 'miss_probability':[.2,.2,.8]})


def test_capacity_and_stable_ties():
    assert plan_outreach(frame(), 2).appointment_id.tolist() == ['TEST-03','TEST-01']
    assert len(plan_outreach(frame(), 100)) == 3
    assert len(plan_outreach(frame(), 0)) == 0


@pytest.mark.parametrize('n', [-1, 1.2, True, '2'])
def test_invalid_capacity(n):
    with pytest.raises(ValueError): plan_outreach(frame(), n)


@pytest.mark.parametrize('p', [float('nan'), float('inf'), -0.1, 1.1])
def test_invalid_probability(p):
    f=frame();f.loc[0,'miss_probability']=p
    with pytest.raises(ValueError): plan_outreach(f, 1)


def test_csv_formula_escape():
    f=pd.DataFrame({'ref':['=1+1','normal',' @cmd'], 'probability':[.1,.2,.3]})
    result=export_csv(f).decode('utf-8-sig')
    assert "'=1+1" in result and "' @cmd" in result and ',0.1' in result
