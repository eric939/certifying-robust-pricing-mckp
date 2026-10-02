from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as F
from itertools import product
import json

import pytest

from robust_mckp import Option, PricingInstance
from robust_mckp.global_certificate import instance_sha256, build_global_certificate, verify_global_certificate
from robust_mckp.primitive_certificate import (ExactPrimitivePricingData,
    build_primitive_certificate, verify_primitive_certificate)


def sample(*,gamma=1,radius=F(1,4)):
    return ExactPrimitivePricingData(references=(F(2),F(3)),costs=(F(1),F(4)),
        exposures=(F(1),F(2)),tolerances=(F(1,2),F(1,2)),
        menus=((F(1),F(2),F(3)),(F(2),F(3),F(4))),
        demands=((F(5),F(4),F(3)),(F(3),F(2),F(1))),
        kappa=F(1),radius=radius,gamma=gamma)


def brute(data):
    feasible=[]
    for selected in product(*[range(len(row)) for row in data.menus]):
        surplus=[];loss=[];revenue=F(0)
        for i,j in enumerate(selected):
            p=data.menus[i][j];g=data.demands[i][j];w=data.exposures[i];q=data.costs[i]
            assert abs(p-data.references[i])<=data.tolerances[i]*data.references[i]
            s=w*(p-data.kappa*q)*g
            surplus.append(s)
            loss.append(data.radius*s if s>=0 else -data.radius*s/(1-data.radius))
            revenue+=w*p*g
        residual=sum(surplus,F(0))-sum(sorted(loss,reverse=True)[:data.gamma],F(0))
        if residual>=0:feasible.append((revenue,selected,residual))
    return feasible


@pytest.mark.parametrize('gamma',[0,1,2])
@pytest.mark.parametrize('radius',[F(0),F(1,4),F(9,10)])
def test_exact_original_primitive_bounds_match_exhaustive_integer_plans(gamma,radius):
    data=sample(gamma=gamma,radius=radius)
    feasible=brute(data)
    for value,selected,residual in feasible:
        payload=build_primitive_certificate(data,selected)
        result=verify_primitive_certificate(data,json.dumps(payload))
        assert result.valid,result.errors
        global_result=result.global_verification
        assert global_result.exact_h==value
        assert global_result.exact_robust_residual==residual
        assert value<=max(v for v,_,_ in feasible)<=global_result.exact_u
    assert feasible


def test_fraction_coefficients_preserve_precision_beyond_binary64_range():
    large=F(10**500)
    data=ExactPrimitivePricingData((large,),(F(1),),(large,),(F(0),),
        ((large,),),((large,),),F(1),F(1,3),1)
    result=verify_primitive_certificate(data,build_primitive_certificate(data,[0]))
    assert result.valid,result.errors
    assert result.global_verification.exact_h==large**3


def test_exact_rational_hash_detects_differences_that_share_float_bits():
    tiny=F(1,2**100)
    a=PricingInstance([[Option(F(1),F(1),F(0),price=F(1),original_index=0)]],0)
    b=PricingInstance([[Option(F(1)+tiny,F(1),F(0),price=F(1),original_index=0)]],0)
    assert float(a.items[0][0].value)==float(b.items[0][0].value)
    assert instance_sha256(a)!=instance_sha256(b)
    payload=build_global_certificate(a,[0])
    assert not verify_global_certificate(b,payload).valid


def test_strict_rounded_price_filter_and_bound_original_ids():
    data=ExactPrimitivePricingData((1.,),(.5,),(1.,),(.15,),
        ((1.-.15,1.),),((100.,5.),),1.,0.,0)
    instance=data.encoded_instance()
    assert instance.original_selection([0])==[1]
    payload=build_primitive_certificate(data,[1]);assert verify_primitive_certificate(data,payload).valid
    with pytest.raises(ValueError,match='absent or inadmissible'):
        build_primitive_certificate(data,[0])
    corrupted=deepcopy(payload);corrupted['selected_original_ids']=[0]
    assert not verify_primitive_certificate(data,corrupted).valid


@pytest.mark.parametrize('field',['references','costs','exposures','tolerances','menus','demands','kappa','radius','gamma'])
def test_certificate_binds_every_original_economic_input(field):
    data=sample();payload=build_primitive_certificate(data,[2,2])
    assert verify_primitive_certificate(data,payload).valid
    if field in ('menus','demands'):
        original=list(getattr(data,field));row=list(original[0]);row[1]+=F(1,100);original[0]=tuple(row);changed=tuple(original)
    elif field in ('kappa','radius'):changed=getattr(data,field)+F(1,100)
    elif field=='gamma':changed=0
    else:changed=(getattr(data,field)[0]+F(1,100),)+getattr(data,field)[1:]
    result=verify_primitive_certificate(replace(data,**{field:changed}),payload)
    assert not result.valid and any('hash mismatch' in error for error in result.errors)


def test_same_price_with_different_raw_option_id_is_not_aliased():
    data=ExactPrimitivePricingData((1,),(F(1,2),),(1,),(0,),
        ((1,1),),((1,2),),1,0,0)
    payload=build_primitive_certificate(data,[1])
    altered=deepcopy(payload);altered['selected_original_ids']=[0]
    result=verify_primitive_certificate(data,altered)
    assert not result.valid and 'original/filtered selection mapping mismatch' in result.errors


@pytest.mark.parametrize('change',[
    {'references':(0,3)},{'costs':(-1,4)},{'exposures':(1,0)},
    {'tolerances':(-1,1)},{'radius':1},{'radius':-1},{'gamma':True},{'gamma':3},
    {'kappa':0},{'demands':((5,4,0),(3,2,1))},{'menus':((1,2),(2,3,4))}])
def test_invalid_original_input_domains_are_rejected(change):
    with pytest.raises((ValueError,TypeError)):replace(sample(),**change)


def test_fully_inadmissible_original_group_has_no_certificate():
    data=replace(sample(),tolerances=(F(0),F(0)),menus=((F(1),F(3),F(4)),(F(2),F(3),F(4))))
    with pytest.raises(ValueError,match='no admissible'):data.encoded_instance()


def test_exact_primitive_schema_and_malformed_original_ids_are_rejected():
    data=sample();payload=build_primitive_certificate(data,[2,2])
    for field,value in [('schema_version',True),('selected_original_ids',[True,2]),
                        ('primitive_sha256','0'*64)]:
        changed=deepcopy(payload);changed[field]=value
        assert not verify_primitive_certificate(data,changed).valid
    changed=deepcopy(payload);changed['global_certificate']['schema_version']=True
    assert not verify_primitive_certificate(data,changed).valid
