"""Exact primitive-input adapter for optimizer-independent global certificates.

This path performs no floating coefficient arithmetic and invokes no optimizer.
Explicit Fraction inputs retain arbitrary rational precision; native real
inputs are first interpreted as their finite binary64 values. It binds every
economic primitive, original menu ID and encoded global certificate. This is
a separate adapter; native floating optimizer results require exact validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
import hashlib
import json
import math
import numbers
from typing import Sequence

from .model import Option, PricingInstance
from .global_certificate import (build_global_certificate, verify_global_certificate,
                                 GlobalDualCertificate, GlobalCertificateVerification)


def exact_input(value):
    if isinstance(value,bool) or not isinstance(value,numbers.Real):
        raise TypeError('Primitive values must be real numbers or Fraction')
    if isinstance(value,F):
        return value
    if isinstance(value,numbers.Integral):
        return F(int(value))
    decoded = float(value)
    if not math.isfinite(decoded):
        raise ValueError('Primitive values must be finite')
    return F.from_float(decoded)


def text(q):
    return str(q.numerator) if q.denominator==1 else f'{q.numerator}/{q.denominator}'


@dataclass(frozen=True)
class ExactPrimitivePricingData:
    references: tuple
    costs: tuple
    exposures: tuple
    tolerances: tuple
    menus: tuple
    demands: tuple
    kappa: F
    radius: F
    gamma: int

    def __post_init__(self):
        for name in ('references','costs','exposures','tolerances'):
            object.__setattr__(self,name,tuple(exact_input(v) for v in getattr(self,name)))
        for name in ('menus','demands'):
            object.__setattr__(self,name,tuple(tuple(exact_input(v) for v in row)
                for row in getattr(self,name)))
        for name in ('kappa','radius'):
            object.__setattr__(self,name,exact_input(getattr(self,name)))
        n = len(self.references)
        if n==0 or any(len(getattr(self,name))!=n for name in
            ('costs','exposures','tolerances','menus','demands')):
            raise ValueError('A nonempty portfolio and equal primitive group counts required')
        if any(v<=0 for values in (self.references,self.costs,self.exposures) for v in values):
            raise ValueError('References, costs and exposures must be strictly positive')
        if any(v<0 for v in self.tolerances) or self.kappa<=0 or not 0<=self.radius<1:
            raise ValueError('Nonnegative tolerances, positive target and radius in [0,1) required')
        if isinstance(self.gamma,bool) or not isinstance(self.gamma,numbers.Integral):
            raise TypeError('Gamma must be an integer')
        if not 0<=self.gamma<=n:
            raise ValueError('Gamma must belong to 0..number of groups')
        object.__setattr__(self,'gamma',int(self.gamma))
        for prices,demands in zip(self.menus,self.demands):
            if not prices or len(prices)!=len(demands) or any(v<=0 for v in prices+demands):
                raise ValueError('Each original menu needs equal nonempty positive prices/demands')

    def canonical_inputs(self):
        return dict(kind='exact_primitive_pricing_inputs',schema_version=1,
            **{name:[text(v) for v in getattr(self,name)] for name in
               ('references','costs','exposures','tolerances')},
            **{name:[[text(v) for v in row] for row in getattr(self,name)] for name in ('menus','demands')},
            kappa=text(self.kappa),radius=text(self.radius),gamma=self.gamma)

    def sha256(self):
        payload=json.dumps(self.canonical_inputs(),sort_keys=True,separators=(',',':')).encode('utf8')
        return hashlib.sha256(payload).hexdigest()

    def encoded_instance(self):
        groups=[]
        for r,q,w,sigma,prices,demands in zip(self.references,self.costs,self.exposures,
                                             self.tolerances,self.menus,self.demands):
            group=[]
            for j,(p,g) in enumerate(zip(prices,demands)):
                if abs(p-r)>sigma*r:
                    continue
                surplus=w*(p-self.kappa*q)*g
                loss=self.radius*surplus if surplus>=0 else -self.radius*surplus/(1-self.radius)
                group.append(Option(value=w*p*g,margin=surplus,uncertainty=loss,price=p,original_index=j))
            if not group:
                raise ValueError('Original menu has no admissible option')
            groups.append(group)
        return PricingInstance(groups,self.gamma)


def original_to_filtered(instance, selected):
    if len(selected)!=instance.n_items:
        raise ValueError('Original selection length must match groups')
    filtered=[]
    for group,j in zip(instance.items,selected):
        if isinstance(j,bool) or not isinstance(j,numbers.Integral):
            raise TypeError('Original IDs must be integers')
        matches=[k for k,option in enumerate(group) if option.original_index==int(j)]
        if len(matches)!=1:
            raise ValueError('Original selection ID is absent or inadmissible')
        filtered.append(matches[0])
    return tuple(filtered)


def build_primitive_certificate(data: ExactPrimitivePricingData, selected_original_ids: Sequence[int],
                                *,include_rounding_provenance=False):
    instance=data.encoded_instance()
    filtered=original_to_filtered(instance,selected_original_ids)
    certificate=build_global_certificate(instance,filtered,
        include_rounding_provenance=include_rounding_provenance)
    return dict(kind='exact_primitive_pricing_global_certificate',schema_version=1,
        primitive_sha256=data.sha256(),selected_original_ids=instance.original_selection(filtered),
        global_certificate=certificate.to_dict())


@dataclass(frozen=True)
class PrimitiveCertificateVerification:
    valid: bool
    errors: tuple
    global_verification: GlobalCertificateVerification | None = None
    selected_original_ids: tuple = ()


def verify_primitive_certificate(data: ExactPrimitivePricingData, payload):
    try:
        if isinstance(payload,str):
            payload=json.loads(payload)
        if not isinstance(payload,dict) or set(payload)!= {
                'kind','schema_version','primitive_sha256','selected_original_ids','global_certificate'}:
            raise ValueError('Malformed primitive certificate fields')
        if payload['kind']!='exact_primitive_pricing_global_certificate' or type(payload['schema_version']) is not int or payload['schema_version']!=1:
            raise ValueError('Unsupported primitive certificate kind/version')
        errors=[]
        if payload['primitive_sha256']!=data.sha256():
            errors.append('primitive input hash mismatch')
        instance=data.encoded_instance()
        if type(payload['global_certificate'].get('schema_version')) is not int:
            raise ValueError('Global certificate schema version must be an integer')
        certificate=GlobalDualCertificate.from_dict(payload['global_certificate'])
        selected=payload['selected_original_ids']
        if not isinstance(selected,list):
            raise ValueError('Original selection must be a list')
        filtered=original_to_filtered(instance,selected)
        if filtered!=certificate.selections:
            errors.append('original/filtered selection mapping mismatch')
        result=verify_global_certificate(instance,certificate)
        errors.extend(result.errors)
        return PrimitiveCertificateVerification(not errors,tuple(errors),result,tuple(selected))
    except Exception as exc:
        return PrimitiveCertificateVerification(False,('malformed primitive certificate: '+str(exc),))
