'''
High-level ANOVA user interface using an R-like aov function.

The "aov" function is modified from statsmodels.stats.anova.anova_lm:
https://github.com/statsmodels/statsmodels/blob/main/statsmodels/stats/anova.py

Copyright (C) 2006, Jonathan E. Taylor
All rights reserved.

Copyright (c) 2006-2008 Scipy Developers.
All rights reserved.

Copyright (c) 2009-2018 statsmodels Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of statsmodels nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL STATSMODELS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
'''

# Copyright (C) 2026  Todd Pataky



import warnings
import numpy as np
from . import designs,models
from .. import _datachecks, _reml, _spm, _spmlist


def _as_scalar(x):
    return float( np.asarray(x).flatten()[0] )

def aov(model, contrasts, f_terms, nFactors=1):
    '''
    This code is modified from statsmodels.stats.anova_lm
    '''
    effects = np.asarray( np.dot(model.QT, model.Y) )
    if model.dim==0:
        effects = effects.flatten()
    SS      = np.dot(contrasts.C, effects**2)
    DF      = np.asarray(contrasts.C.sum(axis=1), dtype=int)
    F       = []
    for term0,term1 in f_terms:
        i       = contrasts.term_labels.index(term0)
        ss0,df0 = SS[i], DF[i]
        ms0     = ss0 / df0
        if term1 == 'Error':
            ss1,df1,ms1  = model._SSE, model._dfE, model._MSE
        else:
            i       = contrasts.term_labels.index(term1)
            ss1,df1 = SS[i], DF[i]
            ms1     = (ss1 / df1) if (df1 > 0) else None
        if df1 == 0:
            msg     = '\n\nThe "%s" effect has no error term:  the "%s" term has zero degrees of freedom.\n\nThis happens when the design is saturated -- one observation per cell -- so that nothing is left over with which to estimate error.  Add replicate observations.\n\n' %(term0, term1)
            raise _datachecks.SPM1DError(msg)
        f           = ms0 / ms1
        if model.dim == 0:
            ss0,ss1,ms0,ms1 = [_as_scalar(x) for x in (ss0, ss1, ms0, ms1)]
            F.append( _spm.SPM0D_F(f, (df0,df1), (ss0,ss1), (ms0,ms1), model.eij, model.QT) )
        else:
            if model.roi is not None:
                f   = np.ma.masked_array(f, np.logical_not(model.roi))
            F.append( _spm.SPM_F(f, (df0,df1), model.fwhm, model.resels, model.X, model._beta, model.eij, model.QT, roi=model.roi) )
    return _spmlist.SPMFList( F, nFactors=nFactors )



### ONE-WAY DESIGNS ##############

def anova1(Y, A=None, equal_var=False, roi=None):
    '''
    One-way ANOVA.

    :Parameters (Option 1):
        - *Y* --- A list or tuple of (J x Q) numpy arrays
        - *equal_var* --- If *True*, equal group variance will be assumed


    :Parameters (Option 2):
        - *Y* --- (J x Q) numpy array
        - *A* --- (J x 1) vector of integer group labels
        - *equal_var* --- If *True*, equal group variance will be assumed


    :Returns:
        - F : An **spm1d._spm.SPM_F** instance

    :Example:

    >>> F = spm1d.stats.anova1((Y0,Y1,Y2))
    >>> Fi = F.inference(alpha=0.05)
    >>> Fi.plot()
    '''
    if isinstance(Y, (list,tuple)):
        _datachecks.check('anova1list', Y)
        A   = np.hstack([[i]*y.shape[0] for i,y in enumerate(Y)])
        Y   = np.hstack(Y) if Y[0].ndim==1 else np.vstack(Y)
    else:
        _datachecks.check('anova1', Y, A)
    design  = designs.ANOVA1(A)
    model   = models.LinearModel(Y, design.X, roi=roi)
    model.fit()
    F       = aov(model, design.contrasts, design.f_terms, nFactors=1)[0]
    if not equal_var:
        warnings.warn('\nWARNING:  Non-sphericity corrections for one-way ANOVA are currently approximate and have not been verified.\n', UserWarning, stacklevel=2)
        Y,X,r = model.Y, model.X, model.eij
        Q,C   = design.A.get_Q(), design.contrasts.C.T
        F.df  = _reml.estimate_df_anova1(Y, X, r, Q, C)
    return F



def anova1rm(Y, A, SUBJ, equal_var=True, roi=None, _force_approx0D=False):
    '''
    One-way repeated-measures ANOVA.

    :Parameters:
        - *Y* --- (J x Q) numpy array
        - *A* --- (J x 1) vector of integer group labels
        - *SUBJ* --- (J x 1) vector of subject labels
        - *equal_var* --- If *True*, equal group variance will be assumed


    :Returns:
        - F : An **spm1d._spm.SPM_F** instance

    :Example:

    >>> Y = np.random.randn(9, 101)
    >>> A = np.array([1,1,1, 2,2,2, 3,3,3])
    >>> SUBJ = np.array([1,2,3, 1,2,3, 1,2,3])
    >>> F = spm1d.stats.anova1(Y, A, SUBJ)
    >>> Fi = F.inference(alpha=0.05)
    >>> Fi.plot()
    '''
    if not equal_var:
        raise( NotImplementedError( 'Non-sphericity corrections are not yet implemented. Set "equal_var" to "True" to force an assumption of equal variance.' ) )
    design  = designs.ANOVA1rm(A, SUBJ)
    model   = models.LinearModel(Y, design.X, roi=roi)
    if ((model.dim == 1) or _force_approx0D)   and   ( design.check_for_single_responses(dim=model.dim) ):
        model.fit( approx_residuals=design.contrasts.C[:3] )
    else:
        model.fit( )
    F       = aov(model, design.contrasts, design.f_terms, nFactors=1)[0]
    return F






### TWO-WAY DESIGNS ##############

def _set_labels(FF, design):
    FF.set_design_label( design.__class__.__name__ )
    FF.set_effect_labels( design.effect_labels )
    # [F.set_effect_label(label)  for F,label in zip(FF, design.effect_labels)]



def anova2(Y, A, B, equal_var=True, roi=None):
    '''
    Two-way ANOVA.

    :Parameters:
        - *Y* --- (J x Q) numpy array
        - *A* --- (J x 1) vector of integer labels for Factor A
        - *B* --- (J x 1) vector of integer labels for Factor B
        - *equal_var* --- If *True*, equal group variance will be assumed

    :Returns:
        - List of three **spm1d._spm.SPM_F** instances in the following order:
            1. Main effect A
            2. Main effect B
            3. Interaction AB
    '''
    if not equal_var:
        raise( NotImplementedError( 'Non-sphericity corrections are not yet implemented. Set "equal_var" to "True" to force an assumption of equal variance.' ) )
    design  = designs.ANOVA2(A, B)
    model   = models.LinearModel(Y, design.X, roi=roi)
    model.fit()
    F       = aov(model, design.contrasts, design.f_terms, nFactors=2)
    _set_labels( F, design )

    # if not equal_var:
        # Y,X,r   = model.Y, model.X, model.eij
        # QA,QB,C = design.A.get_Q(), design.B.get_Q(), design.contrasts.C.T
        # Q       = QA + QB
        # u1,u2   = _reml.estimate_df_anova2(Y, X, r, Q, C)
        # for ff,u in zip(f,u1):
        #     ff.df = u,u2
    return F


def anova2nested(Y, A, B, equal_var=True, roi=None):
    '''
    Two-way nested ANOVA.

    :Parameters:
        - *Y* --- (J x Q) numpy array
        - *A* --- (J x 1) vector of integer labels for Factor A
        - *B* --- (J x 1) vector of integer labels for Factor B (nested in A)
        - *equal_var* --- If *True*, equal group variance will be assumed

    :Returns:
        - List of two **spm1d._spm.SPM_F** instances in the following order:
            1. Main effect A
            2. Main effect B

    :Note:
        - there is no interaction term in nested designs.
    '''
    if equal_var is not True:
        raise( NotImplementedError('Non-sphericity correction not implemented. To continue you must assume equal variance and set "equal_var=True".') )
    design  = designs.ANOVA2nested(A, B)
    model   = models.LinearModel(Y, design.X, roi=roi)
    model.fit()
    F       = aov(model, design.contrasts, design.f_terms, nFactors=2)
    _set_labels( F, design )
    return F



def anova2rm(Y, A, B, SUBJ, equal_var=True, roi=None, _force_approx0D=False):
    '''
    Two-way repeated-measures ANOVA.

    :Parameters:
        - *Y* --- (J x Q) numpy array
        - *A* --- (J x 1) vector of integer labels for Factor A
        - *B* --- (J x 1) vector of integer labels for Factor B
        - *SUBJ* --- (J x 1) vector of integer subject labels
        - *equal_var* --- If *True*, equal group variance will be assumed

    :Returns:
        - List of three **spm1d._spm.SPM_F** instances in the following order:
            1. Main effect A
            2. Main effect B
            3. Interaction AB

    :Note:
        - Non-sphericity correction not implemented. Equal variance must be assumed by setting "equal_var=True".
    '''
    if equal_var is not True:
        raise( NotImplementedError('Non-sphericity correction not implemented. To continue you must assume equal variance and set "equal_var=True".') )
    design  = designs.ANOVA2rm(A, B, SUBJ)
    model   = models.LinearModel(Y, design.X, roi=roi)
    if ((model.dim == 1) or _force_approx0D)   and   ( design.check_for_single_responses(dim=model.dim) ):
        model.fit( approx_residuals=design.contrasts.C[:5] )
    else:
        model.fit( )
    F       = aov(model, design.contrasts, design.f_terms, nFactors=2)
    _set_labels( F, design )
    # if not equal_var:
    #     Y,X,r   = solver.Y, solver.X, solver.eij
    #     QA,QB,C = design.A.get_Q(), design.B.get_Q(), [c.C.T for c in design.contrasts]
    #     Q       = QA + QB
    #     u1,u2   = _reml.estimate_df_anova2(Y, X, r, Q, C)
    #     for ff,u in zip(f,u1):
    #         ff.df = u,u2
    return F


def anova2onerm(Y, A, B, SUBJ, equal_var=True, roi=None, _force_approx0D=False):
    '''
    Two-way ANOVA with repeated-measures on one factor.

    :Parameters:
        - *Y* --- (J x Q) numpy array
        - *A* --- (J x 1) vector of integer labels for Factor A
        - *B* --- (J x 1) vector of integer labels for Factor B (the repeated-measures factor)
        - *SUBJ* --- (J x 1) vector of integer subject labels
        - *equal_var* --- If *True*, equal group variance will be assumed

    :Returns:
        - List of three **spm1d._spm.SPM_F** instances in the following order:
            1. Main effect A
            2. Main effect B
            3. Interaction AB

    :Note:
        - Non-sphericity correction not implemented. Equal variance must be assumed by setting "equal_var=True".
    '''
    if equal_var is not True:
        raise( NotImplementedError('Non-sphericity correction not implemented. To continue you must assume equal variance and set "equal_var=True".') )
    design  = designs.ANOVA2onerm(A, B, SUBJ)
    model   = models.LinearModel(Y, design.X, roi=roi)
    if ((model.dim == 1) or _force_approx0D)   and   ( design.check_for_single_responses(dim=model.dim) ):
        model.fit( approx_residuals=design.contrasts.C[:5] )
    else:
        model.fit( )
    F       = aov(model, design.contrasts, design.f_terms, nFactors=2)
    _set_labels( F, design )
    return F




### THREE-WAY DESIGNS ##############


def anova3(Y, A, B, C, equal_var=True, roi=None):
    '''
    Three-way ANOVA.

    :Parameters:
        - *Y* --- (J x Q) numpy array
        - *A* --- (J x 1) vector of integer labels for Factor A
        - *B* --- (J x 1) vector of integer labels for Factor B
        - *C* --- (J x 1) vector of integer labels for Factor C
        - *equal_var* --- If *True*, equal group variance will be assumed

    :Returns:
        - List of seven **spm1d._spm.SPM_F** instances in the following order:
            1. Main effect A
            2. Main effect B
            3. Main effect C
            4. Interaction AB
            5. Interaction AC
            6. Interaction BC
            7. Interaction ABC
    '''
    if equal_var is not True:
        raise( NotImplementedError('Non-sphericity correction not implemented. To continue you must assume equal variance and set "equal_var=True".') )
    design  = designs.ANOVA3(A, B, C)
    model   = models.LinearModel(Y, design.X, roi=roi)
    model.fit()
    F       = aov(model, design.contrasts, design.f_terms, nFactors=3)
    _set_labels( F, design )
    return F
    # if not equal_var:
    #     Y,X,r   = solver.Y, solver.X, solver.eij
    #     QA,QB,QC,C = design.A.get_Q(), design.B.get_Q(), design.C.get_Q(), [c.C.T for c in design.contrasts]
    #     Q       = QA + QB + QC
    #     u1,u2   = _reml.estimate_df_anova2(Y, X, r, Q, C)
    #     for ff,u in zip(f,u1):
    #         ff.df = u,u2
    # return f

def anova3nested(Y, A, B, C, equal_var=True, roi=None):
    '''
    Three-way fully nested ANOVA.

    :Parameters:
        - *Y* --- (J x Q) numpy array
        - *A* --- (J x 1) vector of integer labels for Factor A
        - *B* --- (J x 1) vector of integer labels for Factor B (nested in A)
        - *C* --- (J x 1) vector of integer labels for Factor C (nested in B)
        - *equal_var* --- If *True*, equal group variance will be assumed

    :Returns:
        - List of three **spm1d._spm.SPM_F** instances in the following order:
            1. Main effect A
            2. Main effect B
            3. Main effect C

    :Note:
        - there are no interaction terms in fully-nested designs.
    '''
    if equal_var is not True:
        raise( NotImplementedError('Non-sphericity correction not implemented. To continue you must assume equal variance and set "equal_var=True".') )
    design  = designs.ANOVA3nested(A, B, C)
    model   = models.LinearModel(Y, design.X, roi=roi)
    model.fit()
    F       = aov(model, design.contrasts, design.f_terms, nFactors=3)
    _set_labels( F, design )
    return F


def anova3rm(Y, A, B, C, SUBJ, equal_var=True, roi=None, _force_approx0D=False):
    '''
    Three-way ANOVA (repeated measures on all factors).

    :Parameters:
        - *Y* --- (J x Q) numpy array
        - *A* --- (J x 1) vector of integer labels for Factor A
        - *B* --- (J x 1) vector of integer labels for Factor B
        - *C* --- (J x 1) vector of integer labels for Factor C
        - *equal_var* --- If *True*, equal group variance will be assumed

    :Returns:
        - List of seven **spm1d._spm.SPM_F** instances in the following order:
            1. Main effect A
            2. Main effect B
            3. Main effect C
            4. Interaction AB
            5. Interaction AC
            6. Interaction BC
            7. Interaction ABC
    '''
    if equal_var is not True:
        raise( NotImplementedError('Non-sphericity correction not implemented. To continue you must assume equal variance and set "equal_var=True".') )
    design  = designs.ANOVA3rm(A, B, C, SUBJ)
    model   = models.LinearModel(Y, design.X, roi=roi)
    if ((model.dim == 1) or _force_approx0D)   and   ( design.check_for_single_responses(dim=model.dim) ):
        model.fit( approx_residuals=design.contrasts.C[:8] )
    else:
        model.fit( )
    F       = aov(model, design.contrasts, design.f_terms, nFactors=3)
    _set_labels( F, design )
    return F



def anova3onerm(Y, A, B, C, SUBJ, equal_var=True, roi=None, _force_approx0D=False):
    '''
    Three-way ANOVA with repeated-measures on one factor.

    :Parameters:
        - *Y* --- (J x Q) numpy array
        - *A* --- (J x 1) vector of integer labels for Factor A
        - *B* --- (J x 1) vector of integer labels for Factor B
        - *C* --- (J x 1) vector of integer labels for Factor C (the repeated-measures factor)
        - *SUBJ* --- (J x 1) vector of integer subject labels
        - *equal_var* --- If *True*, equal group variance will be assumed

    :Returns:
        - List of seven **spm1d._spm.SPM_F** instances in the following order:
            1. Main effect A
            2. Main effect B
            3. Main effect C
            4. Interaction AB
            5. Interaction AC
            6. Interaction BC
            7. Interaction ABC

    :Note:
        - Non-sphericity correction not implemented. Equal variance must be assumed by setting "equal_var=True".
    '''
    if equal_var is not True:
        raise( NotImplementedError('Non-sphericity correction not implemented. To continue you must assume equal variance and set "equal_var=True".') )
    design  = designs.ANOVA3onerm(A, B, C, SUBJ)
    model   = models.LinearModel(Y, design.X, roi=roi)
    if ((model.dim == 1) or _force_approx0D)   and   ( design.check_for_single_responses(dim=model.dim) ):
        model.fit( approx_residuals=design.contrasts.C[:8] )
    else:
        model.fit( )
    F       = aov(model, design.contrasts, design.f_terms, nFactors=3)
    _set_labels( F, design  )
    return F



def anova3tworm(Y, A, B, C, SUBJ, equal_var=True, roi=None, _force_approx0D=False):
    '''
    Three-way ANOVA with repeated-measures on two factors.

    :Parameters:
        - *Y* --- (J x Q) numpy array
        - *A* --- (J x 1) vector of integer labels for Factor A
        - *B* --- (J x 1) vector of integer labels for Factor B (a repeated-measures factor)
        - *C* --- (J x 1) vector of integer labels for Factor C (a repeated-measures factor)
        - *SUBJ* --- (J x 1) vector of integer subject labels
        - *equal_var* --- If *True*, equal group variance will be assumed

    :Returns:
        - List of seven **spm1d._spm.SPM_F** instances in the following order:
            1. Main effect A
            2. Main effect B
            3. Main effect C
            4. Interaction AB
            5. Interaction AC
            6. Interaction BC
            7. Interaction ABC

    :Note:
        - Non-sphericity correction not implemented. Equal variance must be assumed by setting "equal_var=True".
    '''
    if equal_var is not True:
        raise( NotImplementedError('Non-sphericity correction not implemented. To continue you must assume equal variance and set "equal_var=True".') )
    design  = designs.ANOVA3tworm(A, B, C, SUBJ)
    model   = models.LinearModel(Y, design.X, roi=roi)
    if ((model.dim == 1) or _force_approx0D)   and   ( design.check_for_single_responses(dim=model.dim) ):
        model.fit( approx_residuals=design.contrasts.C[:8] )
    else:
        model.fit( )
    F       = aov(model, design.contrasts, design.f_terms, nFactors=3)
    _set_labels( F, design )
    return F


