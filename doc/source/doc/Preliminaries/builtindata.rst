
.. _label-BuiltinData:

Built-in datasets
---------------------------------



**spm1d** comes packages with a variety of 0D and 1D datasets located in **./spm1d/data**.

This document provides a brief overview of the built-in datasets.

To explore the datasets please use the scripts in **spm1d/examples**. 


Loading datasets
============================

All datasets can be accessed using the following syntax:

	>>> dataset = spm1d.data.uv0d.anova1.Cars()
	>>> data = dataset.get_data()

where:

	* spm1d.data.\ ``uv0d`` refers to univariate zero-dimensional data
	* spm1d.data.uv0d.\ ``anova1`` refers to the type of statistical test
	* dataset.\ ``get_data()`` extracts only the variable needed for the given test
	
NOTES:

* For access to univariate 1D datasets, and multivariate 0D and 1D datasets, use **spm1d.data.uv1d**, **spm1d.data.mv0d** and **spm1d.data.mv1d**, respectively.
* The datasets contain a variety of other variables including web links and expected results.


Dataset details
============================

**spm1d** no longer provides detailed dataset descriptions.

Instead users should consult the references and web links provided with each dataset as follows:

	>>> dataset = spm1d.data.uv0d.anova1.Cars()
	>>> print( dataset )
	
The result will look something like this::

	Dataset
		Name      : "Cars"
		Design    :  One-way ANOVA
		Data dim  :  0
		Web       :  http://cba.ualr.edu/smartstat/topics/anova/example.pdf
	(Expected results)
		F         :  25.17
		df        :  (2, 6)
		p         :  0.001207


Checking expected results
============================

The expected results shown in the example above can be corroborated against **spm1d**'s calculations as follows:

	>>> dataset = spm1d.data.uv0d.anova1.Cars()
	>>> Y,A     = dataset.get_data()
	>>> F       = spm1d.stats.anova1(Y, A, equal_var=True)
	>>> Fi      = F.inference(0.05)
	>>> print( Fi )
	
Here *Y* and *A* are both 9-component vectors, where *Y* represents the observations and where *A* continains integers which indicate groups.


The **spm1d** result is::

	SPM{F} (0D) inference
		SPM.z        :  25.17541
		SPM.df       :  (2, 6)
	Inference:
		SPM.alpha    :  0.050
		SPM.zstar    :  5.14325
		SPM.h0reject :  True
		SPM.p        :  0.00121

Note that the F statistic (SPM.z), degrees of freedom (SPM.df) and p value (SPM.p) match the expected dataset results.

The critical test statistic threshold at alpha (SPM.zstar) is not typically reported, but is an essential component of 1D analyses so is also presented in 0D results for comparisons between 0D and 1D critical thresholds.  


Notes on 1D datasets
============================


.. warning:: Only 0D datasets contain expected results.

	For one-dimensional result verifications refer to the references provided.
	
	For example:

	>>>  dataset      = spm1d.data.uv1d.anova2.Besier2009kneeflexion()
	>>>  Y,A,B        = dataset.get_data()  #A:foot, B:speed
	>>>  print( dataset )
	
	This will yield::

	
		Dataset
			Name      : "Besier2009kneeflexion"
			Design    :  Two-way ANOVA
			Data dim  :  1
			Reference :  Besier, T. F., Fredericson, M., Gold, G. E., Beaupré, G. S., & Delp, S. L. (2009). Knee muscle forces during walking and running in patellofemoral pain patients and pain-free controls. Journal of Biomechanics, 42(7), 898–905. http://doi.org/10.1016/j.jbiomech.2009.01.032
			Web       :  https://simtk.org/home/muscleforces
			Data file :  /path/to/spm1d/data/datafiles/Besier2009kneeflexion.npz
			Results   :  Pataky, T. C., Vanrenterghem, J., & Robinson, M. A. (2015). Two-way ANOVA for scalar trajectories, with experimental evidence of non-phasic interactions. Journal of Biomechanics, 48(1), 186–189. http://doi.org/10.1016/j.jbiomech.2014.10.013
		(Expected results)
			F          :  None
			df         :  None
			p          :  None







