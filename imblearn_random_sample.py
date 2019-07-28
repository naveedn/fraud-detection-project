# Big HACK: 
# - imblearn is only on python 3.6+, so i manually ported over the code to 2.7 and imported all dependencies
# - maybe create a backport for imblearn to python2.7? @TODO

import abc
from collections import OrderedDict
from sklearn.base import BaseEstimator
from sklearn.preprocessing import label_binarize
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.multiclass import type_of_target
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length
from sklearn.utils import check_random_state
from sklearn.utils import safe_indexing

# compatible with Python 2 *and* 3:
ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()}) 


SAMPLING_KIND = ('over-sampling', 'under-sampling', 'clean-sampling',
                 'ensemble', 'bypass')
TARGET_KIND = ('binary', 'multiclass', 'multilabel-indicator')


class SamplerMixin(BaseEstimator):
    __metaclass__ = abc.ABCMeta
    """Mixin class for samplers with abstract method.
    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    _estimator_type = 'sampler'

    def fit(self, X, y):
        """Check inputs and statistics of the sampler.
        You should use ``fit_resample`` in all cases.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data array.
        y : array-like, shape (n_samples,)
            Target array.
        Returns
        -------
        self : object
            Return the instance itself.
        """
        self._deprecate_ratio()
        X, y, _ = self._check_X_y(X, y)
        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type)
        return self

    def fit_resample(self, X, y):
        """Resample the dataset.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.
        y : array-like, shape (n_samples,)
            Corresponding label for each sample in X.
        Returns
        -------
        X_resampled : {array-like, sparse matrix}, shape \
(n_samples_new, n_features)
            The array containing the resampled data.
        y_resampled : array-like, shape (n_samples_new,)
            The corresponding label of `X_resampled`.
        """
        self._deprecate_ratio()

        check_classification_targets(y)
        X, y, binarize_y = self._check_X_y(X, y)

        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type)

        output = self._fit_resample(X, y)

        if binarize_y:
            y_sampled = label_binarize(output[1], np.unique(y))
            if len(output) == 2:
                return output[0], y_sampled
            return output[0], y_sampled, output[2]
        return output

    #  define an alias for back-compatibility
    fit_sample = fit_resample

    @abc.abstractmethod
    def _fit_resample(self, X, y):
        """Base method defined in each sampler to defined the sampling
        strategy.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.
        y : array-like, shape (n_samples,)
            Corresponding label for each sample in X.
        Returns
        -------
        X_resampled : {ndarray, sparse matrix}, shape \
(n_samples_new, n_features)
            The array containing the resampled data.
        y_resampled : ndarray, shape (n_samples_new,)
            The corresponding label of `X_resampled`
        """
        pass


class BaseSampler(SamplerMixin):
    """Base class for sampling algorithms.
    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    def __init__(self, sampling_strategy='auto', ratio=None):
        self.sampling_strategy = sampling_strategy
        # FIXME: remove in 0.6
        self.ratio = ratio

    @staticmethod
    def _check_X_y(X, y):
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'])
        return X, y, binarize_y

    @property
    def ratio_(self):
        # FIXME: remove in 0.6
        warnings.warn("'ratio' and 'ratio_' are deprecated. Use "
                      "'sampling_strategy' and 'sampling_strategy_' instead.",
                      DeprecationWarning)
        return self.sampling_strategy_

    def _deprecate_ratio(self):
        # both ratio and sampling_strategy should not be set
        if self.ratio is not None:
            deprecate_parameter(self, '0.4', 'ratio', 'sampling_strategy')
            self.sampling_strategy = self.ratio


def _identity(X, y):
    return X, y



class BaseUnderSampler(BaseSampler):
    """Base class for under-sampling algorithms.
    Warning: This class should not be used directly. Use the derive classes
    instead.
    """
    _sampling_type = 'under-sampling'

    _sampling_strategy_docstring = \
        """sampling_strategy : float, str, dict, callable, (default='auto')
        Sampling information to sample the data set.
        - When ``float``, it corresponds to the desired ratio of the number of
          samples in the minority class over the number of samples in the
          majority class after resampling. Therefore, the ratio is expressed as
          :math:`\\alpha_{us} = N_{m} / N_{rM}` where :math:`N_{m}` is the
          number of samples in the minority class and
          :math:`N_{rM}` is the number of samples in the majority class
          after resampling.
          .. warning::
             ``float`` is only available for **binary** classification. An
             error is raised for multi-class classification.
        - When ``str``, specify the class targeted by the resampling. The
          number of samples in the different classes will be equalized.
          Possible choices are:
            ``'majority'``: resample only the majority class;
            ``'not minority'``: resample all classes but the minority class;
            ``'not majority'``: resample all classes but the majority class;
            ``'all'``: resample all classes;
            ``'auto'``: equivalent to ``'not minority'``.
        - When ``dict``, the keys correspond to the targeted classes. The
          values correspond to the desired number of samples for each targeted
          class.
        - When callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples for each class.
        """.rstrip()



class RandomUnderSampler(BaseUnderSampler):
    """Class to perform random under-sampling.
    Under-sample the majority class(es) by randomly picking samples
    with or without replacement.
    Read more in the :ref:`User Guide <controlled_under_sampling>`.
    Parameters
    ----------
    {sampling_strategy}
    return_indices : bool, optional (default=False)
        Whether or not to return the indices of the samples randomly selected.
        .. deprecated:: 0.4
           ``return_indices`` is deprecated. Use the attribute
           ``sample_indices_`` instead.
    {random_state}
    replacement : boolean, optional (default=False)
        Whether the sample is with or without replacement.
    ratio : str, dict, or callable
        .. deprecated:: 0.4
           Use the parameter ``sampling_strategy`` instead. It will be removed
           in 0.6.
    Attributes
    ----------
    sample_indices_ : ndarray, shape (n_new_samples)
        Indices of the samples selected.
        .. versionadded:: 0.4
           ``sample_indices_`` used instead of ``return_indices=True``.
    Notes
    -----
    Supports multi-class resampling by sampling each class independently.
    Supports heterogeneous data as object array containing string and numeric
    data.
    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import \
RandomUnderSampler # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ...  weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> rus = RandomUnderSampler(random_state=42)
    >>> X_res, y_res = rus.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{0: 100, 1: 100}})
    """

    def __init__(self,
                 sampling_strategy='auto',
                 return_indices=False,
                 random_state=None,
                 replacement=False,
                 ratio=None):
        super(RandomUnderSampler, self).__init__(
            sampling_strategy=sampling_strategy, ratio=ratio)
        self.random_state = random_state
        self.return_indices = return_indices
        self.replacement = replacement

    @staticmethod
    def _check_X_y(X, y):
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X = check_array(X, accept_sparse=['csr', 'csc'], dtype=None)
        y = check_array(y, accept_sparse=['csr', 'csc'], dtype=None,
                        ensure_2d=False)
        check_consistent_length(X, y)
        return X, y, binarize_y

    def _fit_resample(self, X, y):
        if self.return_indices:
            deprecate_parameter(self, '0.4', 'return_indices',
                                'sample_indices_')
        random_state = check_random_state(self.random_state)

        idx_under = np.empty((0, ), dtype=int)

        for target_class in np.unique(y):
            if target_class in self.sampling_strategy_.keys():
                n_samples = self.sampling_strategy_[target_class]
                index_target_class = random_state.choice(
                    range(np.count_nonzero(y == target_class)),
                    size=n_samples,
                    replace=self.replacement)
            else:
                index_target_class = slice(None)

            idx_under = np.concatenate(
                (idx_under,
                 np.flatnonzero(y == target_class)[index_target_class]),
                axis=0)

        self.sample_indices_ = idx_under

        if self.return_indices:
            return (safe_indexing(X, idx_under), safe_indexing(y, idx_under),
                    idx_under)
        return safe_indexing(X, idx_under), safe_indexing(y, idx_under)

    def _more_tags(self):
        # TODO: remove the str tag once the following PR is merged:
        # https://github.com/scikit-learn/scikit-learn/pull/14043
        return {'X_types': ['2darray', 'str', 'string'],
                'sample_indices': True}

def check_target_type(y, indicate_one_vs_all=False):
    """Check the target types to be conform to the current samplers.
    The current samplers should be compatible with ``'binary'``,
    ``'multilabel-indicator'`` and ``'multiclass'`` targets only.
    Parameters
    ----------
    y : ndarray,
        The array containing the target.
    indicate_one_vs_all : bool, optional
        Either to indicate if the targets are encoded in a one-vs-all fashion.
    Returns
    -------
    y : ndarray,
        The returned target.
    is_one_vs_all : bool, optional
        Indicate if the target was originally encoded in a one-vs-all fashion.
        Only returned if ``indicate_multilabel=True``.
    """
    type_y = type_of_target(y)
    if type_y == 'multilabel-indicator':
        if np.any(y.sum(axis=1) > 1):
            raise ValueError(
                "Imbalanced-learn currently supports binary, multiclass and "
                "binarized encoded multiclasss targets. Multilabel and "
                "multioutput targets are not supported.")
        y = y.argmax(axis=1)

    return (y, type_y == 'multilabel-indicator') if indicate_one_vs_all else y


def _sampling_strategy_all(y, sampling_type):
    """Returns sampling target by targeting all classes."""
    target_stats = _count_class_sample(y)
    if sampling_type == 'over-sampling':
        n_sample_majority = max(target_stats.values())
        sampling_strategy = {
            key: n_sample_majority - value
            for (key, value) in target_stats.items()
        }
    elif (sampling_type == 'under-sampling' or
          sampling_type == 'clean-sampling'):
        n_sample_minority = min(target_stats.values())
        sampling_strategy = {
            key: n_sample_minority
            for key in target_stats.keys()
        }
    else:
        raise NotImplementedError

    return sampling_strategy

def _sampling_strategy_minority(y, sampling_type):
    """Returns sampling target by targeting the minority class only."""
    target_stats = _count_class_sample(y)
    if sampling_type == 'over-sampling':
        n_sample_majority = max(target_stats.values())
        class_minority = min(target_stats, key=target_stats.get)
        sampling_strategy = {
            key: n_sample_majority - value
            for (key, value) in target_stats.items() if key == class_minority
        }
    elif (sampling_type == 'under-sampling' or
          sampling_type == 'clean-sampling'):
        raise ValueError("'sampling_strategy'='minority' cannot be used with"
                         " under-sampler and clean-sampler.")
    else:
        raise NotImplementedError

    return sampling_strategy


def _sampling_strategy_auto(y, sampling_type):
    """Returns sampling target auto for over-sampling and not-minority for
    under-sampling."""
    if sampling_type == 'over-sampling':
        return _sampling_strategy_not_majority(y, sampling_type)
    elif (sampling_type == 'under-sampling' or
          sampling_type == 'clean-sampling'):
        return _sampling_strategy_not_minority(y, sampling_type)


def _sampling_strategy_dict(sampling_strategy, y, sampling_type):
    """Returns sampling target by converting the dictionary depending of the
    sampling."""
    target_stats = _count_class_sample(y)
    # check that all keys in sampling_strategy are also in y
    set_diff_sampling_strategy_target = (
        set(sampling_strategy.keys()) - set(target_stats.keys()))
    if len(set_diff_sampling_strategy_target) > 0:
        raise ValueError("The {} target class is/are not present in the"
                         " data.".format(set_diff_sampling_strategy_target))
    # check that there is no negative number
    if any(n_samples < 0 for n_samples in sampling_strategy.values()):
        raise ValueError("The number of samples in a class cannot be negative."
                         "'sampling_strategy' contains some negative value: {}"
                         .format(sampling_strategy))
    sampling_strategy_ = {}
    if sampling_type == 'over-sampling':
        n_samples_majority = max(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)
        for class_sample, n_samples in sampling_strategy.items():
            if n_samples < target_stats[class_sample]:
                raise ValueError("With over-sampling methods, the number"
                                 " of samples in a class should be greater"
                                 " or equal to the original number of samples."
                                 " Originally, there is {} samples and {}"
                                 " samples are asked.".format(
                                     target_stats[class_sample], n_samples))
            if n_samples > n_samples_majority:
                warnings.warn("After over-sampling, the number of samples ({})"
                              " in class {} will be larger than the number of"
                              " samples in the majority class (class #{} ->"
                              " {})".format(n_samples, class_sample,
                                            class_majority,
                                            n_samples_majority))
            sampling_strategy_[class_sample] = (
                n_samples - target_stats[class_sample])
    elif sampling_type == 'under-sampling':
        for class_sample, n_samples in sampling_strategy.items():
            if n_samples > target_stats[class_sample]:
                raise ValueError("With under-sampling methods, the number of"
                                 " samples in a class should be less or equal"
                                 " to the original number of samples."
                                 " Originally, there is {} samples and {}"
                                 " samples are asked.".format(
                                     target_stats[class_sample], n_samples))
            sampling_strategy_[class_sample] = n_samples
    elif sampling_type == 'clean-sampling':
        # FIXME: Turn into an error in 0.6
        warnings.warn("'sampling_strategy' as a dict for cleaning methods is "
                      "deprecated and will raise an error in version 0.6. "
                      "Please give a list of the classes to be targeted by the"
                      " sampling.", DeprecationWarning)
        # clean-sampling can be more permissive since those samplers do not
        # use samples
        for class_sample, n_samples in sampling_strategy.items():
            sampling_strategy_[class_sample] = n_samples
    else:
        raise NotImplementedError

    return sampling_strategy_


def _sampling_strategy_list(sampling_strategy, y, sampling_type):
    """With cleaning methods, sampling_strategy can be a list to target the
 class of interest."""
    if sampling_type != 'clean-sampling':
        raise ValueError("'sampling_strategy' cannot be a list for samplers "
                         "which are not cleaning methods.")

    target_stats = _count_class_sample(y)
    # check that all keys in sampling_strategy are also in y
    set_diff_sampling_strategy_target = (
        set(sampling_strategy) - set(target_stats.keys()))
    if len(set_diff_sampling_strategy_target) > 0:
        raise ValueError("The {} target class is/are not present in the"
                         " data.".format(set_diff_sampling_strategy_target))

    return {
        class_sample: min(target_stats.values())
        for class_sample in sampling_strategy
    }


def _sampling_strategy_float(sampling_strategy, y, sampling_type):
    """Take a proportion of the majority (over-sampling) or minority
    (under-sampling) class in binary classification."""
    type_y = type_of_target(y)
    if type_y != 'binary':
        raise ValueError(
            '"sampling_strategy" can be a float only when the type '
            'of target is binary. For multi-class, use a dict.')
    target_stats = _count_class_sample(y)
    if sampling_type == 'over-sampling':
        n_sample_majority = max(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)
        sampling_strategy_ = {
            key: int(n_sample_majority * sampling_strategy - value)
            for (key, value) in target_stats.items() if key != class_majority
        }
        if any([n_samples <= 0 for n_samples in sampling_strategy_.values()]):
            raise ValueError("The specified ratio required to remove samples "
                             "from the minority class while trying to "
                             "generate new samples. Please increase the "
                             "ratio.")
    elif (sampling_type == 'under-sampling'):
        n_sample_minority = min(target_stats.values())
        class_minority = min(target_stats, key=target_stats.get)
        sampling_strategy_ = {
            key: int(n_sample_minority / sampling_strategy)
            for (key, value) in target_stats.items() if key != class_minority
        }
        if any([n_samples > target_stats[target]
               for target, n_samples in sampling_strategy_.items()]):
            raise ValueError("The specified ratio required to generate new "
                             "sample in the majority class while trying to "
                             "remove samples. Please increase the ratio.")
    else:
        raise ValueError("'clean-sampling' methods do let the user "
                         "specify the sampling ratio.")
    return sampling_strategy_


def check_sampling_strategy(sampling_strategy, y, sampling_type, **kwargs):
    """Sampling target validation for samplers.
    Checks that ``sampling_strategy`` is of consistent type and return a
    dictionary containing each targeted class with its corresponding
    number of sample. It is used in :class:`imblearn.base.BaseSampler`.
    Parameters
    ----------
    sampling_strategy : float, str, dict, list or callable,
        Sampling information to sample the data set.
        - When ``float``:
            For **under-sampling methods**, it corresponds to the ratio
            :math:`\\alpha_{us}` defined by :math:`N_{rM} = \\alpha_{us}
            \\times N_{m}` where :math:`N_{rM}` and :math:`N_{m}` are the
            number of samples in the majority class after resampling and the
            number of samples in the minority class, respectively;
            For **over-sampling methods**, it correspond to the ratio
            :math:`\\alpha_{os}` defined by :math:`N_{rm} = \\alpha_{os}
            \\times N_{m}` where :math:`N_{rm}` and :math:`N_{M}` are the
            number of samples in the minority class after resampling and the
            number of samples in the majority class, respectively.
            .. warning::
               ``float`` is only available for **binary** classification. An
               error is raised for multi-class classification and with cleaning
               samplers.
        - When ``str``, specify the class targeted by the resampling. For
          **under- and over-sampling methods**, the number of samples in the
          different classes will be equalized. For **cleaning methods**, the
          number of samples will not be equal. Possible choices are:
            ``'minority'``: resample only the minority class;
            ``'majority'``: resample only the majority class;
            ``'not minority'``: resample all classes but the minority class;
            ``'not majority'``: resample all classes but the majority class;
            ``'all'``: resample all classes;
            ``'auto'``: for under-sampling methods, equivalent to ``'not
            minority'`` and for over-sampling methods, equivalent to ``'not
            majority'``.
        - When ``dict``, the keys correspond to the targeted classes. The
          values correspond to the desired number of samples for each targeted
          class.
          .. warning::
             ``dict`` is available for both **under- and over-sampling
             methods**. An error is raised with **cleaning methods**. Use a
             ``list`` instead.
        - When ``list``, the list contains the targeted classes. It used only
          for **cleaning methods**.
          .. warning::
             ``list`` is available for **cleaning methods**. An error is raised
             with **under- and over-sampling methods**.
        - When callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples for each class.
    y : ndarray, shape (n_samples,)
        The target array.
    sampling_type : str,
        The type of sampling. Can be either ``'over-sampling'``,
        ``'under-sampling'``, or ``'clean-sampling'``.
    kwargs : dict, optional
        Dictionary of additional keyword arguments to pass to
        ``sampling_strategy`` when this is a callable.
    Returns
    -------
    sampling_strategy_converted : dict,
        The converted and validated sampling target. Returns a dictionary with
        the key being the class target and the value being the desired
        number of samples.
    """
    if sampling_type not in SAMPLING_KIND:
        raise ValueError("'sampling_type' should be one of {}. Got '{}'"
                         " instead.".format(SAMPLING_KIND, sampling_type))

    if np.unique(y).size <= 1:
        raise ValueError("The target 'y' needs to have more than 1 class."
                         " Got {} class instead".format(np.unique(y).size))

    if sampling_type in ('ensemble', 'bypass'):
        return sampling_strategy

    if isinstance(sampling_strategy, str):
        if sampling_strategy not in SAMPLING_TARGET_KIND.keys():
            raise ValueError("When 'sampling_strategy' is a string, it needs"
                             " to be one of {}. Got '{}' instead.".format(
                                 SAMPLING_TARGET_KIND, sampling_strategy))
        return OrderedDict(sorted(
            SAMPLING_TARGET_KIND[sampling_strategy](y, sampling_type).items()))
    elif isinstance(sampling_strategy, dict):
        return OrderedDict(sorted(
            _sampling_strategy_dict(sampling_strategy, y, sampling_type)
            .items()))
    elif isinstance(sampling_strategy, list):
        return OrderedDict(sorted(
            _sampling_strategy_list(sampling_strategy, y, sampling_type)
            .items()))
    elif isinstance(sampling_strategy, Real):
        if sampling_strategy <= 0 or sampling_strategy > 1:
            raise ValueError(
                "When 'sampling_strategy' is a float, it should be "
                "in the range (0, 1]. Got {} instead."
                .format(sampling_strategy))
        return OrderedDict(sorted(
            _sampling_strategy_float(sampling_strategy, y, sampling_type)
            .items()))
    elif callable(sampling_strategy):
        sampling_strategy_ = sampling_strategy(y, **kwargs)
        return OrderedDict(sorted(
            _sampling_strategy_dict(sampling_strategy_, y, sampling_type)
            .items()))

    

def _sampling_strategy_majority(y, sampling_type):
    """Returns sampling target by targeting the majority class only."""
    if sampling_type == 'over-sampling':
        raise ValueError("'sampling_strategy'='majority' cannot be used with"
                         " over-sampler.")
    elif (sampling_type == 'under-sampling' or
          sampling_type == 'clean-sampling'):
        target_stats = _count_class_sample(y)
        class_majority = max(target_stats, key=target_stats.get)
        n_sample_minority = min(target_stats.values())
        sampling_strategy = {
            key: n_sample_minority
            for key in target_stats.keys() if key == class_majority
        }
    else:
        raise NotImplementedError

    return sampling_strategy


def _sampling_strategy_not_majority(y, sampling_type):
    """Returns sampling target by targeting all classes but not the
    majority."""
    target_stats = _count_class_sample(y)
    if sampling_type == 'over-sampling':
        n_sample_majority = max(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)
        sampling_strategy = {
            key: n_sample_majority - value
            for (key, value) in target_stats.items() if key != class_majority
        }
    elif (sampling_type == 'under-sampling' or
          sampling_type == 'clean-sampling'):
        n_sample_minority = min(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)
        sampling_strategy = {
            key: n_sample_minority
            for key in target_stats.keys() if key != class_majority
        }
    else:
        raise NotImplementedError

    return sampling_strategy


def _sampling_strategy_not_minority(y, sampling_type):
    """Returns sampling target by targeting all classes but not the
    minority."""
    target_stats = _count_class_sample(y)
    if sampling_type == 'over-sampling':
        n_sample_majority = max(target_stats.values())
        class_minority = min(target_stats, key=target_stats.get)
        sampling_strategy = {
            key: n_sample_majority - value
            for (key, value) in target_stats.items() if key != class_minority
        }
    elif (sampling_type == 'under-sampling' or
          sampling_type == 'clean-sampling'):
        n_sample_minority = min(target_stats.values())
        class_minority = min(target_stats, key=target_stats.get)
        sampling_strategy = {
            key: n_sample_minority
            for key in target_stats.keys() if key != class_minority
        }
    else:
        raise NotImplementedError

    return sampling_strategy


def _sampling_strategy_minority(y, sampling_type):
    """Returns sampling target by targeting the minority class only."""
    target_stats = _count_class_sample(y)
    if sampling_type == 'over-sampling':
        n_sample_majority = max(target_stats.values())
        class_minority = min(target_stats, key=target_stats.get)
        sampling_strategy = {
            key: n_sample_majority - value
            for (key, value) in target_stats.items() if key == class_minority
        }
    elif (sampling_type == 'under-sampling' or
          sampling_type == 'clean-sampling'):
        raise ValueError("'sampling_strategy'='minority' cannot be used with"
                         " under-sampler and clean-sampler.")
    else:
        raise NotImplementedError

    return sampling_strategy



def _count_class_sample(y):
    unique, counts = np.unique(y, return_counts=True)
    return dict(zip(unique, counts))


SAMPLING_TARGET_KIND = {
    'minority': _sampling_strategy_minority,
    'majority': _sampling_strategy_majority,
    'not minority': _sampling_strategy_not_minority,
    'not majority': _sampling_strategy_not_majority,
    'all': _sampling_strategy_all,
    'auto': _sampling_strategy_auto
}