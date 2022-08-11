#  Copyright (c) 2017-2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A ``unischema`` is a data structure definition which can be rendered as native schema/data-types objects
in several different python libraries. Currently supported are pyspark, tensorflow, and numpy.
"""
import copy
import re
import sys
import warnings
from collections import namedtuple, OrderedDict
from decimal import Decimal
from typing import Dict, Any, Tuple, Optional, NamedTuple

import numpy as np
import pyarrow as pa
import six
from pyarrow.lib import ListType
from pyarrow.lib import StructType as pyStructType
from six import string_types

# _UNISCHEMA_FIELD_ORDER available values are 'preserve_input_order' or 'alphabetical'
# Current default behavior is 'preserve_input_order', the legacy behavior is 'alphabetical', which is deprecated and
# will be removed in future versions.
_UNISCHEMA_FIELD_ORDER = 'preserve_input_order'


def _fields_as_tuple(field):
    """Common representation of UnischemaField for equality and hash operators.
    Defined outside class because the method won't be accessible otherwise.

    Today codec instance also responsible for defining spark dataframe type. This knowledge should move
    to a different class in order to support backends other than Apache Parquet. For now we ignore the codec
    in comparison. From the checks does not seem that it should negatively effect the rest of the code.
    """
    return (field.name, field.numpy_dtype, field.shape, field.nullable)


class UnischemaField(NamedTuple):
    """A type used to describe a single field in the schema:

    - name: name of the field.
    - numpy_dtype: a numpy ``dtype`` reference
    - shape: shape of the multidimensional array. None value is used to define a dimension with variable number of
             elements. E.g. ``(None, 3)`` defines a point cloud with three coordinates but unknown number of points.
    - codec: An instance of a codec object used to encode/decode data during serialization
             (e.g. ``CompressedImageCodec('png')``)
    - nullable: Boolean indicating whether field can be None

    A field is considered immutable, so we override both equality and hash operators for consistency
    and efficiency.
    """

    name: str
    numpy_dtype: Any
    shape: Tuple[Optional[int], ...]
    codec: Optional[Any] = None
    nullable: Optional[bool] = False

    def __eq__(self, other):
        """Comparing field objects via default namedtuple __repr__ representation doesn't work due to
        codec object ID changing when unpickled.

        Instead, compare all field attributes, except for codec type.

        Future: Give codec a mime identifier.
        """
        return _fields_as_tuple(self) == _fields_as_tuple(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(_fields_as_tuple(self))


class _NamedtupleCache(object):
    """_NamedtupleCache makes sure the same instance of a namedtuple is returned for a given schema and a set of
     fields. This makes comparison between types possible. For example, `tf.data.Dataset.concatenate` implementation
     compares types to make sure two datasets can be concatenated."""
    _store: Dict[str, Any] = dict()

    @staticmethod
    def get(parent_schema_name, field_names):
        """Creates a nametuple with field_names as values. Returns an existing instance if was already created.

        :param parent_schema_name: Schema name becomes is part of the cache key
        :param field_names: defines names of the fields in the namedtuple created/returned. Also part of the cache key.
        :return: A namedtuple with field names defined by `field_names`
        """
        # Cache key is a combination of schema name and all field names
        if _UNISCHEMA_FIELD_ORDER.lower() == 'alphabetical':
            field_names = list(sorted(field_names))
        else:
            field_names = list(field_names)
        key = ' '.join([parent_schema_name] + field_names)
        print("here!!!!" * 20)
        print(_NamedtupleCache._store)
        print("field names TUPLECACHE GET" * 8)
        print(field_names)
        if key not in _NamedtupleCache._store:
            _NamedtupleCache._store[key] = _new_gt_255_compatible_namedtuple(
                '{}_view'.format(parent_schema_name), field_names)
        print("here?" * 20)
        print(_NamedtupleCache._store)
        print("here???" * 20)
        print(_NamedtupleCache._store[key])
        print(key)
        return _NamedtupleCache._store[key]


def _new_gt_255_compatible_namedtuple(*args, **kwargs):
    # Between Python 3 - 3.6.8 namedtuple can not have more than 255 fields. We use
    # our custom version of namedtuple in these cases
    if six.PY3 and sys.version_info[1] < 7:
        # Have to hide the codeblock in namedtuple_gt_255_fields.py from Python 2 interpreter
        # as it would trigger "unqualified exec is not allowed in function" SyntaxError
        from bastardizedpetastorm.namedtuple_gt_255_fields import namedtuple_gt_255_fields
        namedtuple_cls = namedtuple_gt_255_fields
    else:  # Python 2 or Python 3.7 and later.
        from bastardizedpetastorm.namedtuple_gt_255_fields import namedtuple_gt_255_fields
        namedtuple_cls = namedtuple_gt_255_fields

    return namedtuple_cls(*args, **kwargs)


def _numpy_to_spark_mapping():
    """Returns a mapping from numpy to pyspark.sql type. Caches the mapping dictionary inorder to avoid instantiation
    of multiple objects in each call."""

    # Refer to the attribute of the function we use to cache the map using a name in the variable instead of a 'dot'
    # notation to avoid copy/paste/typo mistakes
    cache_attr_name = 'cached_numpy_to_pyspark_types_map'
    if not hasattr(_numpy_to_spark_mapping, cache_attr_name):
        import pyspark.sql.types as T

        setattr(_numpy_to_spark_mapping, cache_attr_name,
                {
                    np.int8: T.ByteType(),
                    np.uint8: T.ShortType(),
                    np.int16: T.ShortType(),
                    np.uint16: T.IntegerType(),
                    np.int32: T.IntegerType(),
                    np.int64: T.LongType(),
                    np.float32: T.FloatType(),
                    np.float64: T.DoubleType(),
                    np.string_: T.StringType(),
                    np.str_: T.StringType(),
                    np.unicode_: T.StringType(),
                    np.bool_: T.BooleanType(),
                })

    return getattr(_numpy_to_spark_mapping, cache_attr_name)


# TODO: Changing fields in this class or the UnischemaField will break reading due to the schema being pickled next to
# the dataset on disk
def _field_spark_dtype(field):
    if field.codec is None:
        if field.shape == ():
            spark_type = _numpy_to_spark_mapping().get(field.numpy_dtype, None)
            if not spark_type:
                raise ValueError('Was not able to map type {} to a spark type.'.format(str(field.numpy_dtype)))
        else:
            raise ValueError('An instance of non-scalar UnischemaField \'{}\' has codec set to None. '
                             'Don\'t know how to guess a Spark type for it'.format(field.name))
    else:
        spark_type = field.codec.spark_dtype()

    return spark_type


class Unischema(object):
    """Describes a schema of a data structure which can be rendered as native schema/data-types objects
    in several different python libraries. Currently supported are pyspark, tensorflow, and numpy.
    """

    def __init__(self, name, fields):
        """Creates an instance of a Unischema object.

        :param name: name of the schema
        :param fields: a list of ``UnischemaField`` instances describing the fields. The element order in the list
            represent the schema field order.
        """
        self._name = name
        if _UNISCHEMA_FIELD_ORDER.lower() == 'alphabetical':
            fields = sorted(fields, key=lambda t: t.name)

        self._fields = OrderedDict([(f.name, f) for f in fields])
        # Generates attributes named by the field names as an access syntax sugar.
        for f in fields:
            if not hasattr(self, f.name):
                setattr(self, f.name, f)
            else:
                warnings.warn(('Can not create dynamic property {} because it conflicts with an existing property of '
                               'Unischema').format(f.name))

    def create_schema_view(self, fields):
        """Creates a new instance of the schema using a subset of fields.

        Fields can be either UnischemaField objects or regular expression patterns.

        If one of the fields does not exist in this schema, an error is raised.

        The example returns a schema, with field_1 and any other field matching ``other.*$`` pattern.

        >>> SomeSchema.create_schema_view(
        >>>     [SomeSchema.field_1,
        >>>      'other.*$'])

        :param fields: A list of UnischemaField objects and/or regular expressions
        :return: a new view of the original schema containing only the supplied fields
        """

        # Split fields parameter to regex pattern strings and UnischemaField objects
        regex_patterns = [field for field in fields if isinstance(field, string_types)]
        # We can not check type against UnischemaField because the artifact introduced by
        # pickling, since depickled UnischemaField are of type collections.UnischemaField
        # while withing depickling they are of petastorm.unischema.UnischemaField
        # Since UnischemaField is a tuple, we check against it since it is invariant to
        # pickling
        unischema_field_objects = [field for field in fields if isinstance(field, tuple)]
        if len(unischema_field_objects) + len(regex_patterns) != len(fields):
            raise ValueError('Elements of "fields" must be either a string (regular expressions) or '
                             'an instance of UnischemaField class.')

        # For fields that are specified as instances of Unischema: make sure that this schema contains fields
        # with these names.
        exact_field_names = [field.name for field in unischema_field_objects]
        unknown_field_names = set(exact_field_names) - set(self.fields.keys())
        if unknown_field_names:
            raise ValueError('field {} does not belong to the schema {}'.format(unknown_field_names, self))

        # Do not use instances of Unischema fields passed as an argument as it could contain codec/shape
        # info that is different from the one stored in this schema object
        exact_fields = [self._fields[name] for name in exact_field_names]
        view_fields = exact_fields + match_unischema_fields(self, regex_patterns)

        return Unischema('{}_view'.format(self._name), view_fields)

    def _get_namedtuple(self):
        return _NamedtupleCache.get(self._name, self._fields.keys())

    def __str__(self):
        """Represent this as the following form:

        >>> Unischema(name, [
        >>>   UnischemaField(name, numpy_dtype, shape, codec, field_nullable),
        >>>   ...
        >>> ])
        """
        fields_str = ''
        for field in self._fields.values():
            fields_str += '  {}(\'{}\', {}, {}, {}, {}),\n'.format(type(field).__name__, field.name,
                                                                   field.numpy_dtype.__name__,
                                                                   field.shape, field.codec, field.nullable)
        return '{}({}, [\n{}])'.format(type(self).__name__, self._name, fields_str)

    @property
    def fields(self):
        return self._fields

    def as_spark_schema(self):
        """Returns an object derived from the unischema as spark schema.

        Example:

        >>> spark.createDataFrame(dataset_rows,
        >>>                       SomeSchema.as_spark_schema())
        """
        # Lazy loading pyspark to avoid creating pyspark dependency on data reading code path
        # (currently works only with make_batch_reader)
        import pyspark.sql.types as sql_types

        schema_entries = []
        for field in self._fields.values():
            spark_type = _field_spark_dtype(field)
            schema_entries.append(sql_types.StructField(field.name, spark_type, field.nullable))

        return sql_types.StructType(schema_entries)

    def make_namedtuple(self, **kargs):
        """Returns schema as a namedtuple type intialized with arguments passed to this method.

        Example:

        >>> some_schema.make_namedtuple(field1=10, field2='abc')
        """
        # TODO(yevgeni): verify types
        typed_dict = dict()
        for key in kargs.keys():
            if kargs[key] is not None:
                typed_dict[key] = kargs[key]
            else:
                typed_dict[key] = None
        print("typed_dict" * 20)
        print(typed_dict)
        # need to change **typed_dict
        lol = ['a0_fbisc', 'a1_fbisc', 'a2_fbisc', 'a3_fbisc', 'a4_fbisc', 'a5_fbisc', 'a6_fbisc', 'a7_fbisc', 'a8_fbisc', 'a9_fbisc', 'a10_fbisc', 'a11_fbisc', 'a12_fbisc', 'a13_fbisc', 'a14_fbisc', 'a15_fbisc', 'a16_fbisc', 'a17_fbisc', 'a18_fbisc', 'a19_fbisc', 'a20_fbisc', 'a21_fbisc', 'a22_fbisc', 'a23_fbisc', 'a24_fbisc', 'a25_fbisc', 'a26_fbisc', 'a27_fbisc', 'a28_fbisc', 'a29_fbisc', 'a30_fbisc', 'a31_fbisc', 'a32_fbisc', 'a33_fbisc', 'a34_fbisc', 'a35_fbisc', 'a36_fbisc', 'a37_fbisc', 'a38_fbisc', 'a39_fbisc', 'a40_fbisc', 'a41_fbisc', 'a42_fbisc', 'a43_fbisc', 'a44_fbisc', 'a45_fbisc', 'a46_fbisc', 'a47_fbisc', 'a48_fbisc', 'a49_fbisc', 'a50_fbisc', 'a51_fbisc', 'a52_fbisc', 'a53_fbisc', 'a54_fbisc', 'a55_fbisc', 'a56_fbisc', 'a57_fbisc', 'a58_fbisc', 'a59_fbisc', 'a60_fbisc', 'a61_fbisc', 'a62_fbisc', 'a63_fbisc', 'a64_fbisc', 'a65_fbisc', 'a66_fbisc', 'a67_fbisc', 'a68_fbisc', 'a69_fbisc', 'a70_fbisc', 'a71_fbisc', 'a72_fbisc', 'a73_fbisc', 'a74_fbisc', 'a75_fbisc', 'a76_fbisc', 'a77_fbisc', 'a78_fbisc', 'a79_fbisc', 'a80_fbisc', 'a81_fbisc', 'a82_fbisc', 'a83_fbisc', 'a84_fbisc', 'a85_fbisc', 'a86_fbisc', 'a87_fbisc', 'a88_fbisc', 'a89_fbisc', 'a90_fbisc', 'a91_fbisc', 'a92_fbisc', 'a93_fbisc', 'a94_fbisc', 'a95_fbisc', 'a96_fbisc', 'a97_fbisc', 'a98_fbisc', 'a99_fbisc', 'a100_fbisc', 'a101_fbisc', 'a102_fbisc', 'a103_fbisc', 'a104_fbisc', 'a105_fbisc', 'a106_fbisc', 'a107_fbisc', 'a108_fbisc', 'a109_fbisc', 'a110_fbisc', 'a111_fbisc', 'a112_fbisc', 'a113_fbisc', 'a114_fbisc', 'a115_fbisc', 'a116_fbisc', 'a117_fbisc', 'a118_fbisc', 'a119_fbisc', 'a120_fbisc', 'a121_fbisc', 'a122_fbisc', 'a123_fbisc', 'a124_fbisc', 'a125_fbisc', 'a126_fbisc', 'a127_fbisc', 'a128_fbisc', 'a129_fbisc', 'a130_fbisc', 'a131_fbisc', 'a132_fbisc', 'a133_fbisc', 'a134_fbisc', 'a135_fbisc', 'a136_fbisc', 'a137_fbisc', 'a138_fbisc', 'a139_fbisc', 'a140_fbisc', 'a141_fbisc', 'a142_fbisc', 'a143_fbisc', 'a144_fbisc', 'a145_fbisc', 'a146_fbisc', 'a147_fbisc', 'a148_fbisc', 'a149_fbisc', 'a150_fbisc', 'a151_fbisc', 'a152_fbisc', 'a153_fbisc', 'a154_fbisc', 'a155_fbisc', 'a156_fbisc', 'a157_fbisc', 'a158_fbisc', 'a159_fbisc', 'a160_fbisc', 'a161_fbisc', 'a162_fbisc', 'a163_fbisc', 'a164_fbisc', 'a165_fbisc', 'a166_fbisc', 'a167_fbisc', 'a168_fbisc', 'a169_fbisc', 'a170_fbisc', 'a171_fbisc', 'a172_fbisc', 'a173_fbisc', 'a174_fbisc', 'a175_fbisc', 'a176_fbisc', 'a177_fbisc', 'a178_fbisc', 'a179_fbisc', 'a180_fbisc', 'a181_fbisc', 'a182_fbisc', 'a183_fbisc', 'a184_fbisc', 'a185_fbisc', 'a186_fbisc', 'a187_fbisc', 'a188_fbisc', 'a189_fbisc', 'a190_fbisc', 'a191_fbisc', 'a192_fbisc', 'a193_fbisc', 'a194_fbisc', 'a195_fbisc', 'a196_fbisc', 'a197_fbisc', 'a198_fbisc', 'a199_fbisc', 'a200_fbisc', 'a201_fbisc', 'a202_fbisc', 'a203_fbisc', 'a204_fbisc', 'a205_fbisc', 'a206_fbisc', 'a207_fbisc', 'a208_fbisc', 'a209_fbisc', 'a210_fbisc', 'a211_fbisc', 'a212_fbisc', 'a213_fbisc', 'a214_fbisc', 'a215_fbisc', 'a216_fbisc', 'a217_fbisc', 'a218_fbisc', 'a219_fbisc', 'a220_fbisc', 'a221_fbisc', 'a222_fbisc', 'a223_fbisc', 'a224_fbisc', 'a225_fbisc', 'a226_fbisc', 'a227_fbisc', 'a228_fbisc', 'a229_fbisc', 'a230_fbisc', 'a231_fbisc', 'a232_fbisc', 'a233_fbisc', 'a234_fbisc', 'a235_fbisc', 'a236_fbisc', 'a237_fbisc', 'a238_fbisc', 'a239_fbisc', 'a240_fbisc', 'a241_fbisc', 'a242_fbisc', 'a243_fbisc', 'a244_fbisc', 'a245_fbisc', 'a246_fbisc', 'a247_fbisc', 'a248_fbisc', 'a249_fbisc', 'a250_fbisc', 'a251_fbisc', 'a252_fbisc', 'a253_fbisc', 'a254_fbisc', 'a255_fbisc', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27', 'a28', 'a29', 'a30', 'a31', 'a32', 'a33', 'a34', 'a35', 'a36', 'a37', 'a38', 'a39', 'a40', 'a41', 'a42', 'a43', 'a44', 'a45', 'a46', 'a47', 'a48', 'a49', 'a50', 'a51', 'a52', 'a53', 'a54', 'a55', 'a56', 'a57', 'a58', 'a59', 'a60', 'a61', 'a62', 'a63', 'a64', 'a65', 'a66', 'a67', 'a68', 'a69', 'a70', 'a71', 'a72', 'a73', 'a74', 'a75', 'a76', 'a77', 'a78', 'a79', 'a80', 'a81', 'a82', 'a83', 'a84', 'a85', 'a86', 'a87', 'a88', 'a89', 'a90', 'a91', 'a92', 'a93', 'a94', 'a95', 'a96', 'a97', 'a98', 'a99', 'a100', 'a101', 'a102', 'a103', 'a104', 'a105', 'a106', 'a107', 'a108', 'a109', 'a110', 'a111', 'a112', 'a113', 'a114', 'a115', 'a116', 'a117', 'a118', 'a119', 'a120', 'a121', 'a122', 'a123', 'a124', 'a125', 'a126', 'a127', 'a128', 'a129', 'a130', 'a131', 'a132', 'a133', 'a134', 'a135', 'a136', 'a137', 'a138', 'a139', 'a140', 'a141', 'a142', 'a143', 'a144', 'a145', 'a146', 'a147', 'a148', 'a149', 'a150', 'a151', 'a152', 'a153', 'a154', 'a155', 'a156', 'a157', 'a158', 'a159', 'a160', 'a161', 'a162', 'a163', 'a164', 'a165', 'a166', 'a167', 'a168', 'a169', 'a170', 'a171', 'a172', 'a173', 'a174', 'a175', 'a176', 'a177', 'a178', 'a179', 'a180', 'a181', 'a182', 'a183', 'a184', 'a185', 'a186', 'a187', 'a188', 'a189', 'a190', 'a191', 'a192', 'a193', 'a194', 'a195', 'a196', 'a197', 'a198', 'a199', 'a200', 'a201', 'a202', 'a203', 'a204', 'a205', 'a206', 'a207', 'a208', 'a209', 'a210', 'a211', 'a212', 'a213', 'a214', 'a215', 'a216', 'a217', 'a218', 'a219', 'a220', 'a221', 'a222', 'a223', 'a224', 'a225', 'a226', 'a227', 'a228', 'a229', 'a230', 'a231', 'a232', 'a233', 'a234', 'a235', 'a236', 'a237', 'a238', 'a239', 'a240', 'a241', 'a242', 'a243', 'a244', 'a245', 'a246', 'a247', 'a248', 'a249', 'a250', 'a251', 'a252', 'a253', 'a254', 'a255', 'content_hash']
        return self._get_namedtuple()(**lol)

    def make_namedtuple_tf(self, *args, **kargs):
        return self._get_namedtuple()(*args, **kargs)

    @classmethod
    def from_arrow_schema(cls, parquet_dataset, omit_unsupported_fields=True):
        """
        Convert an apache arrow schema into a unischema object. This is useful for datasets of only scalars
        which need no special encoding/decoding. If there is an unsupported type in the arrow schema, it will
        throw an exception.
        When the warn_only parameter is turned to True, unsupported column types prints only warnings.

        We do not set codec field in the generated fields since all parquet fields are out-of-the-box supported
        by pyarrow and we do not need perform any custom decoding.

        :param arrow_schema: :class:`pyarrow.lib.Schema`
        :param omit_unsupported_fields: :class:`Boolean`
        :return: A :class:`Unischema` object.
        """
        meta = parquet_dataset.pieces[0].get_metadata()
        arrow_schema = meta.schema.to_arrow_schema()
        unischema_fields = []

        for partition in (parquet_dataset.partitions or []):
            if (pa.types.is_binary(partition.dictionary.type) and six.PY2) or \
                    (pa.types.is_string(partition.dictionary.type) and six.PY3):
                numpy_dtype = np.str_
            elif pa.types.is_int64(partition.dictionary.type):
                numpy_dtype = np.int64
            else:
                raise RuntimeError(('Expected partition type to be one of currently supported types: string or int64. '
                                    'Got {}').format(partition.dictionary.type))

            unischema_fields.append(UnischemaField(partition.name, numpy_dtype, (), None, False))

        for column_name in arrow_schema.names:
            arrow_field = arrow_schema.field(column_name)
            field_type = arrow_field.type
            field_shape = ()
            if isinstance(field_type, ListType):
                if isinstance(field_type.value_type, ListType) or isinstance(field_type.value_type, pyStructType):
                    warnings.warn('[ARROW-1644] Ignoring unsupported structure %r for field %r'
                                  % (field_type, column_name))
                    continue
                field_shape = (None,)
            try:
                np_type = _numpy_and_codec_from_arrow_type(field_type)
            except ValueError:
                if omit_unsupported_fields:
                    warnings.warn('Column %r has an unsupported field %r. Ignoring...'
                                  % (column_name, field_type))
                    continue
                else:
                    raise
            unischema_fields.append(UnischemaField(column_name, np_type, field_shape, None, arrow_field.nullable))
        return Unischema('inferred_schema', unischema_fields)

    def __getattr__(self, item) -> Any:
        return super().__getattribute__(item)


def dict_to_spark_row(unischema, row_dict):
    """Converts a single row into a spark Row object.

    Verifies that the data confirms with unischema definition types and encodes the data using the codec specified
    by the unischema.

    The parameters are keywords to allow use of functools.partial.

    :param unischema: an instance of Unischema object
    :param row_dict: a dictionary where the keys match name of fields in the unischema.
    :return: a single pyspark.Row object
    """

    # Lazy loading pyspark to avoid creating pyspark dependency on data reading code path
    # (currently works only with make_batch_reader)
    import pyspark

    assert isinstance(unischema, Unischema)
    # Add null fields. Be careful not to mutate the input dictionary - that would be an unexpected side effect
    copy_row_dict = copy.copy(row_dict)
    insert_explicit_nulls(unischema, copy_row_dict)

    if set(copy_row_dict.keys()) != set(unischema.fields.keys()):
        raise ValueError('Dictionary fields \n{}\n do not match schema fields \n{}'.format(
            '\n'.join(sorted(copy_row_dict.keys())), '\n'.join(unischema.fields.keys())))

    encoded_dict = {}
    for field_name, value in copy_row_dict.items():
        schema_field = unischema.fields[field_name]
        if value is None:
            if not schema_field.nullable:
                raise ValueError('Field {} is not "nullable", but got passes a None value')
        if schema_field.codec:
            encoded_dict[field_name] = schema_field.codec.encode(schema_field, value) if value is not None else None
        else:
            if isinstance(value, (np.generic,)):
                encoded_dict[field_name] = value.tolist()
            else:
                encoded_dict[field_name] = value

    field_list = list(unischema.fields.keys())
    # generate a value list which match the schema column order.
    value_list = [encoded_dict[name] for name in field_list]
    # create a row by value list
    row = pyspark.Row(*value_list)
    # set row fields
    row.__fields__ = field_list
    return row


def insert_explicit_nulls(unischema, row_dict):
    """If input dictionary has missing fields that are nullable, this function will add the missing keys with
    None value.

    If the fields that are missing are not nullable, a ``ValueError`` is raised.

    :param unischema: An instance of a unischema
    :param row_dict: dictionary that would be checked for missing nullable fields. The dictionary is modified inplace.
    :return: None
    """
    for field_name, value in unischema.fields.items():
        if field_name not in row_dict:
            if value.nullable:
                row_dict[field_name] = None
            else:
                raise ValueError('Field {} is not found in the row_dict, but is not nullable.'.format(field_name))


def _fullmatch(regex, string, flags=0):
    """Emulate python-3.4 re.fullmatch()."""
    if six.PY2:
        m = re.match(regex, string, flags=flags)
        if m and (m.span() == (0, len(string))):
            return m
    else:
        return re.fullmatch(regex, string, flags)


def match_unischema_fields(schema, field_regex):
    """Returns a list of :class:`~petastorm.unischema.UnischemaField` objects that match a regular expression.

    :param schema: An instance of a :class:`~petastorm.unischema.Unischema` object.
    :param field_regex: A list of regular expression patterns. A field is matched if the regular expression matches
      the entire field name.
    :return: A list of :class:`~petastorm.unischema.UnischemaField` instances matching at least one of the regular
      expression patterns given by ``field_regex``.
    """
    if field_regex:
        unischema_fields = set()
        legacy_unischema_fields = set()
        for pattern in field_regex:
            unischema_fields |= {field for field_name, field in schema.fields.items() if
                                 _fullmatch(pattern, field_name)}
            legacy_unischema_fields |= {field for field_name, field in schema.fields.items()
                                        if re.match(pattern, field_name)}
        if unischema_fields != legacy_unischema_fields:
            field_names = {f.name for f in unischema_fields}
            legacy_field_names = {f.name for f in legacy_unischema_fields}
            # Sorting list of diff_names so it's easier to unit-test the message
            diff_names = sorted(list((field_names | legacy_field_names) - (field_names & legacy_field_names)))
            warnings.warn('schema_fields behavior has changed. Now, regular expression pattern must match'
                          ' the entire field name. The change in the behavior affects '
                          'the following fields: {}'.format(', '.join(diff_names)))
        return list(unischema_fields)
    else:
        return []


def _numpy_and_codec_from_arrow_type(field_type):
    from pyarrow import types

    if types.is_int8(field_type):
        np_type = np.int8
    elif types.is_uint8(field_type):
        np_type = np.uint8
    elif types.is_int16(field_type):
        np_type = np.int16
    elif types.is_int32(field_type):
        np_type = np.int32
    elif types.is_int64(field_type):
        np_type = np.int64
    elif types.is_string(field_type):
        np_type = np.unicode_
    elif types.is_boolean(field_type):
        np_type = np.bool_
    elif types.is_float32(field_type):
        np_type = np.float32
    elif types.is_float64(field_type):
        np_type = np.float64
    elif types.is_decimal(field_type):
        np_type = Decimal
    elif types.is_binary(field_type):
        np_type = np.string_
    elif types.is_fixed_size_binary(field_type):
        np_type = np.string_
    elif types.is_date(field_type):
        np_type = np.datetime64
    elif types.is_timestamp(field_type):
        np_type = np.datetime64
    elif types.is_list(field_type):
        np_type = _numpy_and_codec_from_arrow_type(field_type.value_type)
    else:
        raise ValueError('Cannot auto-create unischema due to unsupported column type {}'.format(field_type))
    return np_type
