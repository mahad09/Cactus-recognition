	"�4��@"�4��@!"�4��@	5u褔O@5u褔O@!5u褔O@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$"�4��@�	�Y�>�?A �#GO�@Yp��/�D@*	�rh��L�@2|
EIterator::Model::MaxIntraOpParallelism::BatchV2::Shuffle::Zip[0]::Map@Lo.A@!�5;�tT@)�! 8�@@1���K�\T@:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::BatchV2����l�D@!��&l��X@)l"3��@1
Hw���0@:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::BatchV2::Shuffle::Zip@pB!CA@!$��o��T@)�և�F��?1�}tkv�?:Preprocessing2�
MIterator::Model::MaxIntraOpParallelism::BatchV2::Shuffle::Zip[1]::TensorSlice@vöE��?!<2���?)vöE��?1<2���?:Preprocessing2�
RIterator::Model::MaxIntraOpParallelism::BatchV2::Shuffle::Zip[0]::Map::TensorSlice@�\�].��?!�	�~���?)�\�].��?1�	�~���?:Preprocessing2o
8Iterator::Model::MaxIntraOpParallelism::BatchV2::Shuffle@�����SA@!�����T@)�I����?1,��z�.�?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism�H����D@!�A+��X@)���խ��?1�-�H���?:Preprocessing2F
Iterator::Model-��a�D@!      Y@)��)t^cw?1b�ė� �?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 5.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no96u褔O@I�x���W@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�	�Y�>�?�	�Y�>�?!�	�Y�>�?      ��!       "      ��!       *      ��!       2	 �#GO�@ �#GO�@! �#GO�@:      ��!       B      ��!       J	p��/�D@p��/�D@!p��/�D@R      ��!       Z	p��/�D@p��/�D@!p��/�D@b      ��!       JCPU_ONLYY6u褔O@b q�x���W@