Ĥ
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.1.02v2.1.0-rc2-17-ge5bf8de4108??
?
my_model/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_namemy_model/dense/kernel

)my_model/dense/kernel/Read/ReadVariableOpReadVariableOpmy_model/dense/kernel*
_output_shapes

:*
dtype0
~
my_model/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namemy_model/dense/bias
w
'my_model/dense/bias/Read/ReadVariableOpReadVariableOpmy_model/dense/bias*
_output_shapes
:*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
x
d
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
h

kernel
	bias

regularization_losses
trainable_variables
	variables
	keras_api
6
iter
	decay
learning_rate
momentum
 

0
	1

0
	1
?

layers
layer_regularization_losses
regularization_losses
trainable_variables
	variables
metrics
non_trainable_variables
 
NL
VARIABLE_VALUEmy_model/dense/kernel#d/kernel/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEmy_model/dense/bias!d/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
	1

0
	1
?

layers
layer_regularization_losses

regularization_losses
trainable_variables
	variables
metrics
non_trainable_variables
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
 
 
 
 
 
x
	total
	count

_fn_kwargs
regularization_losses
trainable_variables
 	variables
!	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
1
?

"layers
#layer_regularization_losses
regularization_losses
trainable_variables
 	variables
$metrics
%non_trainable_variables
 
 
 

0
1
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1my_model/dense/kernelmy_model/dense/bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8**
f%R#
!__inference_signature_wrapper_933
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)my_model/dense/kernel/Read/ReadVariableOp'my_model/dense/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*
Tin
2
	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

GPU 

CPU2J 8*&
f!R
__inference__traced_save_1028
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemy_model/dense/kernelmy_model/dense/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcount*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

GPU 

CPU2J 8*)
f$R"
 __inference__traced_restore_1064??
?'
?
 __inference__traced_restore_1064
file_prefix*
&assignvariableop_my_model_dense_kernel*
&assignvariableop_1_my_model_dense_bias
assignvariableop_2_sgd_iter 
assignvariableop_3_sgd_decay(
$assignvariableop_4_sgd_learning_rate#
assignvariableop_5_sgd_momentum
assignvariableop_6_total
assignvariableop_7_count

identity_9??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B#d/kernel/.ATTRIBUTES/VARIABLE_VALUEB!d/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*4
_output_shapes"
 ::::::::*
dtypes

2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp&assignvariableop_my_model_dense_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp&assignvariableop_1_my_model_dense_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0	*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_sgd_iterIdentity_2:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_sgd_decayIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp$assignvariableop_4_sgd_learning_rateIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_sgd_momentumIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_totalIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_countIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8?

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*5
_input_shapes$
": ::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
?
?
>__inference_dense_layer_call_and_return_conditional_losses_881

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?7my_model/dense/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
7my_model/dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource^MatMul/ReadVariableOp*
_output_shapes

:*
dtype029
7my_model/dense/kernel/Regularizer/Square/ReadVariableOp?
(my_model/dense/kernel/Regularizer/SquareSquare?my_model/dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2*
(my_model/dense/kernel/Regularizer/Square?
'my_model/dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'my_model/dense/kernel/Regularizer/Const?
%my_model/dense/kernel/Regularizer/SumSum,my_model/dense/kernel/Regularizer/Square:y:00my_model/dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%my_model/dense/kernel/Regularizer/Sum?
'my_model/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2)
'my_model/dense/kernel/Regularizer/mul/x?
%my_model/dense/kernel/Regularizer/mulMul0my_model/dense/kernel/Regularizer/mul/x:output:0.my_model/dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%my_model/dense/kernel/Regularizer/mul?
'my_model/dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'my_model/dense/kernel/Regularizer/add/x?
%my_model/dense/kernel/Regularizer/addAddV20my_model/dense/kernel/Regularizer/add/x:output:0)my_model/dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2'
%my_model/dense/kernel/Regularizer/add?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp8^my_model/dense/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2r
7my_model/dense/kernel/Regularizer/Square/ReadVariableOp7my_model/dense/kernel/Regularizer/Square/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
>__inference_dense_layer_call_and_return_conditional_losses_960

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?7my_model/dense/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
7my_model/dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource^MatMul/ReadVariableOp*
_output_shapes

:*
dtype029
7my_model/dense/kernel/Regularizer/Square/ReadVariableOp?
(my_model/dense/kernel/Regularizer/SquareSquare?my_model/dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2*
(my_model/dense/kernel/Regularizer/Square?
'my_model/dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'my_model/dense/kernel/Regularizer/Const?
%my_model/dense/kernel/Regularizer/SumSum,my_model/dense/kernel/Regularizer/Square:y:00my_model/dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%my_model/dense/kernel/Regularizer/Sum?
'my_model/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2)
'my_model/dense/kernel/Regularizer/mul/x?
%my_model/dense/kernel/Regularizer/mulMul0my_model/dense/kernel/Regularizer/mul/x:output:0.my_model/dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%my_model/dense/kernel/Regularizer/mul?
'my_model/dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'my_model/dense/kernel/Regularizer/add/x?
%my_model/dense/kernel/Regularizer/addAddV20my_model/dense/kernel/Regularizer/add/x:output:0)my_model/dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2'
%my_model/dense/kernel/Regularizer/add?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp8^my_model/dense/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2r
7my_model/dense/kernel/Regularizer/Square/ReadVariableOp7my_model/dense/kernel/Regularizer/Square/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
#__inference_dense_layer_call_fn_967

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_8812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_980D
@my_model_dense_kernel_regularizer_square_readvariableop_resource
identity??7my_model/dense/kernel/Regularizer/Square/ReadVariableOp?
7my_model/dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp@my_model_dense_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype029
7my_model/dense/kernel/Regularizer/Square/ReadVariableOp?
(my_model/dense/kernel/Regularizer/SquareSquare?my_model/dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2*
(my_model/dense/kernel/Regularizer/Square?
'my_model/dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'my_model/dense/kernel/Regularizer/Const?
%my_model/dense/kernel/Regularizer/SumSum,my_model/dense/kernel/Regularizer/Square:y:00my_model/dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%my_model/dense/kernel/Regularizer/Sum?
'my_model/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2)
'my_model/dense/kernel/Regularizer/mul/x?
%my_model/dense/kernel/Regularizer/mulMul0my_model/dense/kernel/Regularizer/mul/x:output:0.my_model/dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%my_model/dense/kernel/Regularizer/mul?
'my_model/dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'my_model/dense/kernel/Regularizer/add/x?
%my_model/dense/kernel/Regularizer/addAddV20my_model/dense/kernel/Regularizer/add/x:output:0)my_model/dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2'
%my_model/dense/kernel/Regularizer/add?
IdentityIdentity)my_model/dense/kernel/Regularizer/add:z:08^my_model/dense/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2r
7my_model/dense/kernel/Regularizer/Square/ReadVariableOp7my_model/dense/kernel/Regularizer/Square/ReadVariableOp
?
?
__inference__traced_save_1028
file_prefix4
0savev2_my_model_dense_kernel_read_readvariableop2
.savev2_my_model_dense_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_ea3a5bdeb3554649948ba38fa88845c5/part2
StringJoin/inputs_1?

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B#d/kernel/.ATTRIBUTES/VARIABLE_VALUEB!d/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_my_model_dense_kernel_read_readvariableop.savev2_my_model_dense_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes

2	2
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*3
_input_shapes"
 : ::: : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
?
?
__inference__wrapped_model_858
input_11
-my_model_dense_matmul_readvariableop_resource2
.my_model_dense_biasadd_readvariableop_resource
identity??%my_model/dense/BiasAdd/ReadVariableOp?$my_model/dense/MatMul/ReadVariableOp?
$my_model/dense/MatMul/ReadVariableOpReadVariableOp-my_model_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02&
$my_model/dense/MatMul/ReadVariableOp?
my_model/dense/MatMulMatMulinput_1,my_model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
my_model/dense/MatMul?
%my_model/dense/BiasAdd/ReadVariableOpReadVariableOp.my_model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%my_model/dense/BiasAdd/ReadVariableOp?
my_model/dense/BiasAddBiasAddmy_model/dense/MatMul:product:0-my_model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
my_model/dense/BiasAdd?
my_model/dense/SoftmaxSoftmaxmy_model/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
my_model/dense/Softmax?
IdentityIdentity my_model/dense/Softmax:softmax:0&^my_model/dense/BiasAdd/ReadVariableOp%^my_model/dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::2N
%my_model/dense/BiasAdd/ReadVariableOp%my_model/dense/BiasAdd/ReadVariableOp2L
$my_model/dense/MatMul/ReadVariableOp$my_model/dense/MatMul/ReadVariableOp:' #
!
_user_specified_name	input_1
?
?
A__inference_my_model_layer_call_and_return_conditional_losses_903
input_1(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identity??dense/StatefulPartitionedCall?7my_model/dense/kernel/Regularizer/Square/ReadVariableOp?
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_8812
dense/StatefulPartitionedCall?
dense/IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
dense/Identity?
7my_model/dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_statefulpartitionedcall_args_1^dense/StatefulPartitionedCall*
_output_shapes

:*
dtype029
7my_model/dense/kernel/Regularizer/Square/ReadVariableOp?
(my_model/dense/kernel/Regularizer/SquareSquare?my_model/dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2*
(my_model/dense/kernel/Regularizer/Square?
'my_model/dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'my_model/dense/kernel/Regularizer/Const?
%my_model/dense/kernel/Regularizer/SumSum,my_model/dense/kernel/Regularizer/Square:y:00my_model/dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%my_model/dense/kernel/Regularizer/Sum?
'my_model/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2)
'my_model/dense/kernel/Regularizer/mul/x?
%my_model/dense/kernel/Regularizer/mulMul0my_model/dense/kernel/Regularizer/mul/x:output:0.my_model/dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%my_model/dense/kernel/Regularizer/mul?
'my_model/dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'my_model/dense/kernel/Regularizer/add/x?
%my_model/dense/kernel/Regularizer/addAddV20my_model/dense/kernel/Regularizer/add/x:output:0)my_model/dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2'
%my_model/dense/kernel/Regularizer/add?
IdentityIdentitydense/Identity:output:0^dense/StatefulPartitionedCall8^my_model/dense/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2r
7my_model/dense/kernel/Regularizer/Square/ReadVariableOp7my_model/dense/kernel/Regularizer/Square/ReadVariableOp:' #
!
_user_specified_name	input_1
?
?
!__inference_signature_wrapper_933
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*'
f"R 
__inference__wrapped_model_8582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
?
?
&__inference_my_model_layer_call_fn_911
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_my_model_layer_call_and_return_conditional_losses_9032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?8
?
d
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
*&&call_and_return_all_conditional_losses
'_default_save_signature
(__call__"?
_tf_keras_model?{"class_name": "my_model", "name": "my_model", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "is_graph_network": false, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "my_model"}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": false}}, "metrics": ["sparse_categorical_accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.009999999776482582, "decay": 0.0, "momentum": 0.0, "nesterov": false}}}}
?

kernel
	bias

regularization_losses
trainable_variables
	variables
	keras_api
*)&call_and_return_all_conditional_losses
*__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}}
I
iter
	decay
learning_rate
momentum"
	optimizer
'
+0"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
?

layers
layer_regularization_losses
regularization_losses
trainable_variables
	variables
metrics
non_trainable_variables
(__call__
'_default_save_signature
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
,
,serving_default"
signature_map
':%2my_model/dense/kernel
!:2my_model/dense/bias
'
+0"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
?

layers
layer_regularization_losses

regularization_losses
trainable_variables
	variables
metrics
non_trainable_variables
*__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
+0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	total
	count

_fn_kwargs
regularization_losses
trainable_variables
 	variables
!	keras_api
*-&call_and_return_all_conditional_losses
.__call__"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "sparse_categorical_accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sparse_categorical_accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

"layers
#layer_regularization_losses
regularization_losses
trainable_variables
 	variables
$metrics
%non_trainable_variables
.__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?2?
A__inference_my_model_layer_call_and_return_conditional_losses_903?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????
?2?
__inference__wrapped_model_858?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????
?2?
&__inference_my_model_layer_call_fn_911?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????
?2?
>__inference_dense_layer_call_and_return_conditional_losses_960?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
#__inference_dense_layer_call_fn_967?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_980?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
0B.
!__inference_signature_wrapper_933input_1
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 ?
__inference__wrapped_model_858k	0?-
&?#
!?
input_1?????????
? "3?0
.
output_1"?
output_1??????????
>__inference_dense_layer_call_and_return_conditional_losses_960\	/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? v
#__inference_dense_layer_call_fn_967O	/?,
%?"
 ?
inputs?????????
? "??????????8
__inference_loss_fn_0_980?

? 
? "? ?
A__inference_my_model_layer_call_and_return_conditional_losses_903]	0?-
&?#
!?
input_1?????????
? "%?"
?
0?????????
? z
&__inference_my_model_layer_call_fn_911P	0?-
&?#
!?
input_1?????????
? "???????????
!__inference_signature_wrapper_933v	;?8
? 
1?.
,
input_1!?
input_1?????????"3?0
.
output_1"?
output_1?????????