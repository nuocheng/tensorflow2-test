þý
«ý
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
dtypetype
¾
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
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.1.02v2.1.0-rc2-17-ge5bf8de4108Ù

minist_model/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
**
shared_nameminist_model/dense/kernel

-minist_model/dense/kernel/Read/ReadVariableOpReadVariableOpminist_model/dense/kernel* 
_output_shapes
:
*
dtype0

minist_model/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameminist_model/dense/bias

+minist_model/dense/bias/Read/ReadVariableOpReadVariableOpminist_model/dense/bias*
_output_shapes	
:*
dtype0

minist_model/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*,
shared_nameminist_model/dense_1/kernel

/minist_model/dense_1/kernel/Read/ReadVariableOpReadVariableOpminist_model/dense_1/kernel*
_output_shapes
:	
*
dtype0

minist_model/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
**
shared_nameminist_model/dense_1/bias

-minist_model/dense_1/bias/Read/ReadVariableOpReadVariableOpminist_model/dense_1/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
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

 Adam/minist_model/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*1
shared_name" Adam/minist_model/dense/kernel/m

4Adam/minist_model/dense/kernel/m/Read/ReadVariableOpReadVariableOp Adam/minist_model/dense/kernel/m* 
_output_shapes
:
*
dtype0

Adam/minist_model/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/minist_model/dense/bias/m

2Adam/minist_model/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/minist_model/dense/bias/m*
_output_shapes	
:*
dtype0
¡
"Adam/minist_model/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*3
shared_name$"Adam/minist_model/dense_1/kernel/m

6Adam/minist_model/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/minist_model/dense_1/kernel/m*
_output_shapes
:	
*
dtype0

 Adam/minist_model/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*1
shared_name" Adam/minist_model/dense_1/bias/m

4Adam/minist_model/dense_1/bias/m/Read/ReadVariableOpReadVariableOp Adam/minist_model/dense_1/bias/m*
_output_shapes
:
*
dtype0

 Adam/minist_model/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*1
shared_name" Adam/minist_model/dense/kernel/v

4Adam/minist_model/dense/kernel/v/Read/ReadVariableOpReadVariableOp Adam/minist_model/dense/kernel/v* 
_output_shapes
:
*
dtype0

Adam/minist_model/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/minist_model/dense/bias/v

2Adam/minist_model/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/minist_model/dense/bias/v*
_output_shapes	
:*
dtype0
¡
"Adam/minist_model/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*3
shared_name$"Adam/minist_model/dense_1/kernel/v

6Adam/minist_model/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/minist_model/dense_1/kernel/v*
_output_shapes
:	
*
dtype0

 Adam/minist_model/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*1
shared_name" Adam/minist_model/dense_1/bias/v

4Adam/minist_model/dense_1/bias/v/Read/ReadVariableOpReadVariableOp Adam/minist_model/dense_1/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
Ì
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueýBú Bó

f
d1
d2
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api
	
signatures
R

trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api

iter

beta_1

beta_2
	decay
learning_ratem;m<m=m>v?v@vAvB

0
1
2
3

0
1
2
3
 

non_trainable_variables
 layer_regularization_losses

!layers
trainable_variables
	variables
"metrics
regularization_losses
 
 
 
 

#non_trainable_variables
$layer_regularization_losses

%layers

trainable_variables
	variables
&metrics
regularization_losses
SQ
VARIABLE_VALUEminist_model/dense/kernel$d1/kernel/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEminist_model/dense/bias"d1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 

'non_trainable_variables
(layer_regularization_losses

)layers
trainable_variables
	variables
*metrics
regularization_losses
US
VARIABLE_VALUEminist_model/dense_1/kernel$d2/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEminist_model/dense_1/bias"d2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 

+non_trainable_variables
,layer_regularization_losses

-layers
trainable_variables
	variables
.metrics
regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1
2

/0
 
 
 
 
 
 
 
 
 
 
 
 
x
	0total
	1count
2
_fn_kwargs
3trainable_variables
4	variables
5regularization_losses
6	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

00
11
 

7non_trainable_variables
8layer_regularization_losses

9layers
3trainable_variables
4	variables
:metrics
5regularization_losses

00
11
 
 
 
vt
VARIABLE_VALUE Adam/minist_model/dense/kernel/m@d1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/minist_model/dense/bias/m>d1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE"Adam/minist_model/dense_1/kernel/m@d2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE Adam/minist_model/dense_1/bias/m>d2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE Adam/minist_model/dense/kernel/v@d1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/minist_model/dense/bias/v>d1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE"Adam/minist_model/dense_1/kernel/v@d2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE Adam/minist_model/dense_1/bias/v>d2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_1Placeholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1minist_model/dense/kernelminist_model/dense/biasminist_model/dense_1/kernelminist_model/dense_1/bias*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
config_proto

GPU 

CPU2J 8*-
f(R&
$__inference_signature_wrapper_159379
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ã
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-minist_model/dense/kernel/Read/ReadVariableOp+minist_model/dense/bias/Read/ReadVariableOp/minist_model/dense_1/kernel/Read/ReadVariableOp-minist_model/dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp4Adam/minist_model/dense/kernel/m/Read/ReadVariableOp2Adam/minist_model/dense/bias/m/Read/ReadVariableOp6Adam/minist_model/dense_1/kernel/m/Read/ReadVariableOp4Adam/minist_model/dense_1/bias/m/Read/ReadVariableOp4Adam/minist_model/dense/kernel/v/Read/ReadVariableOp2Adam/minist_model/dense/bias/v/Read/ReadVariableOp6Adam/minist_model/dense_1/kernel/v/Read/ReadVariableOp4Adam/minist_model/dense_1/bias/v/Read/ReadVariableOpConst* 
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

GPU 

CPU2J 8*(
f#R!
__inference__traced_save_159536
â
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameminist_model/dense/kernelminist_model/dense/biasminist_model/dense_1/kernelminist_model/dense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount Adam/minist_model/dense/kernel/mAdam/minist_model/dense/bias/m"Adam/minist_model/dense_1/kernel/m Adam/minist_model/dense_1/bias/m Adam/minist_model/dense/kernel/vAdam/minist_model/dense/bias/v"Adam/minist_model/dense_1/kernel/v Adam/minist_model/dense_1/bias/v*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

GPU 

CPU2J 8*+
f&R$
"__inference__traced_restore_159605 Á
¿

A__inference_dense_layer_call_and_return_conditional_losses_159297

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢;minist_model/dense/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluó
;minist_model/dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource^MatMul/ReadVariableOp* 
_output_shapes
:
*
dtype02=
;minist_model/dense/kernel/Regularizer/Square/ReadVariableOpÖ
,minist_model/dense/kernel/Regularizer/SquareSquareCminist_model/dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2.
,minist_model/dense/kernel/Regularizer/Square«
+minist_model/dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+minist_model/dense/kernel/Regularizer/Constæ
)minist_model/dense/kernel/Regularizer/SumSum0minist_model/dense/kernel/Regularizer/Square:y:04minist_model/dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)minist_model/dense/kernel/Regularizer/Sum
+minist_model/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+minist_model/dense/kernel/Regularizer/mul/xè
)minist_model/dense/kernel/Regularizer/mulMul4minist_model/dense/kernel/Regularizer/mul/x:output:02minist_model/dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)minist_model/dense/kernel/Regularizer/mul
+minist_model/dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+minist_model/dense/kernel/Regularizer/add/xå
)minist_model/dense/kernel/Regularizer/addAddV24minist_model/dense/kernel/Regularizer/add/x:output:0-minist_model/dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)minist_model/dense/kernel/Regularizer/addÖ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp<^minist_model/dense/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2z
;minist_model/dense/kernel/Regularizer/Square/ReadVariableOp;minist_model/dense/kernel/Regularizer/Square/ReadVariableOp:& "
 
_user_specified_nameinputs
ñ
©
(__inference_dense_1_layer_call_fn_159442

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1593212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
êR

"__inference__traced_restore_159605
file_prefix.
*assignvariableop_minist_model_dense_kernel.
*assignvariableop_1_minist_model_dense_bias2
.assignvariableop_2_minist_model_dense_1_kernel0
,assignvariableop_3_minist_model_dense_1_bias 
assignvariableop_4_adam_iter"
assignvariableop_5_adam_beta_1"
assignvariableop_6_adam_beta_2!
assignvariableop_7_adam_decay)
%assignvariableop_8_adam_learning_rate
assignvariableop_9_total
assignvariableop_10_count8
4assignvariableop_11_adam_minist_model_dense_kernel_m6
2assignvariableop_12_adam_minist_model_dense_bias_m:
6assignvariableop_13_adam_minist_model_dense_1_kernel_m8
4assignvariableop_14_adam_minist_model_dense_1_bias_m8
4assignvariableop_15_adam_minist_model_dense_kernel_v6
2assignvariableop_16_adam_minist_model_dense_bias_v:
6assignvariableop_17_adam_minist_model_dense_1_kernel_v8
4assignvariableop_18_adam_minist_model_dense_1_bias_v
identity_20¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¢	RestoreV2¢RestoreV2_1ö
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueøBõB$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB@d1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>d1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@d2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>d2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@d1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>d1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@d2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>d2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names´
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp*assignvariableop_minist_model_dense_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1 
AssignVariableOp_1AssignVariableOp*assignvariableop_1_minist_model_dense_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2¤
AssignVariableOp_2AssignVariableOp.assignvariableop_2_minist_model_dense_1_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3¢
AssignVariableOp_3AssignVariableOp,assignvariableop_3_minist_model_dense_1_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0	*
_output_shapes
:2

Identity_4
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11­
AssignVariableOp_11AssignVariableOp4assignvariableop_11_adam_minist_model_dense_kernel_mIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12«
AssignVariableOp_12AssignVariableOp2assignvariableop_12_adam_minist_model_dense_bias_mIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13¯
AssignVariableOp_13AssignVariableOp6assignvariableop_13_adam_minist_model_dense_1_kernel_mIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14­
AssignVariableOp_14AssignVariableOp4assignvariableop_14_adam_minist_model_dense_1_bias_mIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15­
AssignVariableOp_15AssignVariableOp4assignvariableop_15_adam_minist_model_dense_kernel_vIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16«
AssignVariableOp_16AssignVariableOp2assignvariableop_16_adam_minist_model_dense_bias_vIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17¯
AssignVariableOp_17AssignVariableOp6assignvariableop_17_adam_minist_model_dense_1_kernel_vIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18­
AssignVariableOp_18AssignVariableOp4assignvariableop_18_adam_minist_model_dense_1_bias_vIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18¨
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesÄ
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
NoOp
Identity_19Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_19
Identity_20IdentityIdentity_19:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_20"#
identity_20Identity_20:output:0*a
_input_shapesP
N: :::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
×
D
(__inference_flatten_layer_call_fn_159390

inputs
identity¬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1592692
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:& "
 
_user_specified_nameinputs
Ü
Õ
!__inference__wrapped_model_159259
input_15
1minist_model_dense_matmul_readvariableop_resource6
2minist_model_dense_biasadd_readvariableop_resource7
3minist_model_dense_1_matmul_readvariableop_resource8
4minist_model_dense_1_biasadd_readvariableop_resource
identity¢)minist_model/dense/BiasAdd/ReadVariableOp¢(minist_model/dense/MatMul/ReadVariableOp¢+minist_model/dense_1/BiasAdd/ReadVariableOp¢*minist_model/dense_1/MatMul/ReadVariableOp
minist_model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
minist_model/flatten/Const¨
minist_model/flatten/ReshapeReshapeinput_1#minist_model/flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
minist_model/flatten/ReshapeÈ
(minist_model/dense/MatMul/ReadVariableOpReadVariableOp1minist_model_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02*
(minist_model/dense/MatMul/ReadVariableOpÌ
minist_model/dense/MatMulMatMul%minist_model/flatten/Reshape:output:00minist_model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
minist_model/dense/MatMulÆ
)minist_model/dense/BiasAdd/ReadVariableOpReadVariableOp2minist_model_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)minist_model/dense/BiasAdd/ReadVariableOpÎ
minist_model/dense/BiasAddBiasAdd#minist_model/dense/MatMul:product:01minist_model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
minist_model/dense/BiasAdd
minist_model/dense/ReluRelu#minist_model/dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
minist_model/dense/ReluÍ
*minist_model/dense_1/MatMul/ReadVariableOpReadVariableOp3minist_model_dense_1_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02,
*minist_model/dense_1/MatMul/ReadVariableOpÑ
minist_model/dense_1/MatMulMatMul%minist_model/dense/Relu:activations:02minist_model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
minist_model/dense_1/MatMulË
+minist_model/dense_1/BiasAdd/ReadVariableOpReadVariableOp4minist_model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02-
+minist_model/dense_1/BiasAdd/ReadVariableOpÕ
minist_model/dense_1/BiasAddBiasAdd%minist_model/dense_1/MatMul:product:03minist_model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
minist_model/dense_1/BiasAdd 
minist_model/dense_1/SoftmaxSoftmax%minist_model/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
minist_model/dense_1/Softmax¬
IdentityIdentity&minist_model/dense_1/Softmax:softmax:0*^minist_model/dense/BiasAdd/ReadVariableOp)^minist_model/dense/MatMul/ReadVariableOp,^minist_model/dense_1/BiasAdd/ReadVariableOp+^minist_model/dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::2V
)minist_model/dense/BiasAdd/ReadVariableOp)minist_model/dense/BiasAdd/ReadVariableOp2T
(minist_model/dense/MatMul/ReadVariableOp(minist_model/dense/MatMul/ReadVariableOp2Z
+minist_model/dense_1/BiasAdd/ReadVariableOp+minist_model/dense_1/BiasAdd/ReadVariableOp2X
*minist_model/dense_1/MatMul/ReadVariableOp*minist_model/dense_1/MatMul/ReadVariableOp:' #
!
_user_specified_name	input_1
ï
§
&__inference_dense_layer_call_fn_159424

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1592972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Î	
Ü
C__inference_dense_1_layer_call_and_return_conditional_losses_159321

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs

_
C__inference_flatten_layer_call_and_return_conditional_losses_159385

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:& "
 
_user_specified_nameinputs
0
	
__inference__traced_save_159536
file_prefix8
4savev2_minist_model_dense_kernel_read_readvariableop6
2savev2_minist_model_dense_bias_read_readvariableop:
6savev2_minist_model_dense_1_kernel_read_readvariableop8
4savev2_minist_model_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop?
;savev2_adam_minist_model_dense_kernel_m_read_readvariableop=
9savev2_adam_minist_model_dense_bias_m_read_readvariableopA
=savev2_adam_minist_model_dense_1_kernel_m_read_readvariableop?
;savev2_adam_minist_model_dense_1_bias_m_read_readvariableop?
;savev2_adam_minist_model_dense_kernel_v_read_readvariableop=
9savev2_adam_minist_model_dense_bias_v_read_readvariableopA
=savev2_adam_minist_model_dense_1_kernel_v_read_readvariableop?
;savev2_adam_minist_model_dense_1_bias_v_read_readvariableop
savev2_1_const

identity_1¢MergeV2Checkpoints¢SaveV2¢SaveV2_1¥
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_e9e9ecdb6228494eb8c8d798b387f646/part2
StringJoin/inputs_1

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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameð
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueøBõB$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB@d1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>d1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@d2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>d2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@d1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>d1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@d2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>d2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names®
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_minist_model_dense_kernel_read_readvariableop2savev2_minist_model_dense_bias_read_readvariableop6savev2_minist_model_dense_1_kernel_read_readvariableop4savev2_minist_model_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop;savev2_adam_minist_model_dense_kernel_m_read_readvariableop9savev2_adam_minist_model_dense_bias_m_read_readvariableop=savev2_adam_minist_model_dense_1_kernel_m_read_readvariableop;savev2_adam_minist_model_dense_1_bias_m_read_readvariableop;savev2_adam_minist_model_dense_kernel_v_read_readvariableop9savev2_adam_minist_model_dense_bias_v_read_readvariableop=savev2_adam_minist_model_dense_1_kernel_v_read_readvariableop;savev2_adam_minist_model_dense_1_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *!
dtypes
2	2
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard¬
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1¢
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesÏ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1ã
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¬
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
~: :
::	
:
: : : : : : : :
::	
:
:
::	
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
¤"

H__inference_minist_model_layer_call_and_return_conditional_losses_159343
input_1(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢;minist_model/dense/kernel/Regularizer/Square/ReadVariableOp½
flatten/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1592692
flatten/PartitionedCall
flatten/IdentityIdentity flatten/PartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten/Identity¯
dense/StatefulPartitionedCallStatefulPartitionedCallflatten/Identity:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1592972
dense/StatefulPartitionedCall§
dense/IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/Identity¶
dense_1/StatefulPartitionedCallStatefulPartitionedCalldense/Identity:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1593212!
dense_1/StatefulPartitionedCall®
dense_1/IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_1/Identity
;minist_model/dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_statefulpartitionedcall_args_1^dense/StatefulPartitionedCall* 
_output_shapes
:
*
dtype02=
;minist_model/dense/kernel/Regularizer/Square/ReadVariableOpÖ
,minist_model/dense/kernel/Regularizer/SquareSquareCminist_model/dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2.
,minist_model/dense/kernel/Regularizer/Square«
+minist_model/dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+minist_model/dense/kernel/Regularizer/Constæ
)minist_model/dense/kernel/Regularizer/SumSum0minist_model/dense/kernel/Regularizer/Square:y:04minist_model/dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)minist_model/dense/kernel/Regularizer/Sum
+minist_model/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+minist_model/dense/kernel/Regularizer/mul/xè
)minist_model/dense/kernel/Regularizer/mulMul4minist_model/dense/kernel/Regularizer/mul/x:output:02minist_model/dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)minist_model/dense/kernel/Regularizer/mul
+minist_model/dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+minist_model/dense/kernel/Regularizer/add/xå
)minist_model/dense/kernel/Regularizer/addAddV24minist_model/dense/kernel/Regularizer/add/x:output:0-minist_model/dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)minist_model/dense/kernel/Regularizer/addí
IdentityIdentitydense_1/Identity:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall<^minist_model/dense/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2z
;minist_model/dense/kernel/Regularizer/Square/ReadVariableOp;minist_model/dense/kernel/Regularizer/Square/ReadVariableOp:' #
!
_user_specified_name	input_1

÷
-__inference_minist_model_layer_call_fn_159353
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity¢StatefulPartitionedCallÍ
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_minist_model_layer_call_and_return_conditional_losses_1593432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
Î	
Ü
C__inference_dense_1_layer_call_and_return_conditional_losses_159435

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
¿

A__inference_dense_layer_call_and_return_conditional_losses_159417

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢;minist_model/dense/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluó
;minist_model/dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource^MatMul/ReadVariableOp* 
_output_shapes
:
*
dtype02=
;minist_model/dense/kernel/Regularizer/Square/ReadVariableOpÖ
,minist_model/dense/kernel/Regularizer/SquareSquareCminist_model/dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2.
,minist_model/dense/kernel/Regularizer/Square«
+minist_model/dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+minist_model/dense/kernel/Regularizer/Constæ
)minist_model/dense/kernel/Regularizer/SumSum0minist_model/dense/kernel/Regularizer/Square:y:04minist_model/dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)minist_model/dense/kernel/Regularizer/Sum
+minist_model/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+minist_model/dense/kernel/Regularizer/mul/xè
)minist_model/dense/kernel/Regularizer/mulMul4minist_model/dense/kernel/Regularizer/mul/x:output:02minist_model/dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)minist_model/dense/kernel/Regularizer/mul
+minist_model/dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+minist_model/dense/kernel/Regularizer/add/xå
)minist_model/dense/kernel/Regularizer/addAddV24minist_model/dense/kernel/Regularizer/add/x:output:0-minist_model/dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)minist_model/dense/kernel/Regularizer/addÖ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp<^minist_model/dense/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2z
;minist_model/dense/kernel/Regularizer/Square/ReadVariableOp;minist_model/dense/kernel/Regularizer/Square/ReadVariableOp:& "
 
_user_specified_nameinputs
ã
î
$__inference_signature_wrapper_159379
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
config_proto

GPU 

CPU2J 8**
f%R#
!__inference__wrapped_model_1592592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1

_
C__inference_flatten_layer_call_and_return_conditional_losses_159269

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:& "
 
_user_specified_nameinputs
ª
·
__inference_loss_fn_0_159455H
Dminist_model_dense_kernel_regularizer_square_readvariableop_resource
identity¢;minist_model/dense/kernel/Regularizer/Square/ReadVariableOp
;minist_model/dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpDminist_model_dense_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
*
dtype02=
;minist_model/dense/kernel/Regularizer/Square/ReadVariableOpÖ
,minist_model/dense/kernel/Regularizer/SquareSquareCminist_model/dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2.
,minist_model/dense/kernel/Regularizer/Square«
+minist_model/dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+minist_model/dense/kernel/Regularizer/Constæ
)minist_model/dense/kernel/Regularizer/SumSum0minist_model/dense/kernel/Regularizer/Square:y:04minist_model/dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)minist_model/dense/kernel/Regularizer/Sum
+minist_model/dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+minist_model/dense/kernel/Regularizer/mul/xè
)minist_model/dense/kernel/Regularizer/mulMul4minist_model/dense/kernel/Regularizer/mul/x:output:02minist_model/dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)minist_model/dense/kernel/Regularizer/mul
+minist_model/dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+minist_model/dense/kernel/Regularizer/add/xå
)minist_model/dense/kernel/Regularizer/addAddV24minist_model/dense/kernel/Regularizer/add/x:output:0-minist_model/dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)minist_model/dense/kernel/Regularizer/add®
IdentityIdentity-minist_model/dense/kernel/Regularizer/add:z:0<^minist_model/dense/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2z
;minist_model/dense/kernel/Regularizer/Square/ReadVariableOp;minist_model/dense/kernel/Regularizer/Square/ReadVariableOp"¯L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¯
serving_default
?
input_14
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:^

f
d1
d2
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api
	
signatures
*C&call_and_return_all_conditional_losses
D__call__
E_default_save_signature"¸
_tf_keras_model{"class_name": "minist_model", "name": "minist_model", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "is_graph_network": false, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "minist_model"}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": false}}, "metrics": ["sparse_categorical_accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
¬

trainable_variables
	variables
regularization_losses
	keras_api
*F&call_and_return_all_conditional_losses
G__call__"
_tf_keras_layer{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
´

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*H&call_and_return_all_conditional_losses
I__call__"
_tf_keras_layerõ{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}}
õ

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*J&call_and_return_all_conditional_losses
K__call__"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}

iter

beta_1

beta_2
	decay
learning_ratem;m<m=m>v?v@vAvB"
	optimizer
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
'
L0"
trackable_list_wrapper
·
non_trainable_variables
 layer_regularization_losses

!layers
trainable_variables
	variables
"metrics
regularization_losses
D__call__
E_default_save_signature
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
,
Mserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

#non_trainable_variables
$layer_regularization_losses

%layers

trainable_variables
	variables
&metrics
regularization_losses
G__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
-:+
2minist_model/dense/kernel
&:$2minist_model/dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
L0"
trackable_list_wrapper

'non_trainable_variables
(layer_regularization_losses

)layers
trainable_variables
	variables
*metrics
regularization_losses
I__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
.:,	
2minist_model/dense_1/kernel
':%
2minist_model/dense_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper

+non_trainable_variables
,layer_regularization_losses

-layers
trainable_variables
	variables
.metrics
regularization_losses
K__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
'
/0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
L0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
À
	0total
	1count
2
_fn_kwargs
3trainable_variables
4	variables
5regularization_losses
6	keras_api
*N&call_and_return_all_conditional_losses
O__call__"
_tf_keras_layerñ{"class_name": "MeanMetricWrapper", "name": "sparse_categorical_accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sparse_categorical_accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper

7non_trainable_variables
8layer_regularization_losses

9layers
3trainable_variables
4	variables
:metrics
5regularization_losses
O__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
2:0
2 Adam/minist_model/dense/kernel/m
+:)2Adam/minist_model/dense/bias/m
3:1	
2"Adam/minist_model/dense_1/kernel/m
,:*
2 Adam/minist_model/dense_1/bias/m
2:0
2 Adam/minist_model/dense/kernel/v
+:)2Adam/minist_model/dense/bias/v
3:1	
2"Adam/minist_model/dense_1/kernel/v
,:*
2 Adam/minist_model/dense_1/bias/v
2
H__inference_minist_model_layer_call_and_return_conditional_losses_159343É
²
FullArgSpec
args
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª **¢'
%"
input_1ÿÿÿÿÿÿÿÿÿ
þ2û
-__inference_minist_model_layer_call_fn_159353É
²
FullArgSpec
args
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª **¢'
%"
input_1ÿÿÿÿÿÿÿÿÿ
ã2à
!__inference__wrapped_model_159259º
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª **¢'
%"
input_1ÿÿÿÿÿÿÿÿÿ
í2ê
C__inference_flatten_layer_call_and_return_conditional_losses_159385¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_flatten_layer_call_fn_159390¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_dense_layer_call_and_return_conditional_losses_159417¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
&__inference_dense_layer_call_fn_159424¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_1_layer_call_and_return_conditional_losses_159435¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_1_layer_call_fn_159442¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
³2°
__inference_loss_fn_0_159455
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
3B1
$__inference_signature_wrapper_159379input_1
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
!__inference__wrapped_model_159259q4¢1
*¢'
%"
input_1ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
¤
C__inference_dense_1_layer_call_and_return_conditional_losses_159435]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 |
(__inference_dense_1_layer_call_fn_159442P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
£
A__inference_dense_layer_call_and_return_conditional_losses_159417^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 {
&__inference_dense_layer_call_fn_159424Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
C__inference_flatten_layer_call_and_return_conditional_losses_159385]3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 |
(__inference_flatten_layer_call_fn_159390P3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ;
__inference_loss_fn_0_159455¢

¢ 
ª " ¯
H__inference_minist_model_layer_call_and_return_conditional_losses_159343c4¢1
*¢'
%"
input_1ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
-__inference_minist_model_layer_call_fn_159353V4¢1
*¢'
%"
input_1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
¤
$__inference_signature_wrapper_159379|?¢<
¢ 
5ª2
0
input_1%"
input_1ÿÿÿÿÿÿÿÿÿ"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
