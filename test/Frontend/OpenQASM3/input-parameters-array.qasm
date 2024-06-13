OPENQASM 3;
// RUN: qss-compiler -X=qasm --enable-parameters --enable-circuits-from-qasm %s | FileCheck %s --check-prefix MLIR
// RUN: qss-compiler -X=qasm --emit=ast-pretty --enable-parameters --enable-circuits-from-qasm %s | FileCheck %s --check-prefix AST-PRETTY


gate rz(theta) q {}

qubit $0;

input angle angle_p;
input float[64] float_p;

input array[angle, 10] array_angle_p;
// AST-PRETTY: DeclarationNode(type=ASTTypeAngleArray, ArrayNode(name=array_angle_p, size=10, elementType=ASTTypeAngle, elementSize=1), inputVariable)

input array[float[64], 10] array_float_p;
// AST-PRETTY: DeclarationNode(type=ASTTypeMPDecimalArray, ArrayNode(name=array_float_p, size=10, elementType=ASTTypeMPDecimal, elementSize=64), inputVariable)


// Access value directly
rz(angle_p) $0;
rz(float_p) $0;

// Access value through array reference
rz(array_angle_p[0]) $0;
// AST-PRETTY: GateGenericNode(name=rz, params=[AngleNode(value=IdentifierNode(name=array_angle_p[0], index=0), bits=64)], qubits=[], qcparams=[$0])

rz(array_angle_p[1]) $0;
// AST-PRETTY: GateGenericNode(name=rz, params=[AngleNode(value=IdentifierNode(name=array_angle_p[1], index=1), bits=64)], qubits=[], qcparams=[$0])

rz(array_float_p[0]) $0;
// AST-PRETTY: GateGenericNode(name=rz, params=[AngleNode(value=IdentifierNode(name=array_float_p[0], index=0), bits=64)], qubits=[], qcparams=[$0])

rz(array_float_p[1]) $0;
// AST-PRETTY: GateGenericNode(name=rz, params=[AngleNode(value=IdentifierNode(name=array_float_p[1], index=1), bits=64)], qubits=[], qcparams=[$0])

// Access value through intermediate assignment
float[64] intermediate_angle = array_angle_p[0];
float[64] intermediate_float = array_float_p[0];
rz(intermediate_angle) $0;
rz(intermediate_float) $0;

bit final = measure $0;
