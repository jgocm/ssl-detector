# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ssl_vision_geometry.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x19ssl_vision_geometry.proto\" \n\x08Vector2f\x12\t\n\x01x\x18\x01 \x02(\x02\x12\t\n\x01y\x18\x02 \x02(\x02\"\x88\x01\n\x14SSL_FieldLineSegment\x12\x0c\n\x04name\x18\x01 \x02(\t\x12\x15\n\x02p1\x18\x02 \x02(\x0b\x32\t.Vector2f\x12\x15\n\x02p2\x18\x03 \x02(\x0b\x32\t.Vector2f\x12\x11\n\tthickness\x18\x04 \x02(\x02\x12!\n\x04type\x18\x05 \x01(\x0e\x32\x13.SSL_FieldShapeType\"\x9d\x01\n\x14SSL_FieldCircularArc\x12\x0c\n\x04name\x18\x01 \x02(\t\x12\x19\n\x06\x63\x65nter\x18\x02 \x02(\x0b\x32\t.Vector2f\x12\x0e\n\x06radius\x18\x03 \x02(\x02\x12\n\n\x02\x61\x31\x18\x04 \x02(\x02\x12\n\n\x02\x61\x32\x18\x05 \x02(\x02\x12\x11\n\tthickness\x18\x06 \x02(\x02\x12!\n\x04type\x18\x07 \x01(\x0e\x32\x13.SSL_FieldShapeType\"\xb0\x03\n\x15SSL_GeometryFieldSize\x12\x14\n\x0c\x66ield_length\x18\x01 \x02(\x05\x12\x13\n\x0b\x66ield_width\x18\x02 \x02(\x05\x12\x12\n\ngoal_width\x18\x03 \x02(\x05\x12\x12\n\ngoal_depth\x18\x04 \x02(\x05\x12\x16\n\x0e\x62oundary_width\x18\x05 \x02(\x05\x12*\n\x0b\x66ield_lines\x18\x06 \x03(\x0b\x32\x15.SSL_FieldLineSegment\x12)\n\nfield_arcs\x18\x07 \x03(\x0b\x32\x15.SSL_FieldCircularArc\x12\x1a\n\x12penalty_area_depth\x18\x08 \x01(\x05\x12\x1a\n\x12penalty_area_width\x18\t \x01(\x05\x12\x1c\n\x14\x63\x65nter_circle_radius\x18\n \x01(\x05\x12\x16\n\x0eline_thickness\x18\x0b \x01(\x05\x12#\n\x1bgoal_center_to_penalty_mark\x18\x0c \x01(\x05\x12\x13\n\x0bgoal_height\x18\r \x01(\x05\x12\x13\n\x0b\x62\x61ll_radius\x18\x0e \x01(\x02\x12\x18\n\x10max_robot_radius\x18\x0f \x01(\x02\"\x80\x03\n\x1dSSL_GeometryCameraCalibration\x12\x11\n\tcamera_id\x18\x01 \x02(\r\x12\x14\n\x0c\x66ocal_length\x18\x02 \x02(\x02\x12\x19\n\x11principal_point_x\x18\x03 \x02(\x02\x12\x19\n\x11principal_point_y\x18\x04 \x02(\x02\x12\x12\n\ndistortion\x18\x05 \x02(\x02\x12\n\n\x02q0\x18\x06 \x02(\x02\x12\n\n\x02q1\x18\x07 \x02(\x02\x12\n\n\x02q2\x18\x08 \x02(\x02\x12\n\n\x02q3\x18\t \x02(\x02\x12\n\n\x02tx\x18\n \x02(\x02\x12\n\n\x02ty\x18\x0b \x02(\x02\x12\n\n\x02tz\x18\x0c \x02(\x02\x12\x1f\n\x17\x64\x65rived_camera_world_tx\x18\r \x01(\x02\x12\x1f\n\x17\x64\x65rived_camera_world_ty\x18\x0e \x01(\x02\x12\x1f\n\x17\x64\x65rived_camera_world_tz\x18\x0f \x01(\x02\x12\x19\n\x11pixel_image_width\x18\x10 \x01(\r\x12\x1a\n\x12pixel_image_height\x18\x11 \x01(\r\"V\n\x1dSSL_BallModelStraightTwoPhase\x12\x11\n\tacc_slide\x18\x01 \x02(\x01\x12\x10\n\x08\x61\x63\x63_roll\x18\x02 \x02(\x01\x12\x10\n\x08k_switch\x18\x03 \x02(\x01\"l\n\x1aSSL_BallModelChipFixedLoss\x12\x1c\n\x14\x64\x61mping_xy_first_hop\x18\x01 \x02(\x01\x12\x1d\n\x15\x64\x61mping_xy_other_hops\x18\x02 \x02(\x01\x12\x11\n\tdamping_z\x18\x03 \x02(\x01\"\x86\x01\n\x12SSL_GeometryModels\x12:\n\x12straight_two_phase\x18\x01 \x01(\x0b\x32\x1e.SSL_BallModelStraightTwoPhase\x12\x34\n\x0f\x63hip_fixed_loss\x18\x02 \x01(\x0b\x32\x1b.SSL_BallModelChipFixedLoss\"\x8d\x01\n\x10SSL_GeometryData\x12%\n\x05\x66ield\x18\x01 \x02(\x0b\x32\x16.SSL_GeometryFieldSize\x12-\n\x05\x63\x61lib\x18\x02 \x03(\x0b\x32\x1e.SSL_GeometryCameraCalibration\x12#\n\x06models\x18\x03 \x01(\x0b\x32\x13.SSL_GeometryModels*\xdb\x02\n\x12SSL_FieldShapeType\x12\r\n\tUndefined\x10\x00\x12\x10\n\x0c\x43\x65nterCircle\x10\x01\x12\x10\n\x0cTopTouchLine\x10\x02\x12\x13\n\x0f\x42ottomTouchLine\x10\x03\x12\x10\n\x0cLeftGoalLine\x10\x04\x12\x11\n\rRightGoalLine\x10\x05\x12\x0f\n\x0bHalfwayLine\x10\x06\x12\x0e\n\nCenterLine\x10\x07\x12\x16\n\x12LeftPenaltyStretch\x10\x08\x12\x17\n\x13RightPenaltyStretch\x10\t\x12\x1f\n\x1bLeftFieldLeftPenaltyStretch\x10\n\x12 \n\x1cLeftFieldRightPenaltyStretch\x10\x0b\x12 \n\x1cRightFieldLeftPenaltyStretch\x10\x0c\x12!\n\x1dRightFieldRightPenaltyStretch\x10\r')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'ssl_vision_geometry_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _SSL_FIELDSHAPETYPE._serialized_start=1664
  _SSL_FIELDSHAPETYPE._serialized_end=2011
  _VECTOR2F._serialized_start=29
  _VECTOR2F._serialized_end=61
  _SSL_FIELDLINESEGMENT._serialized_start=64
  _SSL_FIELDLINESEGMENT._serialized_end=200
  _SSL_FIELDCIRCULARARC._serialized_start=203
  _SSL_FIELDCIRCULARARC._serialized_end=360
  _SSL_GEOMETRYFIELDSIZE._serialized_start=363
  _SSL_GEOMETRYFIELDSIZE._serialized_end=795
  _SSL_GEOMETRYCAMERACALIBRATION._serialized_start=798
  _SSL_GEOMETRYCAMERACALIBRATION._serialized_end=1182
  _SSL_BALLMODELSTRAIGHTTWOPHASE._serialized_start=1184
  _SSL_BALLMODELSTRAIGHTTWOPHASE._serialized_end=1270
  _SSL_BALLMODELCHIPFIXEDLOSS._serialized_start=1272
  _SSL_BALLMODELCHIPFIXEDLOSS._serialized_end=1380
  _SSL_GEOMETRYMODELS._serialized_start=1383
  _SSL_GEOMETRYMODELS._serialized_end=1517
  _SSL_GEOMETRYDATA._serialized_start=1520
  _SSL_GEOMETRYDATA._serialized_end=1661
# @@protoc_insertion_point(module_scope)
