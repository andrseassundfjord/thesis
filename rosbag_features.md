Categorical:

/driver/blink (I can't find related information)
/driver/eye_open {0: undetected,1: open ,2: close}
/vehicle/analog/back_signal {0: off, 1: on}
/vehicle/analog/brake_signal {0: off, 1: on}
/vehicle/analog/turn_signal  {0: off, 1: left, 2: right, 3: hazard lights}
/vehicle/extracted_can/headlight_state {0: off, 1: clearance lamp, 2: low beam, 3: high beam}
/vehicle/extracted_can/seat_belt_state {0: fasten, 1: unfasten}

Continous:

/driver/eye_opening_rate [-12.8:38.3] (0.1 mm)
/driver/face_{pitch, roll, yaw} [-128:127] (1 deg)
/driver/gaze_{horizontal, vertical} [-128:127] (1 deg)
/driver/heart_rate [30:220] (1 bpm)
/vehicle/acceleration [-19.6, 19.6] (0.1 m/s^2)
/vehicle/analog/speed_pulse
/vehicle/extracted_can/brake_pedal_hydraulic [0:depend on vehicle] (Mpa)
/vehicle/extracted_can/gas_pedal_position [0:100] (%)
/vehicle/extracted_can/speed [0:depend on vehicle] (km/h)
/vehicle/extracted_can/steering_angle [depend on vehicle] (deg)

Continuous:

/gps_m2/distance/from_last_intersection
/gps_m2/distance/to_next_intersection
/gps_m2/road/curvature
/gps_m2/vehicle/acceleration
/gps_m2/vehicle/direction
/gps_m2/vehicle/gnss
/gps_m2/vehicle/gnss_reliability
/gps_m2/vehicle/pitch_angle
/gps_m2/vehicle/speed
/gps_m2/vehicle/yaw_rate

Categorical:

/gps_m2/link_id
/gps_m2/road/intersection
/gps_m2/road/lanes
/gps_m2/road/traffic_light
/gps_m2/road/type
/gps_m2/road/width
/gps_m2/vehicle/back_state
/gps_m2/vehicle/turn_state