% actions
action(0)::action(defect);
action(1)::action(cooperate).

sensor_value(0)::sensor(defect).
sensor_value(1)::sensor(cooperate).
sensor_value(2)::sensor(none).
sensor_value(3)::sensor(mult_cert).
sensor_value(4)::sensor(mult_high).

unsafe_next :- \+action(cooperate), sensor(mult_high), sensor(mult_cert).
unsafe_next :- \+action(defect), \+sensor(mult_high), sensor(mult_cert).

% unsafe_next :- \+action(cooperate), sensor(mult_high), sensor(mult_cert).
% unsafe_next :- action(defect), sensor(mult_high), sensor(mult_cert).
safe_next:- \+unsafe_next.
