
% actions
action(0)::action(defect);
action(1)::action(cooperate).

sensor_value(0)::sensor(defect).
sensor_value(1)::sensor(cooperate).
sensor_value(2)::sensor(none).
sensor_value(3)::sensor(high_val).

unsafe_next :- action(defect), sensor(high_val).
safe_next:- \+unsafe_next.
