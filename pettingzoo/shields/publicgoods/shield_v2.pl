% actions
action(0)::action(defect);
action(1)::action(cooperate).

sensor_value(0)::sensor(defect).
sensor_value(1)::sensor(cooperate).
sensor_value(2)::sensor(none).
sensor_value(3)::sensor(mult_unc).
sensor_value(4)::sensor(mult_high).

unsafe_next :- \+action(cooperate).
safe_next:- \+unsafe_next.
