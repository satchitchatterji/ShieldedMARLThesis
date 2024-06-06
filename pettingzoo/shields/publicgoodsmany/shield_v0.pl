% actions
action(0)::action(defect);
action(1)::action(cooperate).

sensor_value(0)::sensor(cooperated).
sensor_value(1)::sensor(high_val).

unsafe_next :- action(defect).
safe_next:- \+unsafe_next.