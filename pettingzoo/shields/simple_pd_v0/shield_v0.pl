% actions
action(0)::action(cooperate);
action(1)::action(defect).

sensor_value(0)::sensor(cooperate).
sensor_value(1)::sensor(defect).

unsafe_next :- action(defect).
safe_next:- \+unsafe_next.
